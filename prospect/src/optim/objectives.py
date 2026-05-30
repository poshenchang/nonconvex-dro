import functools
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted, get_wasserstein_weights


# ---------------------------------------------------------------------------
# Linear loss functions (unchanged)
# ---------------------------------------------------------------------------

def squared_error_loss(w, X, y):
    return 0.5 * (y - torch.matmul(X, w)) ** 2

def squared_error_gradient(w, X, y):
    return (torch.matmul(X, w) - y)[:, None] * X

def binary_cross_entropy_loss(w, X, y):
    logits = torch.matmul(X, w)
    return F.binary_cross_entropy_with_logits(logits, y, reduction="none")

def binary_cross_entropy_gradient(w, X, y):
    logits = torch.matmul(X, w)
    p = 1. / (1. + torch.exp(-logits))
    return (p - y)[:, None] * X

def multinomial_cross_entropy_loss(w, X, y, n_class):
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    return F.cross_entropy(logits, y, reduction="none")

def multinomial_cross_entropy_gradient(w, X, y, n_class):
    n = len(X)
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    p = torch.softmax(logits, dim=1)
    p[torch.arange(n), y] -= 1
    scores = torch.bmm(X[:, :, None], p[:, None, :])
    return scores.view(n, -1)


# ---------------------------------------------------------------------------
# MLP loss function (non-linear, uses autograd for gradients)
# ---------------------------------------------------------------------------

def mlp_binary_cross_entropy_loss(w, X, y, hidden_dim=128):
    """
    One-hidden-layer MLP for binary classification.
    w is a flat 1-D tensor: [W1 | b1 | W2 | b2]
      W1: (d, hidden_dim)
      b1: (hidden_dim,)
      W2: (hidden_dim, 1)
      b2: scalar
    """
    d = X.shape[1]
    w1_end = d * hidden_dim
    b1_end = w1_end + hidden_dim
    w2_end = b1_end + hidden_dim

    W1 = w[:w1_end].view(d, hidden_dim)
    b1 = w[w1_end:b1_end]
    W2 = w[b1_end:w2_end].view(hidden_dim, 1)
    b2 = w[-1]

    h = F.relu(torch.matmul(X, W1) + b1)
    logits = torch.matmul(h, W2).squeeze(-1) + b2

    return F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")


def mlp_num_parameters(d, hidden_dim=128):
    """Total parameters for the MLP: W1 + b1 + W2 + b2."""
    return d * hidden_dim + hidden_dim + hidden_dim + 1


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_loss(name, n_class=None):
    if name == "squared_error":
        return squared_error_loss
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_loss
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_loss(w, X, y, n_class)
    elif name == "mlp_binary_cross_entropy":
        return mlp_binary_cross_entropy_loss          # uses autograd; no closed-form grad
    else:
        raise ValueError(
            f"Unrecognized loss '{name}'! Options: ['squared_error', "
            "'binary_cross_entropy', 'multinomial_cross_entropy', 'mlp_binary_cross_entropy']"
        )

def get_grad_batch(name, n_class=None):
    if name == "squared_error":
        return squared_error_gradient
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_gradient
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_gradient(w, X, y, n_class)
    elif name == "mlp_binary_cross_entropy":
        return None                                   # signal to fall back to autograd
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Objective class
# ---------------------------------------------------------------------------

class Objective:
    def __init__(
        self,
        X,
        y,
        weight_function,
        loss="squared_error",
        l2_reg=None,
        n_class=None,
        risk_name=None,
        dataset=None,
        shift_cost=1.0,
        penalty=None,
        autodiff=True,
        distance_metric="euclidean",
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.weight_function = weight_function
        self.loss = get_loss(loss, n_class=n_class)
        self.grad_batch = get_grad_batch(loss, n_class=n_class)
        self.loss_name = loss
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.autodiff = autodiff

        self.risk_name = risk_name
        self.dataset = dataset

        self.sigmas = weight_function(self.n)
        self.shift_cost = self.n * shift_cost if penalty in ["l2", "wasserstein"] else shift_cost
        self.penalty = penalty
        self.distance_metric = distance_metric

        # ------------------------------------------------------------------
        # Parameter count: flat for MLP, feature-dim for linear models
        # ------------------------------------------------------------------
        if loss == "mlp_binary_cross_entropy":
            self.num_parameters = mlp_num_parameters(self.d)
        elif n_class:
            self.num_parameters = self.d * n_class
        else:
            self.num_parameters = self.d

        if self.penalty == "wasserstein":
            if self.distance_metric == "euclidean":
                self.C = torch.cdist(X.double(), X.double(), p=2.0)
            elif self.distance_metric == "cosine":
                X_normalized = F.normalize(X.double(), p=2.0, dim=1)
                self.C = 1.0 - torch.matmul(X_normalized, X_normalized.T)
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")
            self.K = torch.exp(-self.C / 0.1)
        else:
            self.C = None
            self.K = None

    # ------------------------------------------------------------------
    # Batch loss (for logging / evaluation)
    # ------------------------------------------------------------------

    def _raw_losses(self, w, X, y):
        """Cast per-sample losses to float64 to match sigmas dtype (avoids Double/Float dot errors)."""
        return self.loss(w, X, y).double()

    def get_batch_loss(self, w, include_reg=True):
        with torch.no_grad():
            n = self.n
            if self.penalty == "wasserstein":
                losses = self._raw_losses(w, self.X, self.y)
                epsilon = 0.1
                losses_shifted = losses - torch.max(losses)
                v = torch.exp(losses_shifted / (self.shift_cost * epsilon))
                w_val = torch.matmul(self.K.T, v)
                log_sum_exp = torch.log(w_val + 1e-16) + torch.max(losses) / (self.shift_cost * epsilon)
                risk = (self.shift_cost * epsilon / n) * torch.sum(log_sum_exp + math.log(n))
            elif self.l2_reg:
                sorted_losses = torch.sort(self._raw_losses(w, self.X, self.y), stable=True)[0]
                sm_sigmas = get_smooth_weights_sorted(
                    sorted_losses, self.sigmas, self.shift_cost, self.penalty
                )
                risk = torch.dot(sm_sigmas, sorted_losses) - 0.5 * self.shift_cost * torch.sum((sm_sigmas - 1 / n) ** 2)
            else:
                sorted_losses = torch.sort(self._raw_losses(w, self.X, self.y), stable=True)[0]
                risk = torch.dot(self.sigmas, sorted_losses)
            if self.l2_reg and include_reg:
                risk += 0.5 * self.l2_reg * torch.norm(w) ** 2 / self.n
            return risk

    # ------------------------------------------------------------------
    # Batch subgradient
    # ------------------------------------------------------------------

    def get_batch_subgrad(self, w, idx=None, include_reg=True):
        if self.autodiff:
            return self.get_batch_subgrad_autodiff(w, idx=idx, include_reg=include_reg)
        else:
            return self.get_batch_subgrad_oracle(w, idx=idx, include_reg=include_reg)

    @torch.no_grad()
    def get_batch_subgrad_oracle(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            sigmas = self.weight_function(len(X))
        else:
            X, y = self.X, self.y
            sigmas = self.sigmas

        if self.penalty == "wasserstein":
            losses = self._raw_losses(w, X, y)
            C_sub = self.C[idx][:, idx] if idx is not None else self.C
            K_sub = self.K[idx][:, idx] if idx is not None else self.K
            q = get_wasserstein_weights(losses, C_sub, self.shift_cost, epsilon=0.1, K=K_sub)
            g = torch.matmul(q, self.grad_batch(w, X, y))
        else:
            sorted_losses, perm = torch.sort(self._raw_losses(w, X, y), stable=True)
            q = get_smooth_weights_sorted(sorted_losses, sigmas, self.shift_cost, self.penalty) if self.penalty else sigmas
            g = torch.matmul(q, self.grad_batch(w, X, y)[perm])
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach() / self.n
        return g

    def get_batch_subgrad_autodiff(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            sigmas = self.weight_function(len(X))
        else:
            X, y = self.X, self.y
            sigmas = self.sigmas

        if self.penalty == "wasserstein":
            losses = self.loss(w, X, y).double()
            C_sub = self.C[idx][:, idx] if idx is not None else self.C
            K_sub = self.K[idx][:, idx] if idx is not None else self.K
            with torch.no_grad():
                sm_sigmas = get_wasserstein_weights(losses, C_sub, self.shift_cost, epsilon=0.1, K=K_sub)
            risk = torch.dot(sm_sigmas, losses)
        else:
            sorted_losses = torch.sort(self.loss(w, X, y).double(), stable=True)[0]
            if self.l2_reg:
                with torch.no_grad():
                    sm_sigmas = get_smooth_weights_sorted(sorted_losses, sigmas, self.shift_cost, self.penalty)
                risk = torch.dot(sm_sigmas, sorted_losses)
            else:
                risk = torch.dot(sigmas, sorted_losses)
        g = torch.autograd.grad(outputs=risk, inputs=w)[0]
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach() / self.n
        return g

    # ------------------------------------------------------------------
    # Individual losses
    # ------------------------------------------------------------------

    def get_indiv_loss(self, w, with_grad=False):
        """Always returns float64 so the loss table in Prospect stays float64."""
        if with_grad:
            return self.loss(w, self.X, self.y).double()
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y).double()

    # ------------------------------------------------------------------
    # Individual gradients — falls back to autograd for MLP
    # ------------------------------------------------------------------

    def get_indiv_grad(self, w, X=None, y=None):
        """
        Return per-sample gradients.
        - Linear models: closed-form grad_batch (runs under no_grad).
        - MLP (grad_batch is None): autograd — must NOT be wrapped in no_grad.
        """
        if self.grad_batch is not None:
            # Closed-form gradient; safe to disable grad tracking
            with torch.no_grad():
                if X is not None:
                    return self.grad_batch(w, X, y)
                else:
                    return self.grad_batch(w, self.X, self.y)
        else:
            # Autograd fallback — grad must be enabled for the forward pass
            if X is not None:
                return self._autograd_grad(w, X, y)
            else:
                return self._autograd_grad_all(w)

    def _autograd_grad(self, w, X, y):
        """
        Gradient of sum(losses) w.r.t. w for a mini-batch.
        Runs with grad enabled regardless of outer context.
        """
        with torch.enable_grad():
            w_ = w.detach().requires_grad_(True)
            losses = self.loss(w_, X, y)
            g = torch.autograd.grad(losses.sum(), w_)[0]
        return g.detach()

    def _autograd_grad_all(self, w):
        """
        Per-sample gradients for the full dataset, shape (n, num_parameters).
        Uses functorch.grad + vmap when available, falls back to an explicit loop.
        Runs with grad enabled regardless of outer context.
        """
        with torch.enable_grad():
            w_ = w.detach().requires_grad_(True)

            try:
                # vmap over per-sample scalar losses (PyTorch >= 2.0)
                def per_sample_grad(x, y):
                    loss = self.loss(w_, x.unsqueeze(0), y.unsqueeze(0)).squeeze()
                    return torch.autograd.grad(loss, w_, create_graph=False)[0]

                grads = torch.vmap(per_sample_grad)(self.X, self.y)
            except Exception:
                # Explicit loop fallback
                grads = torch.zeros(
                    self.n, w_.shape[0], dtype=w_.dtype, device=w_.device
                )
                for i in range(self.n):
                    loss_i = self.loss(
                        w_, self.X[i:i+1], self.y[i:i+1]
                    ).squeeze()
                    grads[i] = torch.autograd.grad(
                        loss_i, w_, retain_graph=True
                    )[0].detach()

        return grads.detach()

    # ------------------------------------------------------------------
    # Proximal operators (linear models only)
    # ------------------------------------------------------------------

    def get_indiv_prox_loss(self, w, stepsize, i):
        """
        Closed-form prox step; not applicable to MLP — raises NotImplementedError.
        Use oracle_reg='grad' in Prospect when loss_name is 'mlp_binary_cross_entropy'.
        """
        real_l2_reg = self.l2_reg / self.n
        with torch.no_grad():
            if self.loss_name == "squared_error":
                prox_op = prox_squared_loss
            elif self.loss_name == "multinomial_cross_entropy":
                prox_op = functools.partial(prox_multinomial_log_loss_vec, n_class=self.n_class)
            elif self.loss_name == "binary_cross_entropy":
                prox_op = prox_binary_log_loss
            elif self.loss_name == "mlp_binary_cross_entropy":
                raise NotImplementedError(
                    "Proximal operator is not defined for MLP. "
                    "Use oracle_reg='grad' in Prospect."
                )
            else:
                raise NotImplementedError(f"Prox not implemented for {self.loss_name}")
            return prox_with_l2reg(w, self.X[i], self.y[i], stepsize, prox_op, real_l2_reg)

    def get_indiv_mor_grad(self, w, stepsize, i):
        with torch.no_grad():
            return (w - self.get_indiv_prox_loss(w, stepsize, i)) / stepsize

    def get_model_cfg(self):
        return {
            "objective": self.risk_name,
            "l2_reg": self.l2_reg,
            "loss": self.loss_name,
            "n_class": self.n_class,
            "shift_cost": self.shift_cost / self.n if self.penalty == "l2" else self.shift_cost,
        }


# ---------------------------------------------------------------------------
# Proximal operator helpers (unchanged)
# ---------------------------------------------------------------------------

def prox_squared_loss(w, x, y, stepsize):
    scaling = stepsize / (1 + stepsize * torch.sum(x**2))
    return w - scaling * (x.squeeze().dot(w) - y) * x.squeeze()

def prox_with_l2reg(w, x, y, stepsize, prox, l2reg):
    scaled_stepsize = stepsize / (1 + stepsize * l2reg)
    scaled_w = w / (1 + stepsize * l2reg)
    return prox(scaled_w, x, y, scaled_stepsize)

def prox_multinomial_log_loss(W, x, y, stepsize, n_class):
    logits = W.mv(x.squeeze())
    sigma = torch.nn.functional.softmax(logits, dim=0)
    y_hot = torch.nn.functional.one_hot(y, n_class).squeeze()
    z_3 = torch.ones(n_class) + stepsize * torch.sum(x**2) * sigma
    z_2 = sigma / z_3
    z_1 = -y_hot / z_3 + z_2
    lam = torch.sum(z_1) / torch.sum(z_2)
    z = z_1 - lam * z_2
    return W - stepsize * torch.outer(z, x.squeeze())

def prox_binary_log_loss(w, x, y, stepsize):
    logit = w.dot(x.squeeze())
    g = -y + torch.nn.functional.sigmoid(logit)
    q = 1 / (2 * (torch.cosh(logit) + 1))
    scaling = 1 / (1 + stepsize * q * torch.sum(x**2))
    return w - stepsize * scaling * g * x.squeeze()

def prox_multinomial_log_loss_vec(w, x, y, stepsize, n_class):
    W = w.view(-1, n_class).t()
    sol = prox_multinomial_log_loss(W, x, y, stepsize, n_class)
    return sol.t().view(-1)


# ---------------------------------------------------------------------------
# Risk weight helpers (unchanged)
# ---------------------------------------------------------------------------

def get_erm_weights(n):
    return torch.ones(n, dtype=torch.float64) / n

def get_extremile_weights(n, r):
    return (
        (torch.arange(n, dtype=torch.float64) + 1) ** r
        - torch.arange(n, dtype=torch.float64) ** r
    ) / (n**r)

def get_superquantile_weights(n, q):
    weights = torch.zeros(n, dtype=torch.float64)
    idx = math.floor(n * q)
    frac = 1 - (n - idx - 1) / (n * (1 - q))
    if frac > 1e-12:
        weights[idx] = frac
        weights[(idx + 1):] = 1 / (n * (1 - q))
    else:
        weights[idx:] = 1 / (n - idx)
    return weights

def get_esrm_weights(n, rho):
    upper = torch.exp(rho * ((torch.arange(n, dtype=torch.float64) + 1) / n))
    lower = torch.exp(rho * (torch.arange(n, dtype=torch.float64) / n))
    return math.exp(-rho) * (upper - lower) / (1 - math.exp(-rho))


# ---------------------------------------------------------------------------
# Tests (unchanged)
# ---------------------------------------------------------------------------

def test_prox_multinomial_log_loss():
    d, k, stepsize = 8, 4, 1.0
    torch.manual_seed(0)
    x = torch.randn((1, d))
    y = torch.randint(0, k, (1,))
    w = torch.randn((d * k), requires_grad=True)
    v = prox_multinomial_log_loss_vec(w, x, y, stepsize, k) - w
    func = functools.partial(multinomial_cross_entropy_loss, X=x, y=y, n_class=k)
    hvp = torch.autograd.functional.hvp(func, w, v)[1]
    grad = torch.autograd.grad(func(w), w)[0]
    err = torch.linalg.norm(stepsize * hvp + stepsize * grad + v)
    assert err < 1e-6, f"First-order optimality violated by {err}"

def test_prox_binary_log_loss():
    d, stepsize = 8, 1.0
    torch.manual_seed(0)
    x = torch.randn((1, d))
    y = torch.randint(0, 2, (1,))
    w = torch.randn((d), requires_grad=True)
    v = prox_binary_log_loss(w, x, y, stepsize) - w
    func = functools.partial(binary_cross_entropy_loss, X=x, y=y)
    hvp = torch.autograd.functional.hvp(func, w, v)[1]
    grad = torch.autograd.grad(func(w), w)[0]
    err = torch.linalg.norm(stepsize * hvp + stepsize * grad + v)
    assert err < 1e-6, f"First-order optimality violated by {err}"

if __name__ == "__main__":
    test_prox_multinomial_log_loss()
    test_prox_binary_log_loss()