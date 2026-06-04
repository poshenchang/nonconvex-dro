import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_wasserstein_weights
from src.optim.baselines import Optimizer
from numba import jit
import warnings



_MLP_LOSSES = {"mlp_binary_cross_entropy", "mlp_squared_error", "mlp_multinomial_cross_entropy"}


def _kaiming_init(num_params, objective, seed=0):
    """
    Kaiming (He) initialization for a flat MLP weight vector.
    For non-MLP losses, falls back to zeros (linear models are convex,
    so initialization does not affect convergence qualitatively).
    W ~ N(0, sqrt(2 / fan_in)) per layer; biases set to zero.
    """
    if getattr(objective, 'loss_name', None) not in _MLP_LOSSES:
        return torch.zeros(num_params, dtype=torch.float64)

    rng = torch.Generator()
    rng.manual_seed(seed)
    d          = objective.d
    hidden_dim = getattr(objective, 'hidden_dim', 128)
    n_class    = getattr(objective, 'n_class', None)
    w          = torch.zeros(num_params, dtype=torch.float64)

    # W1: (d, hidden_dim)  fan_in = d
    w1_end = d * hidden_dim
    w[:w1_end] = torch.randn(w1_end, generator=rng, dtype=torch.float64) * (2.0 / d) ** 0.5
    # b1: zeros (already zero)
    b1_end = w1_end + hidden_dim

    if n_class is not None and n_class > 1:
        # Multinomial head: W2 (hidden_dim, n_class), b2 (n_class,)
        w2_size = hidden_dim * n_class
        w[b1_end:b1_end + w2_size] = (
            torch.randn(w2_size, generator=rng, dtype=torch.float64)
            * (2.0 / hidden_dim) ** 0.5
        )
    else:
        # Binary / regression head: W2 (hidden_dim, 1), b2 scalar
        w2_end = b1_end + hidden_dim
        w[b1_end:w2_end] = (
            torch.randn(hidden_dim, generator=rng, dtype=torch.float64)
            * (2.0 / hidden_dim) ** 0.5
        )
    return w


class Prospect(Optimizer):
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        oracle_reg="prox",
    ):
        super(Prospect, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = 1.0 if lrd is None else lrd
        n = self.objective.n

        # ------------------------------------------------------------------
        # Use num_parameters from Objective (handles MLP and linear models).
        # Falls back to d or n_class*d for backwards compatibility.
        # ------------------------------------------------------------------
        num_params = getattr(
            self.objective,
            "num_parameters",
            objective.n_class * objective.d if objective.n_class else objective.d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.grad_table = torch.zeros(n, num_params, dtype=torch.float64)

        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty
        assert oracle_reg in ["prox", "grad"]
        self.oracle_reg = oracle_reg

        # Precompute Wasserstein cost / kernel if needed
        if self.penalty == "wasserstein":
            self.C = getattr(self.objective, "C", None)
            if self.C is None:
                X = self.objective.X
                self.C = torch.cdist(X.double(), X.double(), p=2.0)
            self.K = getattr(self.objective, "K", None)
            if self.K is None:
                self.K = torch.exp(-self.C / 0.1)
        else:
            self.C = None
            self.K = None

        # Build initial loss / dual-weight / gradient tables
        self.losses = self.objective.get_indiv_loss(self.weights).detach()
        if self.penalty == "wasserstein":
            self.lam = get_wasserstein_weights(
                self.losses, self.C, self.shift_cost, epsilon=0.1, K=self.K
            )
        else:
            self.lam = get_smooth_weights(
                self.losses, self.sigmas, self.shift_cost, self.penalty
            )
        self.rho = self.lam.clone()
        real_l2_reg = self.objective.l2_reg / n

        if self.oracle_reg == "grad":
            # Include L2 reg in the stored gradient for gradient-mapping oracle
            self.grad_table = (
                self.objective.get_indiv_grad(self.weights)
                + real_l2_reg * self.weights[None, :]
            )
        else:
            self.grad_table = self.objective.get_indiv_grad(self.weights)

        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)

        self.epoch_len = epoch_len if epoch_len else self.objective.n

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_l2_reg = self.objective.l2_reg / n

        # Sample a random index and compute current gradient
        i = torch.tensor([self.rng_grad.randint(0, n)])
        x = self.objective.X[i]
        y = self.objective.y[i]
        loss = self.objective.loss(self.weights, x, y)
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        if self.oracle_reg == "grad":
            g = g + real_l2_reg * self.weights

        # Retrieve stale gradient from table
        g_old = self.grad_table[i].reshape(-1)

        # SVRG-style variance-reduced direction
        v = n * self.lam[i] * g - n * self.rho[i] * g_old + self.running_subgrad

        # Primal update: proximal (linear) or gradient step (MLP / non-convex)
        if self.oracle_reg == "prox":
            self.weights.copy_(
                1 / (self.lrp * real_l2_reg + 1) * (self.weights - self.lrp * v)
            )
        else:
            self.weights.copy_(self.weights - self.lrp * v)

        # Update loss and dual weights
        self.losses[i] = loss.detach().double()  # cast to float64 to match loss table dtype
        if self.penalty == "wasserstein":
            self.lam = get_wasserstein_weights(
                self.losses, self.C, self.shift_cost, epsilon=0.1, K=self.K
            )
        else:
            self.lam = get_smooth_weights(
                self.losses, self.sigmas, self.shift_cost, self.penalty
            )
        cur_lam = self.lam[i]
        rho_old = self.rho[i]
        self.rho[i] = cur_lam

        # Update gradient table and running subgradient
        self.grad_table[i] = g[None, :]
        self.running_subgrad += cur_lam * g - rho_old * g_old

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class ProspectMoreau(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        device="cpu",
    ):
        super(ProspectMoreau, self).__init__()
        self.objective = objective
        self.lr = lr
        n = self.objective.n

        if not self.objective.autodiff:
            warnings.warn("ProspectMoreau does not currently have a non-autodiff implementation")

        # Use num_parameters to support MLP weight vectors
        num_params = getattr(
            self.objective,
            "num_parameters",
            objective.n_class * objective.d if objective.n_class else objective.d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.grad_table = torch.zeros(n, num_params, dtype=torch.float64)

        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty

        if self.penalty == "wasserstein":
            self.C = getattr(self.objective, "C", None)
            if self.C is None:
                X = self.objective.X
                self.C = torch.cdist(X.double(), X.double(), p=2.0)
            self.K = getattr(self.objective, "K", None)
            if self.K is None:
                self.K = torch.exp(-self.C / 0.1)
        else:
            self.C = None
            self.K = None

        self.losses = self.objective.get_indiv_loss(self.weights)
        if self.penalty == "wasserstein":
            self.lam = get_wasserstein_weights(
                self.losses.detach(), self.C, self.shift_cost, epsilon=0.1, K=self.K
            )
        else:
            self.lam = get_smooth_weights(
                self.losses, self.sigmas, self.shift_cost, self.penalty
            )

        for i in range(n):
            loss = self.objective.loss(
                self.weights, self.objective.X[i, :], self.objective.y[i]
            )
            self.grad_table[i] = (
                torch.autograd.grad(outputs=loss, inputs=self.weights)[0]
                + self.objective.l2_reg * self.weights / n
            )

        self.running_subgrad = torch.matmul(self.grad_table.T, self.lam).cpu()

        self.epoch_len = epoch_len if epoch_len else self.objective.n

    def start_epoch(self):
        pass

    def step(self):
        n = self.objective.n

        p = torch.abs(self.lam) / (torch.sum(abs(self.lam)))
        i = torch.tensor([np.random.choice(n, p=p)])
        g_old_i = self.grad_table[i].reshape(-1).cpu()

        j = torch.tensor([self.rng_grad.randint(0, n)])
        x_j = self.objective.X[j]
        y_j = self.objective.y[j]
        loss_j = self.objective.loss(self.weights, x_j, y_j)
        g_old_j = self.grad_table[j].reshape(-1).cpu()

        v_weights = self.weights + self.lr * (g_old_i - self.running_subgrad)
        v_table = self.weights + self.lr * (g_old_j - self.running_subgrad)

        self.weights = self.objective.get_indiv_prox_loss(v_weights, self.lr, i)

        mor_grad_j = self.objective.get_indiv_mor_grad(v_table, self.lr, j)
        self.grad_table[j] = mor_grad_j.reshape(1, -1)

        self.losses[j] = loss_j.detach().double()  # cast to float64 to match loss table dtype
        if self.penalty == "wasserstein":
            self.lam = get_wasserstein_weights(
                self.losses.detach(), self.C, self.shift_cost, epsilon=0.1, K=self.K
            )
        else:
            self.lam = get_smooth_weights(
                self.losses.detach(), self.sigmas, self.shift_cost, self.penalty
            )

        self.running_subgrad = torch.matmul(self.lam, self.grad_table).cpu()

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


# ---------------------------------------------------------------------------
# Numba-accelerated bubble sort for maintaining sorted loss tables
# ---------------------------------------------------------------------------

@jit(nopython=True)
def bubble_sort(idx, sort, argsort):
    n = len(sort)

    # Bubble left
    j = idx
    while j > 0 and sort[j] < sort[j - 1] - 1e-10:
        sort[j], sort[j - 1] = sort[j - 1], sort[j]
        argsort[j], argsort[j - 1] = argsort[j - 1], argsort[j]
        j -= 1

    # Bubble right
    j = idx
    while j < n - 1 and sort[j] > sort[j + 1] + 1e-10:
        sort[j], sort[j + 1] = sort[j + 1], sort[j]
        argsort[j], argsort[j + 1] = argsort[j + 1], argsort[j]
        j += 1