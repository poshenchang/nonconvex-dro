import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted



def _kaiming_init(num_params, objective, seed=0):
    """
    Kaiming (He) initialization for a flat MLP weight vector.
    For non-MLP losses, falls back to zeros (linear models are convex,
    so initialization does not affect convergence qualitatively).
    W ~ N(0, sqrt(2 / fan_in)) per layer; biases set to zero.
    """
    if not getattr(objective, 'autodiff', False):
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


class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError


class SubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01):
        super(SubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr

        num_params = getattr(
            objective,
            "num_parameters",
            objective.n_class * objective.d if objective.n_class else objective.d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)

    def start_epoch(self):
        pass

    def step(self):
        g = self.objective.get_batch_subgrad(self.weights)
        self.weights = self.weights.detach() - self.lr * g
        self.weights.requires_grad_(True)

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return 1


class StochasticSubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None):
        super(StochasticSubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size

        num_params = getattr(
            objective,
            "num_parameters",
            objective.n_class * objective.d if objective.n_class else objective.d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.order = None
        self.iter = None
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0

    def step(self):
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad_(True)
        g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        self.weights = self.weights.detach() - self.lr * g
        self.weights.requires_grad_(True)
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class StochasticRegularizedDualAveraging(Optimizer):
    def __init__(
        self, objective, lr=0.01, l2_reg=1.0, batch_size=64, seed=25, epoch_len=None
    ):
        super(StochasticRegularizedDualAveraging, self).__init__()
        self.objective = objective
        self.aux_reg = 1 / lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        num_params = getattr(
            objective,
            "num_parameters",
            objective.n_class * objective.d if objective.n_class else objective.d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.dual_avg = torch.zeros(num_params, dtype=torch.float64)

        self.order = None
        self.epoch_iter = None
        self.total_iter = 0
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.epoch_iter = 0

    def step(self):
        idx = self.order[
            self.epoch_iter
            * self.batch_size : min(
                self.objective.n, (self.epoch_iter + 1) * self.batch_size
            )
        ]
        g = self.objective.get_batch_subgrad(self.weights, idx=idx, include_reg=False)
        self.dual_avg = (self.total_iter * self.dual_avg + g) / (self.total_iter + 1)
        self.weights = -self.dual_avg / (
            self.l2_reg / self.objective.n + self.aux_reg / (self.total_iter + 1)
        )
        self.weights = self.weights.detach().requires_grad_(True)
        self.epoch_iter += 1
        self.total_iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class SmoothedLSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
    ):
        super(SmoothedLSVRG, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        num_params = getattr(
            objective,
            "num_parameters",
            objective.n_class * d if objective.n_class else d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0
        self.step_no = 0

    def start_epoch(self):
        pass

    def step(self):
        n = self.objective.n

        # start epoch
        if self.step_no % n == 0:
            with torch.no_grad():
                losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
                sorted_losses, self.argsort = torch.sort(losses, stable=True)
                self.sigmas = get_smooth_weights_sorted(
                    sorted_losses, self.spectrum, self.smooth_coef, self.smoothing
                )
            with torch.enable_grad():
                self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)
            self.weights_checkpt = self.weights.detach().clone()
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[self.argsort[i]]
        y = self.objective.y[self.argsort[i]]

        # get_indiv_grad handles enable_grad internally for MLP
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction = direction + self.objective.l2_reg * self.weights.detach() / n

        self.weights = (self.weights.detach() - self.lr * direction).requires_grad_(True)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch


class SaddleSAGA(Optimizer):

    @torch.no_grad()
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        sm_coef=1.0,
        smoothing="l2",
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        scale_lrd=True,
    ):
        super(SaddleSAGA, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = lrp if lrd is None else lrd
        if scale_lrd:
            self.lrd = self.lrd / self.objective.n
        n, d = self.objective.n, self.objective.d

        num_params = getattr(
            objective,
            "num_parameters",
            objective.n_class * d if objective.n_class else d,
        )
        self.weights = _kaiming_init(num_params, objective).requires_grad_(True)
        self.grad_table = torch.zeros(n, num_params, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)

        self.sm_coef = n * sm_coef if smoothing == "l2" else sm_coef

        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights)
        # self.lam = torch.ones(n, dtype=torch.float64) / n
        self.lam = self.sigmas[torch.argsort(torch.argsort(self.losses))]
        self.rho = self.lam.clone()

        # with torch.enable_grad():
        #     for i in range(n):
        #         loss = self.objective.loss(
        #             self.weights, self.objective.X[i, :], self.objective.y[i]
        #         )
        #         self.grad_table[i] = torch.autograd.grad(outputs=loss, inputs=self.weights)[
        #             0
        #         ]
        self.grad_table = self.objective.get_indiv_grad(self.weights)
        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)
        self.center = torch.ones(n, dtype=torch.float64) / n  # cached; avoids O(n) alloc per step

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        pass

    def step(self):
        n = self.objective.n

        # Compute gradient at current iterate.
        i = torch.tensor([self.rng_grad.randint(0, n)])
        x = self.objective.X[i]
        y = self.objective.y[i]
        loss = self.objective.loss(self.weights, x, y)
        loss_val = loss.item()  # plain scalar; dual update needs value only, not graph
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze().detach()

        with torch.no_grad():
            # Compute variance-reduced direction from table.
            g_old = self.grad_table[i].reshape(-1)
            v = n * self.lam[i] * g - n * self.rho[i] * g_old + self.running_subgrad

            # Update iterate: prox step for linear; plain gradient step for MLP.
            if getattr(self.objective, "autodiff", False):
                new_weights = self.weights.detach() - self.lrp * v
            else:
                new_weights = (
                    1
                    / (self.lrp * self.objective.l2_reg / n + 1)
                    * (self.weights.detach() - self.lrp * v)
                )

            # Dual update: avoid allocating a full (n,) one-hot tensor each step.
            # eta = losses everywhere, except eta[i] shifts by n*(loss_val - losses[i]).
            delta = n * (loss_val - self.losses[i].item())
            eta = self.losses.clone()
            eta[i] = eta[i] + delta
            cur_lam = self.lam[i].clone()

            self.lam = get_smooth_weights(
                (self.lam + self.lrd * eta - self.center).detach(),
                self.sigmas,
                1 + self.lrd * self.sm_coef,
                smoothing="l2",
            )

            # Update tables in-place.
            self.losses[i] = loss_val
            self.grad_table[i] = g.reshape(1, -1)
            rho_old = self.rho[i].clone()
            self.rho[i] = cur_lam
            self.running_subgrad.add_(cur_lam * g - rho_old * g_old)

        self.weights = new_weights.requires_grad_(True)

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len