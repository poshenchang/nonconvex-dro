import torch
import numpy as np
from numba import jit

# import warnings

# warnings.filterwarnings("error")


def sinkhorn_knopp(q, p, C, epsilon, max_iters=100, tol=1e-6):
    """
    Compute the Sinkhorn-Knopp optimal transport plan P and distance.
    :param q: (torch.Tensor) shape (m,) marginal distribution
    :param p: (torch.Tensor) shape (n,) marginal distribution
    :param C: (torch.Tensor) shape (m, n) cost matrix
    :param epsilon: (float) entropy regularization coefficient
    :param max_iters: (int) maximum number of scaling iterations
    :param tol: (float) convergence tolerance on marginal error
    :return: P (joint plan), distance (OT distance)
    """
    # K has shape (m, n)
    K = torch.exp(-C / epsilon)
    
    # Initialize scaling vectors
    u = torch.ones_like(q) / len(q)
    v = torch.ones_like(p) / len(p)
    
    for _ in range(max_iters):
        u_old = u.clone()
        # update u: u = q / (K v)
        u = q / (torch.matmul(K, v) + 1e-16)
        # update v: v = p / (K^T u)
        v = p / (torch.matmul(K.T, u) + 1e-16)
        
        # Check convergence
        if torch.norm(u - u_old) < tol:
            break
            
    P = u[:, None] * K * v[None, :]
    distance = torch.sum(P * C)
    return P, distance


def get_wasserstein_weights(losses, C, shift_cost, epsilon=0.1, K=None):
    """
    Compute adversarial weights subject to a Wasserstein constraint using an optimized kernel-based formulation.
    
    :param losses: (torch.Tensor) shape (n,) loss values at the current iterate
    :param C: (torch.Tensor) shape (n, n) cost matrix between data points
    :param shift_cost: (float) the penalty parameter nu for the Wasserstein distance
    :param epsilon: (float) entropy regularization for the Sinkhorn-like algorithm
    :param K: (torch.Tensor) shape (n, n) optional precomputed kernel exp(-C / epsilon)
    :return: (torch.Tensor) shape (n,) smooth adversarial weights
    """
    if K is None:
        K = torch.exp(-C / epsilon)
    
    # Log-sum-exp stabilization: shift losses by their maximum value to avoid overflow
    losses_shifted = losses - torch.max(losses)
    v = torch.exp(losses_shifted / (shift_cost * epsilon))
    
    # Matrix-vector multiplications instead of constructing the NxN plan matrix
    w = torch.matmul(K.T, v)
    z = 1.0 / (w + 1e-16)
    y = torch.matmul(K, z)
    q = (v * y) / len(losses)
    return q

def get_smooth_weights(losses, spectrum, smooth_coef, smoothing="l2"):
    """
    Losses are the values of the losses at the current iterate, spectrum are the weights of the spectral measure
    considered given in non-decreasing order
    :param losses: (torch.Tensor of shape (n,) values of the losses at the current iterate
    :param spectrum: (torch.Tensor of shape (n,) weights of the spectral measure considered given in non-decreasing
    order
    :param smooth_coef: (float) value of the smoothing coefficient
    :param smoothing: (str) choose between 'l2' and 'neg_entropy' for resulting weights that are either
    smooth w.r.t. l2 norm or l1 norm, see latex notes for more details (note that we use centered smoothing operators)
    :return:
    """
    if smooth_coef < 1e-16:
        return spectrum[torch.argsort(torch.argsort(losses))]
    n = len(losses)
    scaled_losses = losses / smooth_coef
    perm = torch.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if smoothing == "l2":
        primal_sol = l2_centered_isotonic_regression(
            sorted_losses.numpy(), spectrum.numpy()
        )
    elif smoothing == "neg_entropy":
        primal_sol = torch.tensor(
            neg_entropy_centered_isotonic_regression(
                sorted_losses.numpy(), spectrum.numpy()
            )
        )
    elif smoothing == "wasserstein":
        raise ValueError("Wasserstein penalty should be handled by wasserstein_softmax directly with a cost matrix.")
    else:
        raise NotImplementedError
    inv_perm = torch.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if smoothing == "l2":
        smooth_weights = scaled_losses - primal_sol + 1 / n
    elif smoothing == "neg_entropy":
        smooth_weights = torch.exp(scaled_losses - primal_sol) / n
    else:
        raise NotImplementedError
    return smooth_weights


def get_smooth_weights_sorted(losses, spectrum, smooth_coef, smoothing="l2", tol=1e-16):
    """
    Losses are the values of the sorted losses at the current iterate, spectrum are the weights of the spectral measure
    considered given in non-decreasing order
    :param losses: (torch.Tensor of shape (n,) values of the losses at the current iterate
    :param spectrum: (torch.Tensor of shape (n,) weights of the spectral measure considered given in non-decreasing
    order
    :param smooth_coef: (float) value of the smoothing coefficient
    :param smoothing: (str) choose between 'l2' and 'neg_entropy' for resulting weights that are either
    smooth w.r.t. l2 norm or l1 norm, see latex notes for more details (note that we use centered smoothing operators)
    :return: smooth_weights, in sorted order.
    """
    if smooth_coef < 1e-16:
        return spectrum

    n = len(losses)
    sorted_losses = losses / smooth_coef

    if smoothing == "l2":
        primal_sol = torch.tensor(
            l2_centered_isotonic_regression(sorted_losses.numpy(), spectrum.numpy())
        )
    elif smoothing == "neg_entropy":
        primal_sol = torch.tensor(
            neg_entropy_centered_isotonic_regression(
                sorted_losses.numpy(), spectrum.numpy()
            )
        )
    else:
        raise NotImplementedError
    if smoothing == "l2":
        smooth_weights = sorted_losses - primal_sol + 1 / n
    elif smoothing == "neg_entropy":
        smooth_weights = torch.exp(sorted_losses - primal_sol) / n
    else:
        raise NotImplementedError
    return smooth_weights


@jit
def l2_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    means = [losses[0] + 1 / n - spectrum[0]]
    counts = [1]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] + 1 / n - spectrum[i])
        counts.append(1)
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_count, prev_end_point = (
                means.pop(),
                counts.pop(),
                end_points.pop(),
            )
            means[-1] = (counts[-1] * means[-1] + prev_count * prev_mean) / (
                counts[-1] + prev_count
            )
            counts[-1] = counts[-1] + prev_count
            end_points[-1] = prev_end_point

    # Previous output without numba
    # sol = output_sol_iso_reg(end_points, means, n)

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol


@jit(nopython=True)
def neg_entropy_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    logn = np.log(n)
    log_spectrum = np.log(spectrum)

    lse_losses = [losses[0]]
    lse_log_spectrum = [log_spectrum[0]]
    means = [losses[0] - log_spectrum[0] - logn]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] - log_spectrum[i] - logn)
        lse_losses.append(losses[i])
        lse_log_spectrum.append(log_spectrum[i])
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean = means.pop()
            prev_lse_loss = lse_losses.pop()
            prev_lse_log_spectrum = lse_log_spectrum.pop()
            prev_end_point = end_points.pop()
            
            new_lse_loss = np.logaddexp(lse_losses[-1], prev_lse_loss)
            new_lse_log_spectrum = np.logaddexp(lse_log_spectrum[-1], prev_lse_log_spectrum)
            
            means[-1] = new_lse_loss - new_lse_log_spectrum - logn
            lse_losses[-1] = new_lse_loss
            lse_log_spectrum[-1] = new_lse_log_spectrum
            end_points[-1] = prev_end_point

    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol


def output_sol_iso_reg(end_points, means, n):
    sol = torch.zeros(n)
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol


def test_centered_isotonic_regression():
    n = 1000
    smooth_coef = 0.1
    # Try with extremile coefficients below
    r = 5
    spectrum = (
        (torch.arange(n, dtype=torch.float64) + 1) ** r
        - torch.arange(n, dtype=torch.float64) ** r
    ) / (n**r)
    for i in range(20):
        losses = torch.randn(n, dtype=torch.float64)
        perm = torch.argsort(losses)
        invperm = torch.argsort(perm)

        # l2 smoothing
        # The right scaling for the l2 smoothing should be n times the smoothing coefficient see notes
        smooth_weights = get_smooth_weights(
            losses, spectrum, n * smooth_coef, smoothing="l2"
        )
        print(
            "Sum smooth weights l2 smoothing (should be 1):{}".format(
                torch.sum(smooth_weights)
            )
        )
        print(
            "Norm diff btw non-smooth & smoothed weights l2 smoothing:{}".format(
                torch.norm(smooth_weights - spectrum[invperm])
            )
        )

        # Negative entropy smoothing
        smooth_weights = get_smooth_weights(
            losses, spectrum, smooth_coef, smoothing="neg_entropy"
        )
        print(
            "Sum smooth weights neg ent smoothing (should be 1):{}".format(
                torch.sum(smooth_weights)
            )
        )
        print(
            "Norm diff btw non-smooth & smoothed weights neg ent smoothing:{}".format(
                torch.norm(smooth_weights - spectrum[invperm])
            )
        )

    # Try with erm, i.e., uniform spectrum, should give us smooth_weights = uniform
    smooth_coef = 0.1
    spectrum = torch.ones(n) / n
    for i in range(20):
        losses = torch.randn(n, dtype=torch.float64)
        smooth_weights = get_smooth_weights(
            losses, spectrum, n * smooth_coef, smoothing="l2"
        )
        print(
            "Norm diff l2 smooth weights uniform (should be 0):{}".format(
                torch.norm(spectrum - smooth_weights)
            )
        )
        smooth_weights = get_smooth_weights(
            losses, spectrum, n * smooth_coef, smoothing="neg_entropy"
        )
        print(
            "Norm diff neg ent smooth weights uniform (should be 0):{}".format(
                torch.norm(spectrum - smooth_weights)
            )
        )


if __name__ == "__main__":
    test_centered_isotonic_regression()
