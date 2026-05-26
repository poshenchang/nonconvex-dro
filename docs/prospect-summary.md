# Technical Summary of the Prospect Algorithm: Framework, Theory, and Assumptions

## 1. Mathematical Problem Formulation

The Primal Objective and Spectral Risk The Prospect algorithm addresses the distributionally robust optimization (DRO) problem by minimizing the regularized primal objective $F_\sigma(w)$. This formulation leverages spectral risk measures (SRMs) to provide sensitivity to distribution shifts and subpopulation performance:
$$
F_\sigma(w) := R_\sigma(\ell(w)) + \frac{\mu}{2} \|w\|_2^2
$$
The risk functional $R_\sigma(l)$ is defined as a weighted sum of the order statistics of the loss vector $l \in \mathbb{R}^n$:
$$
R_\sigma(l) := \sum_{i=1}^n \sigma_i l_{(i)}, \quad \text{where } l_{(1)} \le l_{(2)} \le \dots \le l_{(n)}
$$
Here, \sigma is the spectrum vector satisfying $0 \le \sigma_1 \le \dots \le \sigma_n$ and $\sum \sigma_i = 1$. This corresponds to a dual game against an adversary selecting a reweighting $q$ from the permutahedron $\mathcal{P}(\sigma)$—the convex hull of all permutations of $\sigma$:
$$
R_P(l) := \max_{q \in \mathcal{P}(\sigma)} \left\{ \sum_{i=1}^n q_i l_i - \frac{\nu}{n} \sum_{i=1}^n f(nq_i) \right\}
$$
Uncertainty Sets and Divergence Penalties The shift cost hyperparameter $\nu \ge 0$ scales the $f$-divergence penalty $D_f(q || 1_n/n)$. Prospect supports various spectra $\sigma$ to recover canonical risk measures:

* Conditional Value-at-Risk (CVaR): $np$ elements of $\sigma$ are non-zero and equal.
* Extremile: $\sigma_i = (i/n)^b - ((i-1)/n)^b$ for $b \ge 1$.
* Exponential Spectral Risk Measure (ESRM): $\sigma_i \propto e^{\gamma i/n}$ for $\gamma > 0$.

The divergence generator $f$ determines the geometry of the shift penalty. Canonical examples include the $\chi^2$-divergence ($f(x) = x^2 - 1$) and Kullback-Leibler (KL) divergence ($f(x) = x \ln x$).

## 2. Theoretical Assumptions and Conditions

The convergence analysis of Prospect relies on the Lipschitz-smoothness of the individual losses and the strong convexity of the divergence generator.

| Property	| Requirement |
|-|-|
| Loss Function ($\ell_i$)	| Convex, G-Lipschitz continuous, and L-smooth. |
| Divergence Generator ($f$)	| $\alpha_n$-strongly convex on $[0, n]$. For $\chi^2$, $\alpha_n = 2$; for KL, $\alpha_n = 1/n$. |
| Combined Smoothness ($M$)	| The smoothness constant of the regularized loss $r_i$ is $M = L + \mu$. |
| Regularization ($\mu$)	| Strictly positive ($\mu > 0$). |

Critical Constants and Skewness The algorithm's iteration complexity is governed by two primary condition numbers:

* Primal Condition Number ($\kappa$): $1 + L/\mu$, the ratio of smoothness to regularization.
* Spectrum Skewness ($\kappa_\sigma$): $n\sigma_n$, measuring the maximum adversarial weight relative to the uniform distribution.

## 3. The Prospect Algorithmic Framework

Prospect utilizes a sophisticated two-part mechanism to mitigate the bias and variance inherent in stochastic DRO updates.

Bias Reduction via Lipschitz Weight Mapping The gradient of a spectral risk measure is $\nabla R_\sigma(l) = q_l$, where $q_l$ is the maximizer of the dual problem. Since the mapping $l \mapsto q_l$ is $(n\alpha_n\nu)^{-1}$-Lipschitz continuous, Prospect maintains a table of losses $l \approx \ell(w)$. By updating a single entry $l_j = \ell_j(w)$ per iteration, the algorithm tracks the adversarial weights $q$ such that the resulting gradient estimate is asymptotically unbiased.

Variance Reduction via SAGA-inspired Control Variates To avoid the sublinear convergence typical of stochastic methods with decreasing learning rates, Prospect employs a control variate scheme. It maintains a gradient table $g_i \approx \nabla r_i(w)$ and a weight table $\rho_i \approx q_i$. The stochastic gradient estimator $v$ is constructed as follows:
$$
v = n q_{i} \nabla r_i(w) - (n \rho_i g_i - \bar{g})
$$
where $\bar{g} = \sum \rho_j g_j$. This correction ensures that the variance $E\|v - \nabla F_\sigma(w)\|^2$ vanishes as $w \to w^*$, enabling linear convergence with a constant step size \eta.

The Iterative Procedure

1. Sample: Select indices $i, j \sim \text{Unif}[n]$ independently.
2. Iterate Update: Update $w \leftarrow w - \eta v$ using the estimator above.
3. Bias Reducer Update: Update $l_j \leftarrow \ell_j(w)$ and solve the inner maximization problem exactly for the updated $q$ using the loss table (Algorithm 1, Line 10).
4. Variance Reducer Update: Refresh the stored gradient $g_i \leftarrow \nabla r_i(w)$ and stored weight $\rho_i \leftarrow q_i$, then update the running sum $\bar{g}$.


## 4. Convergence Properties

Prospect provides the first unconditional linear convergence guarantee for SRM-based DRO with a single hyperparameter.

* Linear Convergence (Large Shift Cost): For $\nu \ge \Omega(G^2/\mu\alpha_n)$, Prospect achieves an iteration complexity of $O((n + \kappa\kappa_\sigma) \ln(1/\epsilon))$.
* Unconditional Convergence: Unlike methods such as LSVRG, Prospect maintains linear convergence for any $\nu > 0$.
* Hidden Smoothness (Proposition 2): Prospect can recover the minimizer of a non-smooth objective ($\nu = 0$). If the losses are distinct at the optimum $w^*_0$, there exists a threshold $\nu_0 > 0$ such that $w^*_\nu = w^*_0$ for all $\nu \in (0, \nu_0]$. Specifically, $\nu_0$ is proportional to the minimum gap between distinct optimal losses: $l_{(i+1)}(w^*_0) - l_{(i)}(w^*_0)$.


## 5. Computational Complexity and Implementation

Complexity Analysis The algorithm decouples the cost of n and d, achieving an iteration complexity of $O(n + d)$. This is significantly more efficient than full-batch methods ($O(nd)$) for large-scale datasets.

Inner Maximization Routine The weight update $q = q_{opt}(l)$ is solved exactly in $O(n \ln n)$ (or amortized $O(n)$) via:

1. Sorting: Sorting the loss table. Since only one entry changes per iteration, a Bubble sort starting at the updated index j resorts the list in $O(s)$ where $s$ is the number of swaps.
2. Isotonic Regression: The Pool Adjacent Violators Algorithm (PAV) solves the dual problem (Equation 13) in $O(n)$ time.
3. Conversion: Converting the PAV output back to adversarial weights $q$. For KL-divergence, the metadata requires a logsumexp operation to maintain numerical stability.

Memory Efficiency For generalized linear models (GLMs), memory is optimized from $O(nd)$ to $O(n + d)$ by storing only scalar derivatives $h'(x_i^\top w, y_i)$ rather than full gradient vectors.


## 6. Extensions and Reference for Non-Convex Proposals

Moreau Envelope Variant For non-smooth losses (e.g., $L_1$ penalties), Prospect can be adapted using Moreau envelopes. By replacing standard gradients with gradients of the Moreau envelope—computed via proximal operators—the algorithm retains linear convergence. The update becomes $w(t+1) = \text{prox}_{\eta r_{it}}(w(t) + \eta(g_{it} - \bar{g}))$, utilizing the co-coercivity of the proximal operator in the Lyapunov analysis.

Strategic Advantages for Research Proposals

* Single Hyperparameter Tuning: Only the learning rate $\eta$ requires tuning, simplifying deployment compared to primal-dual saddle-point methods.
* Empirical Robustness and Stability: Prospect consistently outperforms LSVRG on benchmarks like Diabetes (ACSIncome) and Amazon Reviews. It avoids the "stale checkpoint" problem of epoch-based methods by dynamically updating running estimates.
* Global Efficiency: The $O(n+d)$ iteration cost and stability on fairness metrics (Statistical Parity) make it a superior base for non-convex DRO extensions.
* Asymptotic Unbiasedness: The loss-tracking mechanism provides a theoretically grounded method to handle biased stochastic gradients in robust optimization.

