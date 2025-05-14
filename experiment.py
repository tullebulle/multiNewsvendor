import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# ------------------------------------------------------------------ #
# helper : empirical objective  ( eq. (1) in main.tex )
# ------------------------------------------------------------------ #
def empirical_loss(samples, q, b, h):
    """
    Compute  ĝ(q) = (1/m) Σ_j Σ_i [ b_i (d_ij - q_i)^+ + h_i (q_i - d_ij)^+ ].
    """
    k        = len(samples)
    m        = len(samples[0])
    loss_acc = 0.0
    for j in range(m):
        for i in range(k):
            d = samples[i][j]
            loss_acc += b[i] * max(d - q[i], 0.0) + h[i] * max(q[i] - d, 0.0)
    return loss_acc / m


def find_optimal_allocation(samples, b, h, Q,
                            tol_lambda=1e-6,
                            tol_sum   =1e-8,
                            max_it    =60):
    """
    SAA newsvendor with a single capacity.
    Returns q* with Σ q_i = Q (up to tol_sum) and prints ĝ(q*).
    """
    k = len(samples)
    sorted_samples = [np.sort(s) for s in samples]

    # --------------------------------------------------  helpers ----
    def q_i(i, lam):
        if lam >= b[i] - 1e-12:
            return 0.0
        m   = len(sorted_samples[i])
        p   = (b[i] - lam) / (b[i] + h[i])
        idx = max(0, min(int(np.ceil(m * p)) - 1, m - 1))
        return float(sorted_samples[i][idx])

    def S(lam):          # total order size
        return sum(q_i(i, lam) for i in range(k))
    # ---------------------------------------------------------------

    # ----  unconstrained check
    if S(0.0) <= Q + tol_sum:
        q_uncon = [q_i(i, 0.0) for i in range(k)]
        # print(f"ĝ(unconstrained) = {empirical_loss(samples, q_uncon, b, h):.4f}")
        return q_uncon

    # ----  bisection to bracket the jump
    lam_lo, lam_hi = 0.0, max(b)
    for _ in range(max_it):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if S(lam_mid) > Q:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if lam_hi - lam_lo < tol_lambda:
            break

    q_lo = np.array([q_i(i, lam_hi) for i in range(k)])      # below or == Q
    left_over = Q - sum(q_lo)
    if left_over <= tol_sum:
        loss = empirical_loss(samples, q_lo, b, h)
        print(f"ĝ(q*) = {loss:.4f}")
        return q_lo

    # ----  continuous adjustment inside the last step ---------------
    q_hi = np.array([q_i(i, lam_lo) for i in range(k)])
    candidates = []
    for i in range(k):
        jump = q_hi[i] - q_lo[i]    # >0 only at jump sites
        if jump > 0:
            mean_demand = np.mean(sorted_samples[i])
            candidates.append((i, mean_demand, jump))

    # sort candidates by descending mean (or any rule you like)
    # candidates.sort(key=lambda x: x[1], reverse=True)

    for i, _, jump in candidates:
        add = min(jump, left_over)
        q_lo[i] += add
        left_over -= add
        if left_over <= tol_sum:
            break

    assert abs(sum(q_lo) - Q) <= 1e-6, "capacity not met"

    loss = empirical_loss(samples, q_lo, b, h)
    # print(f"ĝ(q*) = {loss:.4f}")
    return q_lo

    # # (iv) sites that jumped in the last step give us a continuous interval
    # q_hi = [q_i(i, lam_lo) for i in range(k)]   # S(q_hi) ≥ Q
    # deltas = [(i, q_hi[i] - q_lo[i])
    #           for i in range(k) if q_hi[i] > q_lo[i]]

    # # distribute the leftover inside those open intervals
    # for i, delta in deltas:
    #     add = min(delta, leftover)
    #     q_lo[i] += add
    #     leftover -= add
    #     if leftover <= tol_sum:
    #         break

    # # safety check
    # assert abs(sum(q_lo) - Q) <= 1e-6, \
    #     f"Capacity mismatch: {sum(q_lo)} vs {Q}"
    # return q_lo

def plot_error_vs_m(errors, m_values):
    # log log plot
    plt.figure(figsize=(6, 4))
    plt.plot(m_values, errors, 'o-', linewidth=2)
    plt.xlabel('Number of Samples (m)', fontsize=14)
    plt.ylabel('Mean Absolute Error |q_i - μ_i|', fontsize=14)
    plt.title('Convergence of Optimal Allocation to Expected Value', fontsize=16)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def plot_true_vs_estimated(mu, q_star, Q = None):
    # rather true on y axis and estimated on x axis
    S_lambda_star = sum(q_star)
    string_Q = "" if Q is None else ", $Q$ = " + str(round(Q, 2))
    plt.figure(figsize=(6, 4))
    plt.plot(mu, q_star, 'o', linewidth=2)
    plt.plot(mu, mu, linewidth=2)
    plt.xlabel(r'$\mu_i$')
    plt.ylabel(r'$\hat q_i$')
    plt.title(r'Expected value vs. estimated allocations, $S(\lambda^*)$ = ' + str(round(S_lambda_star, 2)) + string_Q, fontsize=16)
    plt.savefig('true_vs_estimated.png')
    plt.grid(True)
    plt.show()

def plot_true_vs_estimated_with_costs(mu, q_star, b, h, Q=None):
    """
    Plot true vs estimated values with vertical lines representing costs.
    
    Parameters:
    - mu: true mean values
    - q_star: estimated optimal allocation
    - b: lost-sales costs
    - h: holding costs
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the diagonal line (true = estimated)
    min_val = min(min(mu), min(q_star))
    max_val = max(max(mu), max(q_star))
    buffer = (max_val - min_val) * 0
    plt.plot([min_val - buffer, max_val + buffer], 
             [min_val - buffer, max_val + buffer], 
             'orange', linewidth=2, label='True = Estimated')
    
    # Plot the points
    plt.scatter(mu, q_star, color='blue', s=80, alpha=0.7, label='Allocation')
    
    # Add vertical lines for h_i (red, going up) and b_i (green, going down)
    for i in range(len(mu)):
        # Red line for h_i (going up from the diagonal)
        plt.plot([mu[i], mu[i]], [mu[i], mu[i] + h[i]], 
                 color='red', linewidth=1.5, alpha=0.6)
        
        # Green line for b_i (going down from the diagonal)
        plt.plot([mu[i], mu[i]], [mu[i], mu[i] - b[i]], 
                 color='green', linewidth=1.5, alpha=0.6)
    
    # Add legend entries for the vertical lines
    plt.plot([], [], color='red', linewidth=1.5, label='Holding cost (h_i)')
    plt.plot([], [], color='green', linewidth=1.5, label='Lost-sales cost (b_i)')
    
    # Add labels and title
    string_Q = "" if Q is None else r"$S(\lambda^*) =$ " + str(round(sum(q_star), 2)) + ", $Q$ = " + str(round(Q, 2))
    plt.xlabel(r'True Mean ($\mu_i$)', fontsize=14)
    plt.ylabel(r'Optimal Allocation ($\hat q_i$)', fontsize=14)
    plt.title('Optimal Allocation vs. Mean, ' + string_Q, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Ensure equal scaling
    plt.axis('equal')
    
    # Add some padding to the plot
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer - max(b), max_val + buffer + max(h))
    
    plt.tight_layout()
    plt.show()

def run_experiment_1():
    # Parameters
    k = 10  # Number of locations
    m_values = [10, 20, 30, 40, 50, 100, 200, 500, 1000]  # Sample sizes to test
    
    # Set up mean vector and covariance matrix for multivariate Gaussian
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    
    # Create a random covariance matrix with some dependence
    A = np.random.randn(k, k)
    cov = np.dot(A, A.T)  # Ensures positive semi-definite
    
    # Set equal lost-sales and holding costs
    b = np.ones(k) * 10
    h = np.ones(k) * 10
    
    # Set a very large capacity so it doesn't bind
    Q = sum(mu) * 5
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate absolute error
        abs_error = np.mean(np.abs(np.array(q_star) - mu))
        errors.append(abs_error)
        
        print(f"m = {m}, Mean absolute error: {abs_error:.4f}")
    
    # Plot results
    plot_error_vs_m(errors, m_values)
    plot_true_vs_estimated(mu, q_star)
    print("Budget (sum q_i): ", sum(q_star), "Capacity Q: ", Q)

def run_experiment_2():
    # Parameters
    k = 10  # Number of locations
    m_values = [1000]  # Sample sizes to test
    
    # Set up mean vector and covariance matrix for multivariate Gaussian
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    
    # Create a random covariance matrix with some dependence
    A = np.random.randn(k, k)
    cov = np.dot(A, A.T)  # Ensures positive semi-definite
    
    # Set equal lost-sales and holding costs
    b = np.ones(k) * 10
    # b += np.random.normal(0, 1, k)

    h = np.ones(k) * 10
    
    # Set a very large capacity so it doesn't bind
    Q = 0.8 * sum(mu)
    
    # Store results
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)

        print("Budget (sum q_i): ", sum(q_star), "Capacity Q: ", Q)
    
    # Plot results
    # plot_error_vs_m(errors, m_values)
    plot_true_vs_estimated(mu, q_star, Q)
    # plot_true_vs_estimated_with_costs(mu, q_star, b, h)

def run_experiment_3():
    # Parameters
    k = 10  # Number of locations
    m_values = [10, 20, 30, 40, 50, 100, 200, 500, 1000]  # Sample sizes to test
    np.random.seed(42)
    
    # Set up mean vector and covariance matrix for multivariate Gaussian
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    # mu[0] = 150
    # Create a random covariance matrix with some dependence
    A = np.random.randn(k, k)
    cov = np.dot(A, A.T)  # Ensures positive semi-definite
    
    # Set equal lost-sales and holding costs
    b = np.random.uniform(0, 10, k)
    h = np.random.uniform(0, 10, k)
    # h[0] = 100
    
    # Set a very large capacity so it doesn't bind
    Q = 0.8 * sum(mu)
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate absolute error
        abs_error = np.mean(np.abs(np.array(q_star) - mu))
        errors.append(abs_error)
        
        print(f"m = {m}, Mean absolute error: {abs_error:.4f}")
        print("Budget (sum q_i): ", sum(q_star), "Capacity Q: ", Q)
    
    # Plot results
    # plot_true_vs_estimated_with_costs(mu, q_star, b, h)
    plot_true_vs_estimated_with_costs(mu, q_star, b, h, Q)

def run_experiment_4(m_values, m_reference, h, b, plot=True):
    """
    Experiment with almost independent demands between locations.
    """
    print("\n=== EXPERIMENT 4: INDEPENDENT DEMANDS ===")
    # Parameters
    k = 10  # Number of locations
    
    # Set up mean vector and covariance matrix for multivariate Gaussian
    np.random.seed(42)  # For reproducibility
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    
    # Create a nearly diagonal covariance matrix (independent demands)
    cov = np.eye(k) * 100  # Diagonal with variance 100
    # Add very small off-diagonal elements
    cov += np.ones((k, k)) * 1  # Small positive correlation
    
    
    # Set capacity to 80% of sum of means
    Q = 0.5 * sum(mu)
    
    # First get the reference solution with m_reference
    samples_matrix_ref = multivariate_normal.rvs(mean=mu, cov=cov, size=m_reference)
    samples_ref = [samples_matrix_ref[:, i] for i in range(k)]
    q_star_ref = find_optimal_allocation(samples_ref, b, h, Q)
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate error relative to reference solution
        rel_indices = np.argwhere(q_star_ref != 0.0).flatten()

        rel_error = np.mean(np.abs(np.array(q_star)[rel_indices] - np.array(q_star_ref)[rel_indices]) / np.array(q_star_ref)[rel_indices])
        errors.append(rel_error)
        
    
    print("sum q_star: ", sum(q_star), "Q: ", Q)

    # Also plot the final allocation vs true means
    if plot:
        plot_error_vs_m(errors, m_values)
        # plot_true_vs_estimated_with_costs(mu, q_star_ref, b, h)

    return errors, m_values
def run_experiment_5(m_values, m_reference, h, b, plot=True):
    """
    Experiment with positively correlated demands between locations.
    Each demand vector is Gaussian centered around a shared mean.
    """
    print("\n=== EXPERIMENT 5: POSITIVELY CORRELATED DEMANDS ===")
    # Parameters
    k = 10  # Number of locations
    
    # Set up mean vector and base covariance matrix
    np.random.seed(43)  # Different seed
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    
    # Create a highly correlated covariance matrix
    # We'll use a factor model: d_i = μ_i + β_i * common_factor + ε_i
    beta = np.ones(k) * 0.8  # Common factor loading
    common_var = 100  # Variance of common factor
    idiosyncratic_var = 20  # Variance of idiosyncratic noise
    
    # Covariance matrix from factor model
    cov = np.outer(beta, beta) * common_var
    np.fill_diagonal(cov, np.diag(cov) + idiosyncratic_var)

    
    # Set capacity to 80% of sum of means
    Q = 0.5 * sum(mu)
    
    # First get the reference solution with m_reference
    samples_matrix_ref = multivariate_normal.rvs(mean=mu, cov=cov, size=m_reference)
    samples_ref = [samples_matrix_ref[:, i] for i in range(k)]
    q_star_ref = find_optimal_allocation(samples_ref, b, h, Q)
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate error relative to reference solution
        rel_indices = np.argwhere(q_star_ref != 0.0).flatten()
        rel_error = np.mean(np.abs(np.array(q_star)[rel_indices] - np.array(q_star_ref)[rel_indices]) / np.array(q_star_ref)[rel_indices])
        errors.append(rel_error)
    
    print("sum q_star: ", sum(q_star), "Q: ", Q)
    
    # Plot convergence
    if plot:
        plot_error_vs_m(errors, m_values)
        # Also plot the final allocation vs true means
        # plot_true_vs_estimated_with_costs(mu, q_star_ref, b, h)
    return errors, m_values
def run_experiment_6(m_values, m_reference, h, b, plot=True):
    """
    Experiment with negatively correlated demands between locations.
    """
    print("\n=== EXPERIMENT 6: NEGATIVELY CORRELATED DEMANDS ===")
    # Parameters
    k = 10  # Number of locations
    
    # Set up mean vector
    np.random.seed(44)  # Different seed
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    
    # Create a negatively correlated covariance matrix
    # Start with a positive definite matrix
    A = np.random.randn(k, k)
    base_cov = np.dot(A, A.T)
    
    # Convert to correlation matrix
    D = np.sqrt(np.diag(base_cov))
    base_corr = base_cov / np.outer(D, D)
    
    # Flip the sign of off-diagonal elements to create negative correlation
    neg_corr = base_corr.copy()
    neg_corr[~np.eye(k, dtype=bool)] *= -0.5  # Scale to ensure matrix remains positive definite
    
    # Convert back to covariance matrix with desired variances
    variances = np.ones(k) * 100
    cov = neg_corr * np.outer(np.sqrt(variances), np.sqrt(variances))
    
    # Ensure the matrix is positive definite (add small diagonal if needed)
    min_eig = np.min(np.linalg.eigvals(cov))
    if min_eig < 0:
        cov += (-min_eig + 1e-6) * np.eye(k)

    
    # Set capacity to 80% of sum of means
    Q = 0.5 * sum(mu)
    
    # First get the reference solution with m_reference
    samples_matrix_ref = multivariate_normal.rvs(mean=mu, cov=cov, size=m_reference)
    samples_ref = [samples_matrix_ref[:, i] for i in range(k)]
    q_star_ref = find_optimal_allocation(samples_ref, b, h, Q)
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate error relative to reference solution
        rel_indices = np.argwhere(q_star_ref != 0.0).flatten()
        rel_error = np.mean(np.abs(np.array(q_star)[rel_indices] - np.array(q_star_ref)[rel_indices]) / np.array(q_star_ref)[rel_indices])
        errors.append(rel_error)
        
    print("sum q_star: ", sum(q_star), "Q: ", Q)
    # Plot convergence
    if plot:
        plot_error_vs_m(errors, m_values)
        # Also plot the final allocation vs true means
        # plot_true_vs_estimated_with_costs(mu, q_star_ref, b, h)
    return errors, m_values

def compare_convergence(errors4, errors5, errors6, m_values4, m_values5, m_values6, m_reference):
    """
    Run all three experiments and compare their convergence rates.
    """
    print("\n=== COMPARING CONVERGENCE RATES ===")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(m_values4, errors4, 'o-', linewidth=2, color='blue', label='Independent')
    plt.plot(m_values5, errors5, 's-', linewidth=2, color='green', label='Positive Correlation')
    plt.plot(m_values6, errors6, '^-', linewidth=2, color='red', label='Negative Correlation')
    
    plt.xlabel('Number of Samples (m)', fontsize=12)
    plt.ylabel(r'Mean Relative Error $\frac{|q_i - q_i^*|}{|q_i^*|}$', fontsize=12)
    plt.title('Convergence Speed Comparison by Correlation Structure, reference m = ' + str(m_reference), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.savefig('convergence_comparison.png')
    plt.show()


def run_experiment_7(m_values, m_reference):
    # Parameters
    k = 10
    h = np.random.uniform(0, 10, k)
    b = np.random.uniform(0, 10, k)
    errors4, m_values4 = run_experiment_4(m_values, m_reference=m_reference, h=h, b=b, plot=False)
    errors5, m_values5 = run_experiment_5(m_values, m_reference=m_reference, h=h, b=b, plot=False)
    errors6, m_values6 = run_experiment_6(m_values, m_reference=m_reference, h=h, b=b, plot=False)


    compare_convergence(errors4, errors5, errors6, m_values4, m_values5, m_values6, m_reference)  # This runs all three experiments and compares them


def run_experiment_8():
    # Parameters
    k = 10  # Number of locations
    m_values = [10, 20, 30, 40, 50, 100, 200, 500, 1000]  # Sample sizes to test
    
    # Set up mean vector and covariance matrix for multivariate Gaussian
    mu = np.random.uniform(50, 150, k)  # Random means between 50 and 150
    # mu[0] = 150
    # Create a random covariance matrix with some dependence
    A = np.random.randn(k, k)
    cov = np.dot(A, A.T)  # Ensures positive semi-definite
    
    # Set equal lost-sales and holding costs
    b = np.ones(k) * 10 + np.random.uniform(0, 10, k)
    h = np.ones(k) * 10 + np.random.uniform(0, 10, k)
    h[0] = 0.000001
    b[0] = 1000
    
    # Set a very large capacity so it doesn't bind
    Q = 0.5 * sum(mu)
    
    # Store results
    errors = []
    
    for m in m_values:
        # Generate m samples from multivariate Gaussian
        samples_matrix = multivariate_normal.rvs(mean=mu, cov=cov, size=m)
        
        # Reorganize samples by location
        samples = [samples_matrix[:, i] for i in range(k)]
        
        # Find optimal allocation
        q_star = find_optimal_allocation(samples, b, h, Q)
        
        # Calculate absolute error
        abs_error = np.mean(np.abs(np.array(q_star) - mu))
        errors.append(abs_error)
        
        print(f"m = {m}, Mean absolute error: {abs_error:.4f}")
        print("Budget (sum q_i): ", sum(q_star), "Capacity Q: ", Q)
    
    # Plot results
    plot_true_vs_estimated(mu, q_star)
    # plot_true_vs_estimated_with_costs(mu, q_star, b, h)

def run_experiment_9(error_type="relative", # LOG NORMAL DISTRIBUTION
        r: int = 40,
        ratios=(1, 10, 100, 1000),
        h_base: float = 10.0):
    """
    Experiment 9-bis — k = 1, different degrees of b/h "un-balancedness".

    Parameters
    ----------
    r       : int
        Number of repetitions per sample size m.
    ratios  : iterable
        The desired b : h ratios (h is kept fixed at h_base).
        e.g. (1, 10, 100, 1000)  ⇒
             (b,h) ∈ {(10,10), (100,10), (1000,10), (10000,10)}.
    h_base  : float
        Baseline holding cost h; b = ratio * h_base.
    """
    # -------------- imports (local to keep the outer namespace clean)
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # -------------- problem set-up
    np.random.seed(42)
    mu, sigma = 4.0, 0.5                    # log-normal demand parameters
    Q_big     = 1_000_000.0                 # capacity never binds

    m_ref  = 200_000                        # "ground truth" sample size
    m_vals = [5, 10, 20, 50, 100, 200,
              500, 1_000, 2_000, 5_000,
              10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 9-bis   (r = {r})   varying b/h ratios ===")

    # generate one huge reference sample once (re-used for all ratios)
    ref_samples = [np.random.lognormal(mu, sigma, m_ref)]

    # -------------------------------- containers  ---------------------------
    err_mean = {ratio: [] for ratio in ratios}
    err_std  = {ratio: [] for ratio in ratios}
    slopes   = {}

    # -------------------------------- reference solutions for each ratio ----
    q_ref = {}
    for ratio in ratios:
        b_val, h_val = ratio*h_base, h_base
        q_ref[ratio] = find_optimal_allocation(ref_samples,
                                               [b_val], [h_val], Q_big)[0]
        print(f"ratio {ratio:>5}:  b = {b_val:<7.1f},   "
              f"h = {h_val:<4.1f}  ⇒  q* = {q_ref[ratio]:.3f}")

    # -------------------------------- main loop over m ----------------------
    for m in m_vals:
        # collect one list of errors per ratio for the r repetitions
        rep_errors = {ratio: [] for ratio in ratios}

        for _ in range(r):
            samples = [np.random.lognormal(mu, sigma, m)]

            for ratio in ratios:
                b_val, h_val = ratio*h_base, h_base
                q_hat = find_optimal_allocation(samples,
                                                [b_val], [h_val], Q_big)[0]
                if error_type == "relative":
                    rel_err = abs(q_hat - q_ref[ratio]) / q_ref[ratio]
                elif error_type == "absolute":
                    rel_err = abs(q_hat - q_ref[ratio])
                rep_errors[ratio].append(rel_err)

        # aggregate mean / std for this m
        for ratio in ratios:
            err_mean[ratio].append(np.mean(rep_errors[ratio]))
            err_std [ratio].append(np.std (rep_errors[ratio]))

        # pretty progress line
        prog = "  |  ".join(
            f"{ratio:>4}: μ={err_mean[ratio][-1]:.3e}"
            for ratio in ratios)
        print(f"m = {m:6d}  |  {prog}")

    # -------------------------------- plotting ------------------------------
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'P', 'X'])
    for ratio, mark in zip(ratios, markers):
        # plt.errorbar(m_vals, err_mean[ratio], yerr=err_std[ratio],
        #              fmt=f'{mark}-', capsize=3,
        #              label=f'b/h = {ratio}')
        plt.plot(m_vals, err_mean[ratio], label=f'b/h = {ratio}', lw=1)

    # reference m^{-1/2} line from the *first* ratio
    ref_x = np.array([m_vals[0], m_vals[-1]])
    first_ratio = ratios[0]
    ref_y = err_mean[first_ratio][0] * (ref_x / m_vals[0])**(-0.5)
    title_string =  r'$\; \mathbb{E}[\,|\hat q - q^*|\,/\,q^*]$'
    if error_type == "absolute":
        title_string =  r'$\; \mathbb{E}[\,|\hat q - q^*|\,]$'
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$')
    plt.ylabel(title_string, fontsize=14)
    plt.title('Relative error, log-normal distribution', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(ls='--', alpha=.3)
    plt.tight_layout()
    plt.savefig('experiment9_ratios_convergence.png', dpi=300)
    plt.show()

    # -------------------------------- slopes --------------------------------
    for ratio in ratios:
        slopes[ratio] = np.mean([
            np.log(err_mean[ratio][i-1] / err_mean[ratio][i]) /
            np.log(m_vals[i]          /  m_vals[i-1])
            for i in range(1, len(m_vals))
        ])
    print("\nEmpirical average slopes:")
    for ratio in ratios:
        print(f"   ratio {ratio:>4} :  ≈  m^{slopes[ratio]:.3f}")

    # -------------- return a dict of results --------------------------------
    return dict(m_vals=m_vals,
                ratios=ratios,
                err_mean=err_mean,
                err_std =err_std,
                slopes  =slopes,
                q_ref   =q_ref)

def run_experiment_10(r: int = 30):
    """
    Experiment 10 · k = 2 · binding capacity  (averaged over r repetitions)

    Two cost structures
        1) balanced   :  b₁ = b₂ = h₁ = h₂ = 10
        2) unbalanced :  (b₁,h₁) = (50,5) , (b₂,h₂) = (10,10)

    For every sample size m we run `r` independent replications,
    compute the L2-relative error w.r.t. a large-sample reference
    solution, and report the mean ± st.dev.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # ------------------------- large reference sample (ground truth for SAA)
    m_ref  = 200_000
    mu1,  sigma1  = 4.0, 0.5     # demand location 1  ~  log-normal
    mu2,  sigma2  = 3.5, 0.4     # demand location 2

    ref_samples = [np.random.lognormal(mu1, sigma1, m_ref),
                   np.random.lognormal(mu2, sigma2, m_ref)]

    # balanced costs
    b_bal, h_bal = [10.0, 10.0], [10.0, 10.0]
    # unbalanced costs
    b_unb, h_unb = [50.0, 10.0], [ 5.0, 10.0]

    # ------------------------- choose a binding capacity
    median1, median2 = np.exp(mu1), np.exp(mu2)
    Q = 0.8 * (median1 + median2)          # 80 % of median-sum  ⇒  binding

    print("\n=== EXPERIMENT 10 :  k = 2  ·  binding Q  ·  r =",
          r, "repetitions per m ===")
    print(f"Capacity Q = {Q:.2f}")

    # reference optimal allocations
    q_ref_bal = find_optimal_allocation(ref_samples, b_bal,  h_bal,  Q)
    q_ref_unb = find_optimal_allocation(ref_samples, b_unb, h_unb, Q)
    print("\nReference solution (balanced)  :", np.round(q_ref_bal, 3),
          "   sum =", sum(q_ref_bal))
    print("Reference solution (unbalanced):", np.round(q_ref_unb, 3),
          "   sum =", sum(q_ref_unb))

    # ------------------------- sample sizes to test
    m_vals = [5, 10, 20, 50, 100, 200, 500,
              1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

    # containers
    bal_mu, bal_sd = [], []
    unb_mu, unb_sd = [], []

    # ------------------------- main loop over m
    for m in m_vals:
        e_bal, e_unb = [], []

        for _ in range(r):
            # generate fresh samples
            samples = [np.random.lognormal(mu1, sigma1, m),
                       np.random.lognormal(mu2, sigma2, m)]

            # solve both cost structures
            q_bal = find_optimal_allocation(samples, b_bal,  h_bal,  Q)
            q_unb = find_optimal_allocation(samples, b_unb, h_unb, Q)

            relevant_indices_bal = np.argwhere(q_ref_bal != 0.0).flatten()
            relevant_indices_unb = np.argwhere(q_ref_unb != 0.0).flatten()

            # L2 relative error
            err_bal = np.linalg.norm((np.array(q_bal)[relevant_indices_bal] - q_ref_bal[relevant_indices_bal])
                                      / q_ref_bal[relevant_indices_bal]) / np.sqrt(2)
            err_unb = np.linalg.norm((np.array(q_unb)[relevant_indices_unb] - q_ref_unb[relevant_indices_unb])
                                      / q_ref_unb[relevant_indices_unb]) / np.sqrt(2)

            e_bal.append(err_bal)
            e_unb.append(err_unb)

        bal_mu.append(np.mean(e_bal));  bal_sd.append(np.std(e_bal))
        unb_mu.append(np.mean(e_unb));  unb_sd.append(np.std(e_unb))

        print(f"m = {m:7d} |  "
              f"balanced μ±σ = {bal_mu[-1]:.3e} ± {bal_sd[-1]:.3e}   |  "
              f"unbalanced μ±σ = {unb_mu[-1]:.3e} ± {unb_sd[-1]:.3e}")

    # ------------------------- plot mean ± 1 s.d.
    plt.figure(figsize=(6, 4))
    plt.errorbar(m_vals, bal_mu, yerr=bal_sd, fmt='o-', capsize=3,
                 label='balanced costs')
    plt.errorbar(m_vals, unb_mu, yerr=unb_sd, fmt='s-', capsize=3,
                 label='unbalanced costs')

    # reference slope  m^{-1/2}
    ref_x = np.array([m_vals[0], m_vals[-1]])
    ref_y = bal_mu[0] * (ref_x / m_vals[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel('mean L2 relative error', fontsize=14)
    plt.title('Experiment 10  ·  k = 2, binding capacity', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment10_convergence.png', dpi=300)
    plt.show()

    # ------------------------- empirical convergence slopes
    slope_bal = np.mean([ np.log(bal_mu[i-1]/bal_mu[i]) /
                          np.log(m_vals[i]/m_vals[i-1])
                          for i in range(1, len(m_vals)) ])
    slope_unb = np.mean([ np.log(unb_mu[i-1]/unb_mu[i]) /
                          np.log(m_vals[i]/m_vals[i-1])
                          for i in range(1, len(m_vals)) ])

    print("\nEmpirical average slopes:")
    print(f"  balanced   ≈  m^{slope_bal:.3f}")
    print(f"  unbalanced ≈  m^{slope_unb:.3f}")

    return dict(m_vals=m_vals,
                bal_mean=bal_mu, bal_std=bal_sd, slope_bal=slope_bal,
                unb_mean=unb_mu, unb_std=unb_sd, slope_unb=slope_unb,
                q_ref_bal=q_ref_bal, q_ref_unb=q_ref_unb, Q=Q)

def run_experiment_11(r: int = 30,
                      rho_levels=(0.0, 0.9, -0.9)):
    """
    Experiment 11 – influence of demand *correlation* on SAA convergence
    (k = 2, binding capacity)

    Three correlation regimes are compared
        • independent  (ρ = 0)
        • highly  positive  (ρ ≈ +0.9)
        • highly *negative* (ρ ≈ –0.9)

    For every sample size m we draw r independent batches of size m,
    solve the SAA problem and average the absolute-error-over-capacity.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools     import cycle

    # ---------- helpers ----------------------------------------------------
    def _sample_demand(size: int, rho: float) -> list[np.ndarray]:
        """
        Draw `size` correlated log-normal pairs (D₁,D₂) with
        given Pearson correlation ρ on the *normal* scale.
        """
        mean = np.array([mu1,   mu2])
        sdev = np.array([sigma1, sigma2])
        cov  = np.array([[sdev[0]**2,       rho*sdev[0]*sdev[1]],
                         [rho*sdev[0]*sdev[1], sdev[1]**2      ]])
        z    = np.random.multivariate_normal(mean, cov, size)          # (size,2)
        d    = np.exp(z)                                                # log-normal
        return [d[:, 0], d[:, 1]]                                       # list of len = 2

    def _err_over_Q(q_hat, q_ref):
        return np.linalg.norm(np.array(q_hat) - q_ref) / Q              # absolute / Q

    # ---------- fixed problem data ----------------------------------------
    np.random.seed(42)
    mu1, sigma1 = 4.0, 0.5
    mu2, sigma2 = 3.5, 0.4

    b_vec = h_vec = [10.0, 10.0]                # balanced costs  (b = h)
    median1, median2 = np.exp(mu1), np.exp(mu2)
    Q = 0.8 * (median1 + median2)               # ensure capacity binds

    m_ref  = 200_000                            # very large reference sample
    m_vals = [5, 10, 20, 50, 100, 200, 500,
              1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

    print("\n=== EXPERIMENT 11 :  correlation & convergence  "
          f"(r = {r} repetitions) ===")
    print(f"Capacity Q = {Q:.2f}")

    # ---------- reference solutions for each ρ -----------------------------
    q_ref = {}
    for rho in rho_levels:
        ref_samples = _sample_demand(m_ref, rho)
        q_ref[rho]  = find_optimal_allocation(ref_samples, b_vec, h_vec, Q)
        print(f"ρ = {rho:>5.2f}  →  q* = {np.round(q_ref[rho], 3)}")

    # containers
    err_mean = {rho: [] for rho in rho_levels}
    err_std  = {rho: [] for rho in rho_levels}
    slopes   = {}

    # ---------- main loop over m ------------------------------------------
    for m in m_vals:
        rep_err = {rho: [] for rho in rho_levels}

        for _ in range(r):
            for rho in rho_levels:
                samples = _sample_demand(m, rho)
                q_hat   = find_optimal_allocation(samples, b_vec, h_vec, Q)
                rep_err[rho].append(_err_over_Q(q_hat, q_ref[rho]))

        for rho in rho_levels:
            err_mean[rho].append(np.mean(rep_err[rho]))
            err_std [rho].append(np.std (rep_err[rho]))

        prog = "  |  ".join(
            f"ρ={rho:+.2f}: μ={err_mean[rho][-1]:.3e}"
            for rho in rho_levels)
        print(f"m = {m:6d}  |  {prog}")

    # ---------- plotting ---------------------------------------------------
    plt.figure(figsize=(6, 4))
    marks = cycle(['o', 's', 'v', '^', 'D', 'X'])
    for rho, mk in zip(rho_levels, marks):
        plt.errorbar(m_vals, err_mean[rho], yerr=err_std[rho],
                     fmt=f'{mk}-', capsize=3,
                     label=f'ρ = {rho:+.2f}')

    ref_x = np.array([m_vals[0], m_vals[-1]])
    ref_y = err_mean[rho_levels[0]][0] * (ref_x / m_vals[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'mean allocation error  $\|q\!\-\!q^*\|_2 / Q$', fontsize=14)
    plt.title('SAA convergence – effect of demand correlation', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment11_correlation_convergence.png', dpi=300)
    plt.show()

    # ---------- empirical slopes ------------------------------------------
    for rho in rho_levels:
        slopes[rho] = np.mean([
            np.log(err_mean[rho][i-1]/err_mean[rho][i]) /
            np.log(m_vals[i]           / m_vals[i-1])
            for i in range(1, len(m_vals))
        ])
    print("\nEmpirical average slopes:")
    for rho in rho_levels:
        print(f"   ρ = {rho:+.2f}  →  m^{slopes[rho]:.3f}")

    return dict(m_vals  = m_vals,
                rho     = rho_levels,
                err_mean= err_mean,
                err_std = err_std,
                slopes  = slopes,
                q_ref   = q_ref,
                Q       = Q)

def run_experiment_12(r: int = 40,
                      bh_pairs=((100, 1), (1, 100), (100, 100), (1, 1)),
                      h_base: float = 1.0,
                      error_type: str = "relative"):
    """
    Experiment 12 — k = 1, convergence for specific (b, h) pairs.

    Compares convergence for:
        (b, h) = (100, 1), (100, 100), (1, 100), (1, 1)

    For every sample size m, run `r` independent repetitions,
    average the relative error, and plot mean ± 1 s.d.

    Additionally, plots the error in terms of |cost(hat q) - cost(q*)|,
    where cost is estimated using a very large sample from the true distribution.

    Parameters
    ----------
    r        : int
        Number of repetitions per sample size m.
    bh_pairs : iterable of tuples
        List of (b, h) pairs to test.
    h_base   : float
        Not used, kept for interface consistency.
    error_type : str
        "relative" or "absolute"
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # Problem setup
    np.random.seed(42)
    mu, sigma = 4.0, 0.5                    # log-normal demand parameters
    Q_big     = 1_000_000.0                 # capacity never binds

    m_ref  = 200_000                        # "ground truth" sample size
    m_vals = [5, 10, 20, 50, 100, 200,
              500, 1_000, 2_000, 5_000,
              10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 12   (r = {r})   for (b, h) pairs ===")

    # generate one huge reference sample once (re-used for all pairs)
    ref_samples = [np.random.lognormal(mu, sigma, m_ref)]

    # Containers
    err_mean = {pair: [] for pair in bh_pairs}
    err_std  = {pair: [] for pair in bh_pairs}
    cost_err_mean = {pair: [] for pair in bh_pairs}
    cost_err_std  = {pair: [] for pair in bh_pairs}
    slopes   = {}

    # Reference solutions for each (b, h) pair
    q_ref = {}
    for b, h in bh_pairs:
        q_ref[(b, h)] = find_optimal_allocation(ref_samples, [b], [h], Q_big)[0]
        print(f"(b, h) = ({b:>3}, {h:>3})  ⇒  q* = {q_ref[(b, h)]:.3f}")

    # Large sample for cost estimation (true distribution)
    cost_eval_samples = np.random.lognormal(mu, sigma, 500_000)

    def empirical_cost(q, b, h, demand_samples):
        # q, b, h are scalars; demand_samples is 1D array
        over = np.maximum(q - demand_samples, 0)
        under = np.maximum(demand_samples - q, 0)
        return np.mean(b * under + h * over)

    # Main loop over m
    for m in m_vals:
        rep_errors = {pair: [] for pair in bh_pairs}
        rep_rel_cost_errors = {pair: [] for pair in bh_pairs}

        for _ in range(r):
            samples = [np.random.lognormal(mu, sigma, m)]

            for b, h in bh_pairs:
                q_hat = find_optimal_allocation(samples, [b], [h], Q_big)[0]
                if error_type == "relative":
                    rel_err = abs(q_hat - q_ref[(b, h)]) / q_ref[(b, h)]
                elif error_type == "absolute":
                    rel_err = abs(q_hat - q_ref[(b, h)])
                rep_errors[(b, h)].append(rel_err)

                # Cost error: |cost(q_hat) - cost(q_ref)|
                cost_hat = empirical_cost(q_hat, b, h, cost_eval_samples)
                cost_star = empirical_cost(q_ref[(b, h)], b, h, cost_eval_samples)
                rep_rel_cost_errors[(b, h)].append(abs(cost_hat - cost_star) / cost_star)

        # aggregate mean / std for this m
        for pair in bh_pairs:
            err_mean[pair].append(np.mean(rep_errors[pair]))
            err_std [pair].append(np.std (rep_errors[pair]))
            cost_err_mean[pair].append(np.mean(rep_rel_cost_errors[pair]))
            cost_err_std[pair].append(np.std (rep_rel_cost_errors[pair]))

        # pretty progress line
        prog = "  |  ".join(
            f"(b={b:>3},h={h:>3}): μ={err_mean[(b,h)][-1]:.3e}, μΔC={cost_err_mean[(b,h)][-1]:.3e}"
            for (b, h) in bh_pairs)
        print(f"m = {m:6d}  |  {prog}")

    # Plotting: relative error
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'P', 'X'])
    for (b, h), mark in zip(bh_pairs, markers):
        # plt.errorbar(m_vals, err_mean[(b, h)], yerr=err_std[(b, h)],
        #              fmt=f'{mark}-', capsize=3,
        #              label=f'(b={b}, h={h})')
        if (b, h) == (100, 100):
            plt.plot(m_vals, err_mean[(b, h)], label=f'(b={b}, h={h})', lw=2)
        else:
            plt.plot(m_vals, err_mean[(b, h)], label=f'(b={b}, h={h})', lw=1)

    # reference m^{-1/2} line from the *first* pair
    ref_x = np.array([m_vals[0], m_vals[-1]])
    first_pair = bh_pairs[0]
    ref_y = err_mean[first_pair][0] * (ref_x / m_vals[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')
    title_string = r'$\; \mathbb{E}[\,|\hat q - q^*|\,/\,q^*]$'
    if error_type == "absolute":
        title_string = r'$\; \mathbb{E}[\,|\hat q - q^*|\,]$'
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(title_string, fontsize=14)
    plt.title('Relative allocation error, log-normal distribution')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment12_bh_convergence.png', dpi=300)
    plt.show()

    # Plotting: cost error
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'P', 'X'])
    for (b, h), mark in zip(bh_pairs, markers):
            # plt.errorbar(m_vals, cost_err_mean[(b, h)], yerr=cost_err_std[(b, h)],
            #             fmt=f'{mark}-', capsize=3,
            #             label=f'(b={b}, h={h})')
        if (b, h) == (100, 100):
            plt.plot(m_vals, cost_err_mean[(b, h)], label=f'(b={b}, h={h})', lw=2)
        else:
            plt.plot(m_vals, cost_err_mean[(b, h)], label=f'(b={b}, h={h})', lw=1)
    # reference m^{-1/2} line from the *first* pair, scaled to cost error
    # ref_y_cost = cost_err_mean[first_pair][0] * (ref_x / m_vals[0])**(-0.5)
    # plt.loglog(ref_x, ref_y_cost, 'k--', label=r'$O(m^{-1/2})$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel( r'$\; \mathbb{E}[\,|C(\hat q) - C(q^*)| / C(q^*)\,]$', fontsize=14)
    plt.title('Relative cost error, log-normal distribution', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment12_bh_rel_cost_error.png', dpi=300)
    plt.show()

    # Slopes
    for pair in bh_pairs:
        slopes[pair] = np.mean([
            np.log(err_mean[pair][i-1] / err_mean[pair][i]) /
            np.log(m_vals[i]          /  m_vals[i-1])
            for i in range(1, len(m_vals))
        ])
    print("\nEmpirical average slopes:")
    for pair in bh_pairs:
        print(f"   (b={pair[0]:>3}, h={pair[1]:>3}) :  ≈  m^{slopes[pair]:.3f}")

    return dict(m_vals=m_vals,
                bh_pairs=bh_pairs,
                err_mean=err_mean,
                err_std =err_std,
                cost_err_mean=cost_err_mean,
                cost_err_std=cost_err_std,
                slopes  =slopes,
                q_ref   =q_ref)



def run_experiment_12_version2(r: int = 40,
                      bh_pairs=((100, 1), (1, 100), (100, 100), (1, 1)),
                      h_base: float = 1.0,
                      error_type: str = "relative"):
    """
    Experiment 12 — k = 1, convergence for specific (b, h) pairs.

    Compares convergence for:
        (b, h) = (100, 1), (100, 100), (1, 100), (1, 1)

    For every sample size m, run `r` independent repetitions,
    average the relative error, and plot mean ± 1 s.d.

    Additionally, plots the error in terms of |cost(hat q) - cost(q*)|,
    where cost is estimated using a very large sample from the true distribution.

    Parameters
    ----------
    r        : int
        Number of repetitions per sample size m.
    bh_pairs : iterable of tuples
        List of (b, h) pairs to test.
    h_base   : float
        Not used, kept for interface consistency.
    error_type : str
        "relative" or "absolute"
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # Problem setup
    np.random.seed(42)
    mu, sigma = 4.0, 0.5                    # log-normal demand parameters
    Q_big     = 1_000_000.0                 # capacity never binds

    m_ref  = 200_000                        # "ground truth" sample size
    m_vals = [5, 10, 20, 50, 100, 200,
              500, 1_000, 2_000, 5_000,
              10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 12   (r = {r})   for (b, h) pairs ===")

    # generate one huge reference sample once (re-used for all pairs)
    triangle_left = 0.0
    triangle_mode = 50.0
    triangle_right = 100.0
    ref_samples = [np.random.triangular(triangle_left, triangle_mode, triangle_right, m_ref)]

    # Containers
    err_mean = {pair: [] for pair in bh_pairs}
    err_std  = {pair: [] for pair in bh_pairs}
    cost_err_mean = {pair: [] for pair in bh_pairs}
    cost_err_std  = {pair: [] for pair in bh_pairs}
    slopes   = {}

    # Reference solutions for each (b, h) pair
    q_ref = {}
    for b, h in bh_pairs:
        q_ref[(b, h)] = find_optimal_allocation(ref_samples, [b], [h], Q_big)[0]
        print(f"(b, h) = ({b:>3}, {h:>3})  ⇒  q* = {q_ref[(b, h)]:.3f}")

    # Large sample for cost estimation (true distribution)
    cost_eval_samples = np.random.triangular(triangle_left, triangle_mode, triangle_right, 500_000)

    def empirical_cost(q, b, h, demand_samples):
        # q, b, h are scalars; demand_samples is 1D array
        over = np.maximum(q - demand_samples, 0)
        under = np.maximum(demand_samples - q, 0)
        return np.mean(b * under + h * over)

    # Main loop over m
    for m in m_vals:
        rep_errors = {pair: [] for pair in bh_pairs}
        rep_rel_cost_errors = {pair: [] for pair in bh_pairs}

        for _ in range(r):
            samples = [np.random.triangular(triangle_left, triangle_mode, triangle_right, m)]

            for b, h in bh_pairs:
                q_hat = find_optimal_allocation(samples, [b], [h], Q_big)[0]
                if error_type == "relative":
                    rel_err = abs(q_hat - q_ref[(b, h)]) / q_ref[(b, h)]
                elif error_type == "absolute":
                    rel_err = abs(q_hat - q_ref[(b, h)])
                rep_errors[(b, h)].append(rel_err)

                # Cost error: |cost(q_hat) - cost(q_ref)|
                cost_hat = empirical_cost(q_hat, b, h, cost_eval_samples)
                cost_star = empirical_cost(q_ref[(b, h)], b, h, cost_eval_samples)
                rep_rel_cost_errors[(b, h)].append(abs(cost_hat - cost_star) / cost_star)

        # aggregate mean / std for this m
        for pair in bh_pairs:
            err_mean[pair].append(np.mean(rep_errors[pair]))
            err_std [pair].append(np.std (rep_errors[pair]))
            cost_err_mean[pair].append(np.mean(rep_rel_cost_errors[pair]))
            cost_err_std[pair].append(np.std (rep_rel_cost_errors[pair]))

        # pretty progress line
        prog = "  |  ".join(
            f"(b={b:>3},h={h:>3}): μ={err_mean[(b,h)][-1]:.3e}, μΔC={cost_err_mean[(b,h)][-1]:.3e}"
            for (b, h) in bh_pairs)
        print(f"m = {m:6d}  |  {prog}")

    # Plotting: relative error
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'P', 'X'])
    for (b, h), mark in zip(bh_pairs, markers):
        # plt.errorbar(m_vals, err_mean[(b, h)], yerr=err_std[(b, h)],
        #              fmt=f'{mark}-', capsize=3,
        #              label=f'(b={b}, h={h})')
        if (b, h) == (100, 100):
            plt.plot(m_vals, err_mean[(b, h)], label=f'(b={b}, h={h})', lw=2)
        else:
            plt.plot(m_vals, err_mean[(b, h)], label=f'(b={b}, h={h})', lw=1)

    # reference m^{-1/2} line from the *first* pair
    ref_x = np.array([m_vals[0], m_vals[-1]])
    first_pair = bh_pairs[0]
    ref_y = err_mean[first_pair][0] * (ref_x / m_vals[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')
    title_string = r'$\; \mathbb{E}[\,|\hat q - q^*|\,/\,q^*]$'
    if error_type == "absolute":
        title_string = r'$\; \mathbb{E}[\,|\hat q - q^*|\,]$'
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(title_string, fontsize=14)
    plt.title('Relative allocation error, triangular distribution')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment12_version2_bh_convergence.png', dpi=300)
    plt.show()

    # Plotting: cost error
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'P', 'X'])
    for (b, h), mark in zip(bh_pairs, markers):
            # plt.errorbar(m_vals, cost_err_mean[(b, h)], yerr=cost_err_std[(b, h)],
            #             fmt=f'{mark}-', capsize=3,
            #             label=f'(b={b}, h={h})')
        if (b, h) == (100, 100):
            plt.plot(m_vals, cost_err_mean[(b, h)], label=f'(b={b}, h={h})', lw=2)
        else:
            plt.plot(m_vals, cost_err_mean[(b, h)], label=f'(b={b}, h={h})', lw=1)
    # reference m^{-1/2} line from the *first* pair, scaled to cost error
    # ref_y_cost = cost_err_mean[first_pair][0] * (ref_x / m_vals[0])**(-0.5)
    # plt.loglog(ref_x, ref_y_cost, 'k--', label=r'$O(m^{-1/2})$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel( r'$\; \mathbb{E}[\,|C(\hat q) - C(q^*)| / C(q^*)\,]$', fontsize=14)
    plt.title('Relative cost error, triangular distribution', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment12_version2_bh_rel_cost_error.png', dpi=300)
    plt.show()

    # Slopes
    for pair in bh_pairs:
        slopes[pair] = np.mean([
            np.log(err_mean[pair][i-1] / err_mean[pair][i]) /
            np.log(m_vals[i]          /  m_vals[i-1])
            for i in range(1, len(m_vals))
        ])
    print("\nEmpirical average slopes:")
    for pair in bh_pairs:
        print(f"   (b={pair[0]:>3}, h={pair[1]:>3}) :  ≈  m^{slopes[pair]:.3f}")

    return dict(m_vals=m_vals,
                bh_pairs=bh_pairs,
                err_mean=err_mean,
                err_std =err_std,
                cost_err_mean=cost_err_mean,
                cost_err_std=cost_err_std,
                slopes  =slopes,
                q_ref   =q_ref)



def run_experiment_13(
        r: int = 40,
        bh_pairs=((100, 1), (100, 100), (1, 100), (1, 1)),
        alpha: float = 5.0,
        beta:  float = 5.0,
        scale: float = 100.0):
    """
    Experiment 13 — k = 1, symmetric demand distribution.

    Demand ~ scale·Beta(alpha, beta) with alpha = beta (symmetric).
    Everything else (b,h pairs, error metrics, plots) mirrors
    Experiment 12.

    Parameters
    ----------
    r        : int
        Repetitions per sample size m.
    bh_pairs : iterable[(b,h)]
        Lost–sales / holding cost pairs to compare.
    alpha, beta : float
        Shape parameters of the Beta distribution (alpha=beta ⇒ symmetry).
    scale : float
        Multiplicative factor to bring the support to [0,scale].
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # ------------- helper --------------------------------------------------
    def beta_samples(size: int) -> np.ndarray:
        """Scaled symmetric Beta(α,β) draws, shape (size,)."""
        return scale * np.random.beta(alpha, beta, size)

    def empirical_cost(q, b, h, demand):
        over  = np.maximum(q - demand, 0)
        under = np.maximum(demand - q, 0)
        return np.mean(b * under + h * over)

    # ------------- fixed settings -----------------------------------------
    np.random.seed(42)
    Q_big   = 1_000_000.0                  # capacity never binds
    m_ref   = 200_000                      # reference sample size
    m_vals  = [5, 10, 20, 50, 100, 200,
               500, 1_000, 2_000, 5_000,
               10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 13  (r = {r})  —  symmetric Beta demand ===")
    print(f"Beta(α={alpha}, β={beta})")
    print(f"scale = {scale}")

    
def run_experiment_13_symm(
        r: int = 40,
        bh_pairs=((100, 1), (100, 100), (1, 100), (1, 1)),
        tri_left: float = 0.0,
        tri_mode: float = 50.0,
        tri_right: float = 100.0):
    """
    Experiment 13 — k = 1, symmetric demand.

    Demand D  ~  Triangular(left, mode, right)  with mode exactly in the
    middle.  All other mechanics follow experiment 12:

        • four (b,h) pairs
        • mean relative allocation error  |q̂−q*|/q*
        • mean *cost* error  |C(q̂)−C(q*)|
        • r repetitions per sample size, results shown with error bars
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # ---------------------------------------------------  helpers
    def sample_demand(size: int) -> np.ndarray:
        return np.random.triangular(tri_left, tri_mode, tri_right, size)

    def empirical_cost(q: float, b: float, h: float,
                       demand_samples: np.ndarray) -> float:
        over   = np.maximum(q - demand_samples, 0.)
        under  = np.maximum(demand_samples - q, 0.)
        return np.mean(b * under + h * over)

    # ---------------------------------------------------  fixed set-up
    np.random.seed(42)
    Q_big   = 1_000_000.0                 # capacity never binds
    m_ref   = 200_000                     # "ground truth" sample size
    m_vals  = [5, 10, 20, 50, 100, 200,
               500, 1_000, 2_000, 5_000,
               10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 13  –  symmetric triangular demand "
          f"(r = {r}) ===")

    # reference sample & solutions
    ref_samples = [sample_demand(m_ref)]
    q_ref = {}
    for b, h in bh_pairs:
        q_ref[(b, h)] = find_optimal_allocation(ref_samples, [b], [h], Q_big)[0]
        print(f"(b,h)=({b:3},{h:3})  →  q* = {q_ref[(b,h)]:.3f}")

    # very large evaluation sample for cost estimation
    eval_samples = sample_demand(500_000)

    # containers
    err_mean, err_std       = {p: [] for p in bh_pairs}, {p: [] for p in bh_pairs}
    cost_mean, cost_std     = {p: [] for p in bh_pairs}, {p: [] for p in bh_pairs}

    # ---------------------------------------------------  loops
    for m in m_vals:
        batched_err  = {p: [] for p in bh_pairs}
        batched_cost = {p: [] for p in bh_pairs}

        # r repetitions
        for _ in range(r):
            samp = [sample_demand(m)]
            for b, h in bh_pairs:
                q_hat   = find_optimal_allocation(samp, [b], [h], Q_big)[0]
                # allocation error
                batched_err[(b, h)].append(abs(q_hat - q_ref[(b, h)]) / q_ref[(b, h)])
                # cost error
                C_hat   = empirical_cost(q_hat,          b, h, eval_samples)
                C_star  = empirical_cost(q_ref[(b, h)],  b, h, eval_samples)
                batched_cost[(b, h)].append(abs(C_hat - C_star))

        # aggregate statistics
        for pair in bh_pairs:
            err_mean [pair].append(np.mean(batched_err [pair]))
            err_std  [pair].append(np.std (batched_err [pair]))
            cost_mean[pair].append(np.mean(batched_cost[pair]))
            cost_std [pair].append(np.std (batched_cost[pair]))

        # progress line
        prog = " | ".join(
            f"(b={b:3},h={h:3}) μ={err_mean[(b,h)][-1]:.2e}"
            for (b,h) in bh_pairs)
        print(f"m={m:6d} | {prog}")

    # ---------------------------------------------------  plots
    markers = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    # 1) relative allocation error
    plt.figure(figsize=(6, 4))
    for (b,h), m in zip(bh_pairs, markers):
        plt.errorbar(m_vals, err_mean[(b,h)], yerr=err_std[(b,h)],
                     fmt=f'{m}-', capsize=3, label=f'(b={b},h={h})')
    ref_x = np.array([m_vals[0], m_vals[-1]])
    ref_y = err_mean[bh_pairs[0]][0]*(ref_x/ref_x[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\,\mathbb{E}[| \hat q - q^*|/q^* ]$', fontsize=14)
    plt.title('Experiment 13 – allocation error (symmetric demand)', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment13_alloc_error.png', dpi=300)
    plt.show()

    # 2) cost error
    plt.figure(figsize=(6, 4))
    markers = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    for (b,h), m in zip(bh_pairs, markers):
        plt.errorbar(m_vals, cost_mean[(b,h)], yerr=cost_std[(b,h)],
                     fmt=f'{m}-', capsize=3, label=f'(b={b},h={h})')
    ref_yc = cost_mean[bh_pairs[0]][0]*(ref_x/ref_x[0])**(-0.5)
    plt.loglog(ref_x, ref_yc, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\,\mathbb{E}[|C(\hat q)-C(q^*)|\,]$', fontsize=14)
    plt.title('Experiment 13 – cost error (symmetric demand)', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment13_cost_error.png', dpi=300)
    plt.show()

    # return raw results
    return dict(m_vals=m_vals,
                bh_pairs=bh_pairs,
                err_mean=err_mean, err_std=err_std,
                cost_mean=cost_mean, cost_std=cost_std,
                q_ref=q_ref)

def run_experiment_14_symm_ratios(
        r: int = 40,
        ratios=(1, 10, 100, 1000),
        h_base: float = 1.0,
        tri_left: float = 0.0,
        tri_mode: float = 50.0,
        tri_right: float = 100.0):
    """
    Experiment 14 – symmetric demand, varying b/h ratios (k = 1).

    Demand  D ~ Triangular(left, mode, right) with mode in the middle,
    hence the pdf is *symmetric* about  (left+right)/2.

    We investigate how the ratio ρ = b/h ∈ {1, 10, 100, 1000}
    affects convergence of

        • relative allocation error      |q̂ − q*| / q*
        • cost error                     |C(q̂) − C(q*)|

    Both errors are averaged over `r` independent repetitions for each
    sample size m and displayed with ±1 s.d. error bars.

    Parameters
    ----------
    r        : int
        Number of repetitions per sample size m.
    ratios   : iterable
        Desired b/h ratios (holding cost fixed at ``h_base``).
    h_base   : float
        Holding cost h (kept constant);  b = ratio * h_base.
    tri_left, tri_mode, tri_right : float
        Parameters of the symmetric triangular demand distribution.
    """
    # ------------------ imports (local to avoid polluting global ns)
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # ------------------ helper functions
    def sample_demand(size: int) -> np.ndarray:
        """Draw `size` samples from the symmetric triangular distribution."""
        return np.random.triangular(tri_left, tri_mode, tri_right, size)

    def empirical_cost(q: float, b: float, h: float,
                       demand_samples: np.ndarray) -> float:
        """Monte-Carlo estimate of the news-vendor cost at quantity q."""
        over  = np.maximum(q - demand_samples, 0.0)
        under = np.maximum(demand_samples - q, 0.0)
        return np.mean(b * under + h * over)

    # ------------------ experimental set-up
    np.random.seed(42)
    Q_big  = 1_000_000.0                     # capacity never binds
    m_ref  = 200_000                         # "ground-truth" sample size
    m_vals = [5, 10, 20, 50, 100, 200,
              500, 1_000, 2_000, 5_000,
              10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 14 – symmetric demand, b/h ratios {ratios} "
          f"(r = {r}) ===")

    # large reference sample (shared for all ratios)
    ref_samples = [sample_demand(m_ref)]

    # evaluation sample for cost error
    eval_samples = sample_demand(500_000)

    # reference solutions q*
    q_ref = {}
    for rho in ratios:
        b_val, h_val = rho * h_base, h_base
        q_ref[rho]   = find_optimal_allocation(ref_samples,
                                               [b_val], [h_val], Q_big)[0]
        print(f"ρ={rho:4}:  b={b_val:>6.1f}, h={h_val:>4.1f} "
              f"→  q* = {q_ref[rho]:.3f}")

    # containers for statistics
    rel_mean, rel_std   = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    cost_mean, cost_std = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    rel_cost_mean, rel_cost_std = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    slopes              = {}

    # ---------------------------------------------------  loops
    for m in m_vals:
        rep_rel  = {ρ: [] for ρ in ratios}
        rep_cost = {ρ: [] for ρ in ratios}
        rep_rel_cost = {ρ: [] for ρ in ratios}
        for _ in range(r):
            samp = [sample_demand(m)]
            for ρ in ratios:
                b_val, h_val = ρ * h_base, h_base
                q_hat   = find_optimal_allocation(samp, [b_val], [h_val], Q_big)[0]

                # relative allocation error
                rep_rel[ρ].append(abs(q_hat - q_ref[ρ]) / q_ref[ρ])

                # cost error
                C_hat  = empirical_cost(q_hat,     b_val, h_val, eval_samples)
                C_star = empirical_cost(q_ref[ρ], b_val, h_val, eval_samples)
                rep_cost[ρ].append(abs(C_hat - C_star))

                # relative cost error
                rep_rel_cost[ρ].append(abs(C_hat - C_star) / C_star)

        # aggregate stats for current m
        for ρ in ratios:
            rel_mean [ρ].append(np.mean(rep_rel [ρ]))
            rel_std [ρ].append(np.std (rep_rel [ρ]))
            cost_mean[ρ].append(np.mean(rep_cost[ρ]))
            cost_std[ρ].append(np.std (rep_cost[ρ]))
            rel_cost_mean[ρ].append(np.mean(rep_rel_cost[ρ]))
            rel_cost_std[ρ].append(np.std (rep_rel_cost[ρ]))

        # progress print-out
        prog = " | ".join(f"ρ={ρ:>4}: μ={rel_mean[ρ][-1]:.2e}"
                          for ρ in ratios)
        print(f"m={m:6d} | {prog}")

    # ---------------------------------------------------  plots
    markers = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    # 1) relative allocation error
    plt.figure(figsize=(6, 4))
    for ρ, mk in zip(ratios, markers):
            # plt.errorbar(m_vals, rel_mean[ρ], yerr=rel_std[ρ],
            #             fmt=f'{mk}-', capsize=3, label=f'ρ={ρ}')
        plt.plot(m_vals, rel_mean[ρ], label=f'b/h={ρ}')
    ref_x = np.array([m_vals[0], m_vals[-1]])
    ref_y = rel_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\mathbb{E}[\,|\hat q - q^*|/q^*]$', fontsize=14)
    plt.title('Relative allocation error, triangular distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment14_alloc_error.png', dpi=300)
    plt.show()

    # 2) cost error
    # plt.figure(figsize=(6, 4))
    # marker_cycle = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    # for ρ, mk in zip(ratios, marker_cycle):
    #     # plt.errorbar(m_vals, cost_mean[ρ], yerr=cost_std[ρ],
    #     #              fmt=f'{mk}-', capsize=3, label=f'b/h={ρ}')
    #     plt.plot(m_vals, cost_mean[ρ], label=f'b/h={ρ}')
    # # ref_yc = cost_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # # plt.loglog(ref_x, ref_yc, 'k--', label=r'$O(m^{-1/2})$')
    # plt.xscale('log'); plt.yscale('log')
    # plt.grid(True, alpha=0.3)
    # plt.xlabel('sample size  $m$', fontsize=14)
    # plt.ylabel(r'$\mathbb{E}[\,|C(\hat q)-C(q^*)|\,]$', fontsize=14)
    # plt.title('Cost error, triangular distribution', fontsize=12)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.savefig('experiment14_cost_error.png', dpi=300)
    # plt.show()
    
    # 3) relative cost error
    plt.figure(figsize=(6, 4))
    marker_cycle = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    for ρ, mk in zip(ratios, marker_cycle):
        # plt.errorbar(m_vals, cost_mean[ρ], yerr=cost_std[ρ],
        #              fmt=f'{mk}-', capsize=3, label=f'b/h={ρ}')
        plt.plot(m_vals, rel_cost_mean[ρ], label=f'b/h={ρ}')
    # ref_yc = cost_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # plt.loglog(ref_x, ref_yc, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\mathbb{E}[\,|C(\hat q)-C(q^*)|/C(q^*)\,]$', fontsize=14)
    plt.title('Relative cost error, triangular distribution', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment14_rel_cost_error.png', dpi=300)
    plt.show()

    # ------------------ compute slopes for allocation error
    for ρ in ratios:
        slopes[ρ] = np.mean([
            np.log(rel_mean[ρ][i-1]/rel_mean[ρ][i]) /
            np.log(m_vals[i]     /m_vals[i-1])
            for i in range(1, len(m_vals))
        ])

    print("\nEmpirical average slopes (allocation error):")
    for ρ in ratios:
        print(f"   ρ={ρ:>4}  →  m^{slopes[ρ]:.3f}")

    # ------------------ return dictionary of results
    return dict(m_vals=m_vals,
                ratios=ratios,
                rel_mean=rel_mean, rel_std=rel_std,
                cost_mean=cost_mean, cost_std=cost_std,
                slopes=slopes,
                q_ref=q_ref)


def run_experiment_15(
        r: int = 100,
        ratios=(1, 10, 100, 1000),
        h_base: float = 1.0,
        tri_left: float = 0.0,
        tri_mode: float = 50.0,
        tri_right: float = 100.0):
    """
    Experiment 14 – symmetric demand, varying b/h ratios (k = 1).

    Demand  D ~ Triangular(left, mode, right) with mode in the middle,
    hence the pdf is *symmetric* about  (left+right)/2.

    We investigate how the ratio ρ = b/h ∈ {1, 10, 100, 1000}
    affects convergence of

        • relative allocation error      |q̂ − q*| / q*
        • cost error                     |C(q̂) − C(q*)|

    Both errors are averaged over `r` independent repetitions for each
    sample size m and displayed with ±1 s.d. error bars.

    Parameters
    ----------
    r        : int
        Number of repetitions per sample size m.
    ratios   : iterable
        Desired b/h ratios (holding cost fixed at ``h_base``).
    h_base   : float
        Holding cost h (kept constant);  b = ratio * h_base.
    tri_left, tri_mode, tri_right : float
        Parameters of the symmetric triangular demand distribution.
    """
    # ------------------ imports (local to avoid polluting global ns)
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # ------------------ helper functions
    def sample_demand(size: int) -> np.ndarray:
        """Draw `size` samples from the symmetric triangular distribution."""
        np.random.seed(42)
        mu, sigma = 4.0, 0.5   
        return np.random.lognormal(mu, sigma, size)

    def empirical_cost(q: float, b: float, h: float,
                       demand_samples: np.ndarray) -> float:
        """Monte-Carlo estimate of the news-vendor cost at quantity q."""
        over  = np.maximum(q - demand_samples, 0.0)
        under = np.maximum(demand_samples - q, 0.0)
        return np.mean(b * under + h * over)

    # ------------------ experimental set-up
    np.random.seed(42)
    Q_big  = 1_000_000.0                     # capacity never binds
    m_ref  = 200_000                         # "ground-truth" sample size
    m_vals = [5, 10, 20, 50, 100, 200,
              500, 1_000, 2_000, 5_000,
              10_000, 20_000, 50_000]

    print(f"\n=== EXPERIMENT 14 – symmetric demand, b/h ratios {ratios} "
          f"(r = {r}) ===")

    # large reference sample (shared for all ratios)
    ref_samples = [sample_demand(m_ref)]

    # evaluation sample for cost error
    eval_samples = sample_demand(500_000)

    # reference solutions q*
    q_ref = {}
    for rho in ratios:
        b_val, h_val = rho * h_base, h_base
        q_ref[rho]   = find_optimal_allocation(ref_samples,
                                               [b_val], [h_val], Q_big)[0]
        print(f"ρ={rho:4}:  b={b_val:>6.1f}, h={h_val:>4.1f} "
              f"→  q* = {q_ref[rho]:.3f}")

    # containers for statistics
    rel_mean, rel_std   = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    cost_mean, cost_std = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    rel_cost_mean, rel_cost_std = {ρ: [] for ρ in ratios}, {ρ: [] for ρ in ratios}
    slopes              = {}

    # ---------------------------------------------------  loops
    for m in m_vals:
        rep_rel  = {ρ: [] for ρ in ratios}
        rep_cost = {ρ: [] for ρ in ratios}
        rep_rel_cost = {ρ: [] for ρ in ratios}
        for _ in range(r):
            samp = [sample_demand(m)]
            for ρ in ratios:
                b_val, h_val = ρ * h_base, h_base
                q_hat   = find_optimal_allocation(samp, [b_val], [h_val], Q_big)[0]

                # relative allocation error
                rep_rel[ρ].append(abs(q_hat - q_ref[ρ]) / q_ref[ρ])

                # cost error
                C_hat  = empirical_cost(q_hat,     b_val, h_val, eval_samples)
                C_star = empirical_cost(q_ref[ρ], b_val, h_val, eval_samples)
                rep_cost[ρ].append(abs(C_hat - C_star))

                # relative cost error
                rep_rel_cost[ρ].append(abs(C_hat - C_star) / C_star)

        # aggregate stats for current m
        for ρ in ratios:
            rel_mean [ρ].append(np.mean(rep_rel [ρ]))
            rel_std [ρ].append(np.std (rep_rel [ρ]))
            cost_mean[ρ].append(np.mean(rep_cost[ρ]))
            cost_std[ρ].append(np.std (rep_cost[ρ]))
            rel_cost_mean[ρ].append(np.mean(rep_rel_cost[ρ]))
            rel_cost_std[ρ].append(np.std (rep_rel_cost[ρ]))

        # progress print-out
        prog = " | ".join(f"ρ={ρ:>4}: μ={rel_mean[ρ][-1]:.2e}"
                          for ρ in ratios)
        print(f"m={m:6d} | {prog}")

    # ---------------------------------------------------  plots
    markers = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    # 1) relative allocation error
    plt.figure(figsize=(6, 4))
    for ρ, mk in zip(ratios, markers):
            # plt.errorbar(m_vals, rel_mean[ρ], yerr=rel_std[ρ],
            #             fmt=f'{mk}-', capsize=3, label=f'ρ={ρ}')
        plt.plot(m_vals, rel_mean[ρ], label=f'b/h={ρ}')
    ref_x = np.array([m_vals[0], m_vals[-1]])
    ref_y = rel_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # plt.loglog(ref_x, ref_y, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\mathbb{E}[\,|\hat q - q^*|/q^*]$', fontsize=14)
    plt.title('Relative allocation error, log-normal distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment15_alloc_error.png', dpi=300)
    plt.show()

    # 2) cost error
    # plt.figure(figsize=(6, 4))
    # marker_cycle = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    # for ρ, mk in zip(ratios, marker_cycle):
    #     # plt.errorbar(m_vals, cost_mean[ρ], yerr=cost_std[ρ],
    #     #              fmt=f'{mk}-', capsize=3, label=f'b/h={ρ}')
    #     plt.plot(m_vals, cost_mean[ρ], label=f'b/h={ρ}')
    # # ref_yc = cost_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # # plt.loglog(ref_x, ref_yc, 'k--', label=r'$O(m^{-1/2})$')
    # plt.xscale('log'); plt.yscale('log')
    # plt.grid(True, alpha=0.3)
    # plt.xlabel('sample size  $m$', fontsize=14)
    # plt.ylabel(r'$\mathbb{E}[\,|C(\hat q)-C(q^*)|\,]$', fontsize=14)
    # plt.title('Cost error, triangular distribution', fontsize=12)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.savefig('experiment14_cost_error.png', dpi=300)
    # plt.show()
    
    # 3) relative cost error
    plt.figure(figsize=(6, 4))
    marker_cycle = cycle(['o', 's', 'v', '^', 'D', 'X', 'P'])
    for ρ, mk in zip(ratios, marker_cycle):
        # plt.errorbar(m_vals, cost_mean[ρ], yerr=cost_std[ρ],
        #              fmt=f'{mk}-', capsize=3, label=f'b/h={ρ}')
        plt.plot(m_vals, rel_cost_mean[ρ], label=f'b/h={ρ}')
    # ref_yc = cost_mean[ratios[0]][0] * (ref_x / ref_x[0])**(-0.5)
    # plt.loglog(ref_x, ref_yc, 'k--', label=r'$O(m^{-1/2})$')
    plt.xscale('log'); plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('sample size  $m$', fontsize=14)
    plt.ylabel(r'$\mathbb{E}[\,|C(\hat q)-C(q^*)|/C(q^*)\,]$', fontsize=14)
    plt.title('Relative cost error, log-normal distribution', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('experiment15_rel_cost_error.png', dpi=300)
    plt.show()

    # ------------------ compute slopes for allocation error
    for ρ in ratios:
        slopes[ρ] = np.mean([
            np.log(rel_mean[ρ][i-1]/rel_mean[ρ][i]) /
            np.log(m_vals[i]     /m_vals[i-1])
            for i in range(1, len(m_vals))
        ])

    print("\nEmpirical average slopes (allocation error):")
    for ρ in ratios:
        print(f"   ρ={ρ:>4}  →  m^{slopes[ρ]:.3f}")

    # ------------------ return dictionary of results
    return dict(m_vals=m_vals,
                ratios=ratios,
                rel_mean=rel_mean, rel_std=rel_std,
                cost_mean=cost_mean, cost_std=cost_std,
                slopes=slopes,
                q_ref=q_ref)



if __name__ == "__main__":
    # np.random.seed(42)  # For reproducibility
    # run_experiment_1()
    # run_experiment_2()
    # run_experiment_3()
    # run_experiment_7()
    # run_experiment_4([10000], m_reference=50000, plot=False)
    # run_convergence_experiment([2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000], m_reference=100000)
    # run_experiment_9(error_type="relative") # 
    # run_experiment_10()
    # run_experiment_11()
    run_experiment_12(error_type="relative")
    # run_experiment_12_version2(error_type="relative")
    # run_experiment_13_symm()
    # run_experiment_14_symm_ratios()
    # run_experiment_15()