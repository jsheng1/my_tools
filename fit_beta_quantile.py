import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize

# Given values
mean_target = np.random.uniform(1e-6, 1e-4)
std_target = mean_target*np.random.uniform(3,10)
k2 = np.array([0.98, 0.99, 0.995])
quantile_target = mean_target + np.sqrt(1/(1-k2)) * std_target  # ≈ 0.00167
print(mean_target, quantile_target)

def log_cosh_error(beta_quantile, quantile_target):
    return np.log(np.cosh(beta_quantile - quantile_target))

def objective(params):
    a, b = params
    # Ensure parameters are positive
    if a <= 0 or b <= 0:
        return np.inf

    quantile = beta.ppf(k2, a, b)

    error = ((quantile - quantile_target)/quantile_target)**2
    error = log_cosh_error(quantile, quantile_target)
    return error.sum()


# Retry mechanism parameters
max_retries = 10
tolerance = 0.1  # 1% relative error tolerance

# Optimization loop with retry
for retry in range(max_retries):
    # Initial guess for alpha and beta with slight randomization
    a0 = np.random.uniform(1e-4, 1e-2)
    b0 = a0*1000
    initial_guess = [a0, b0]
    
    # Bounds to ensure alpha and beta remain positive
    bounds = [(1e-12, None), (1e-12, None)]
    
    # Perform the optimization
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    # Check if optimization was successful
    if not result.success:
        print(f"Optimization failed on attempt {retry + 1}. Retrying...")
        continue
    
    # Extract optimized alpha and beta
    a_opt, b_opt = result.x
    
    # Verify the quantile and relative error
    beta_quantile = beta.ppf(k2, a_opt, b_opt)
    relative_error = np.max(abs((beta_quantile - quantile_target) / quantile_target))

    
    
    # Check if the relative error requirement is met
    if relative_error <= tolerance:
        print(f'Try {retry+1}/{max_retries}: {beta_quantile} {quantile_target} {relative_error}')
        print(f'{initial_guess}, {a_opt}, {b_opt}')
        break

        # print("\nThe relative error between target quantile and optimized quantile is within 1%.")
        # # Check the relative error with sampled data
        # samples = beta.rvs(a_opt, b_opt, size=100000, random_state=123)
        # sample_quantile = np.quantile(samples, 0.995)
        # sample_relative_error = abs((sample_quantile - quantile_target) / quantile_target)

        # print("\nSample statistics from generated data:")
        # print(f"Sample 99.5% Quantile: {sample_quantile}")
        # print(f"Sample Relative Error: {sample_relative_error * 100:.2f}%")
        # break

else:
    print(f"Failed to achieve tolerance: {beta_quantile} {quantile_target} {relative_error}")
