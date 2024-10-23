import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize

# Given values
mean_target = 0.001
std_target = 0.005
quantile_target = mean_target + np.sqrt(200) * std_target  # ≈ 0.00167

def objective(params):
    a, b = params
    # Ensure parameters are positive
    if a <= 0 or b <= 0:
        return np.inf
    # Compute the mean and variance of the Beta distribution
    mean = a / (a + b)
    var = (a * b) / ((a + b)**2 * (a + b + 1))
    std = np.sqrt(var)
    # Compute the 99.5% quantile
    quantile = beta.ppf(0.995, a, b)
    # Calculate the squared relative errors
    error = ((mean - mean_target)/mean_target)**2 + \
            ((std - std_target)/std_target)**2 + \
            (1e3*(quantile - quantile_target)/quantile_target)**2
    return error

# Initial guess for alpha and beta
initial_guess = [1e-4, 1e2]

# Bounds to ensure alpha and beta remain positive
bounds = [(1e-12, None), (1e-12, None)]

# Perform the optimization to find the best alpha and beta
result = minimize(objective, initial_guess, bounds=bounds, tol=1e-10)

# Extract optimized alpha and beta
a_opt, b_opt = result.x

print("Optimized parameters:")
print("alpha =", a_opt)
print("beta =", b_opt)

# Verify the moments and quantile
mean = a_opt / (a_opt + b_opt)
var = (a_opt * b_opt) / ((a_opt + b_opt)**2 * (a_opt + b_opt + 1))
std = np.sqrt(var)
quantile = beta.ppf(0.995, a_opt, b_opt)

print("\nComputed statistics:")
print(f"Mean: {mean} (Target: {mean_target})")
print(f"Standard Deviation: {std} (Target: {std_target})")
print(f"99.5% Quantile: {quantile} (Target: {quantile_target})")


# Generate samples from the customized Beta distribution
samples = beta.rvs(a_opt, b_opt, size=100000)

# Ensure samples are within [0,1]
samples = np.clip(samples, 0, 1)

# Verify sample statistics
sample_mean = np.mean(samples)
sample_std = np.std(samples)
sample_quantile = np.quantile(samples, 0.995)

print("\nSample statistics from generated data:")
print(f"Sample Mean: {sample_mean}")
print(f"Sample Standard Deviation: {sample_std}")
print(f"Sample 99.5% Quantile: {sample_quantile}")
