import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import arma_generate_sample
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define AR(2) parameters
ar_params = np.array([0.6, -0.3])  # phi_1 = 0.6, phi_2 = -0.3
ar = np.r_[1, -ar_params]  # Add the leading 1

# Generate AR(2) process
n_samples = 500
burnin = 50
x = arma_generate_sample(ar=ar, ma=[1], nsample=n_samples + burnin)
x = x[burnin:]  # Remove burn-in period


# Function to compute Yule-Walker estimates
def yule_walker_method(x, order=2):
    r = np.zeros(order + 1)
    # Compute autocorrelations
    for k in range(order + 1):
        r[k] = np.mean(x[k:] * x[: len(x) - k])

    # Create Toeplitz matrix
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]

    # Solve Yule-Walker equations
    phi = np.linalg.solve(R, r[1 : order + 1])

    # Calculate innovation variance
    sigma2 = r[0] - np.sum(phi * r[1 : order + 1])

    return phi, sigma2


# Implement Burg algorithm properly
def burg_method(x, order=2):
    N = len(x)
    f = np.copy(x)  # Forward prediction error
    b = np.copy(x)  # Backward prediction error
    a = np.ones(order + 1)  # AR coefficients, a[0] = 1

    # Reflection coefficients
    k = np.zeros(order)

    # Error power
    e = np.zeros(order + 1)
    e[0] = np.sum(x**2) / N

    for m in range(order):
        # Compute reflection coefficient
        num = 0
        den = 0
        for n in range(N - m - 1):
            num += f[n + m + 1] * b[n]
            den += f[n + m + 1] ** 2 + b[n] ** 2

        k[m] = -2 * num / den

        # Update AR coefficients
        a_prev = a[: m + 1].copy()
        for i in range(m + 1):
            a[i] = a_prev[i] + k[m] * a_prev[m - i]
        a[m + 1] = k[m]

        # Update forward and backward prediction errors
        f_old = f.copy()
        for n in range(N - m - 1):
            f[n + m + 1] = f_old[n + m + 1] + k[m] * b[n]
            b[n] = b[n] + k[m] * f_old[n + m + 1]

        # Update error power
        e[m + 1] = e[m] * (1 - k[m] ** 2)

    # Convert to standard AR parameters
    ar_params = -a[1:]

    return ar_params, e[-1]


# Estimate parameters using Yule-Walker method
yw_params, yw_var = yule_walker_method(x, order=2)
print(f"True AR(2) parameters: phi_1 = {ar_params[0]}, phi_2 = {ar_params[1]}")
print(f"Yule-Walker estimates: phi_1 = {yw_params[0]:.4f}, phi_2 = {yw_params[1]:.4f}")

# Estimate parameters using Burg method
burg_params, burg_var = burg_method(x, order=2)
print(f"Burg estimates: phi_1 = {burg_params[0]:.4f}, phi_2 = {burg_params[1]:.4f}")


# Calculate residuals for both methods
def calculate_residuals(x, params, order=2):
    residuals = np.zeros(len(x) - order)
    for t in range(order, len(x)):
        pred = np.sum(params * x[t - order : t][::-1])
        residuals[t - order] = x[t] - pred
    return residuals


yw_residuals = calculate_residuals(x, yw_params)
burg_residuals = calculate_residuals(x, burg_params)

# Statistical comparison of residuals
yw_resid_mean = np.mean(yw_residuals)
yw_resid_var = np.var(yw_residuals)
burg_resid_mean = np.mean(burg_residuals)
burg_resid_var = np.var(burg_residuals)

print("\nResidual Statistics:")
print(f"Yule-Walker: Mean = {yw_resid_mean:.6f}, Variance = {yw_resid_var:.6f}")
print(f"Burg: Mean = {burg_resid_mean:.6f}, Variance = {burg_resid_var:.6f}")


# Calculate AIC and BIC for both methods
def calculate_information_criteria(residuals, n_params=2):
    n = len(residuals)
    resid_var = np.var(residuals)
    aic = n * np.log(resid_var) + 2 * n_params
    bic = n * np.log(resid_var) + n_params * np.log(n)
    return aic, bic


yw_aic, yw_bic = calculate_information_criteria(yw_residuals)
burg_aic, burg_bic = calculate_information_criteria(burg_residuals)

print("\nInformation Criteria:")
print(f"Yule-Walker: AIC = {yw_aic:.2f}, BIC = {yw_bic:.2f}")
print(f"Burg: AIC = {burg_aic:.2f}, BIC = {burg_bic:.2f}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Original time series
plt.subplot(3, 2, 1)
plt.plot(x)
plt.title("AR(2) Process")
plt.xlabel("Time")
plt.ylabel("Value")

# Plot 2: Parameter comparison
plt.subplot(3, 2, 2)
methods = ["True", "Yule-Walker", "Burg"]
phi1_values = [ar_params[0], yw_params[0], burg_params[0]]
phi2_values = [ar_params[1], yw_params[1], burg_params[1]]

x_pos = np.arange(len(methods))
width = 0.35
plt.bar(x_pos - width / 2, phi1_values, width, label="phi_1")
plt.bar(x_pos + width / 2, phi2_values, width, label="phi_2")
plt.xticks(x_pos, methods)
plt.ylabel("Parameter Value")
plt.title("AR(2) Parameter Comparison")
plt.legend()

# Plot 3: Residuals time series
plt.subplot(3, 2, 3)
plt.plot(yw_residuals, label="Yule-Walker")
plt.plot(burg_residuals, label="Burg")
plt.title("Residuals")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Plot 4: Residuals histogram
plt.subplot(3, 2, 4)
plt.hist(yw_residuals, bins=30, alpha=0.5, label="Yule-Walker")
plt.hist(burg_residuals, bins=30, alpha=0.5, label="Burg")
plt.title("Residuals Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# Plot 5: Residuals QQ plot for Yule-Walker
plt.subplot(3, 2, 5)
import scipy.stats as stats

stats.probplot(yw_residuals, dist="norm", plot=plt)
plt.title("Yule-Walker Residuals QQ Plot")

# Plot 6: Residuals QQ plot for Burg
plt.subplot(3, 2, 6)
stats.probplot(burg_residuals, dist="norm", plot=plt)
plt.title("Burg Residuals QQ Plot")

plt.tight_layout()
plt.show()


# Run a Monte Carlo simulation to compare methods stability
def monte_carlo_comparison(n_iter=100, n_samples=500):
    yw_params_list = []
    burg_params_list = []

    for i in range(n_iter):
        # Generate new AR(2) process
        x_mc = arma_generate_sample(ar=ar, ma=[1], nsample=n_samples + burnin)
        x_mc = x_mc[burnin:]

        # Estimate parameters
        yw_mc_params, _ = yule_walker_method(x_mc)
        burg_mc_params, _ = burg_method(x_mc)

        yw_params_list.append(yw_mc_params)
        burg_params_list.append(burg_mc_params)

    return np.array(yw_params_list), np.array(burg_params_list)


# Run Monte Carlo simulation
n_iter = 100
print("\nRunning Monte Carlo simulation with 100 iterations...")
yw_mc_results, burg_mc_results = monte_carlo_comparison(n_iter)

# Create summary statistics
yw_mc_mean = np.mean(yw_mc_results, axis=0)
yw_mc_std = np.std(yw_mc_results, axis=0)
burg_mc_mean = np.mean(burg_mc_results, axis=0)
burg_mc_std = np.std(burg_mc_results, axis=0)

print("\nMonte Carlo Results (100 iterations):")
print("Yule-Walker:")
print(f"  phi_1: Mean = {yw_mc_mean[0]:.4f}, Std = {yw_mc_std[0]:.4f}")
print(f"  phi_2: Mean = {yw_mc_mean[1]:.4f}, Std = {yw_mc_std[1]:.4f}")
print("Burg:")
print(f"  phi_1: Mean = {burg_mc_mean[0]:.4f}, Std = {burg_mc_std[0]:.4f}")
print(f"  phi_2: Mean = {burg_mc_mean[1]:.4f}, Std = {burg_mc_std[1]:.4f}")

# Create visualizations for Monte Carlo results
plt.figure(figsize=(12, 6))

# Plot parameter distributions
plt.subplot(1, 2, 1)
plt.scatter(yw_mc_results[:, 0], yw_mc_results[:, 1], alpha=0.5, label="Yule-Walker")
plt.scatter(burg_mc_results[:, 0], burg_mc_results[:, 1], alpha=0.5, label="Burg")
plt.scatter(
    [ar_params[0]], [ar_params[1]], color="red", marker="*", s=200, label="True"
)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Parameter Estimates Distribution (100 iterations)")
plt.legend()
plt.grid(True)

# Plot parameter variances
plt.subplot(1, 2, 2)
parameters = ["phi_1", "phi_2"]
yw_variances = [yw_mc_std[0] ** 2, yw_mc_std[1] ** 2]
burg_variances = [burg_mc_std[0] ** 2, burg_mc_std[1] ** 2]

x_pos = np.arange(len(parameters))
width = 0.35
plt.bar(x_pos - width / 2, yw_variances, width, label="Yule-Walker")
plt.bar(x_pos + width / 2, burg_variances, width, label="Burg")
plt.xticks(x_pos, parameters)
plt.ylabel("Parameter Variance")
plt.title("Parameter Estimation Variance")
plt.legend()

plt.tight_layout()
plt.show()

# Create a dataframe for all results
results_df = pd.DataFrame(
    {
        "Method": ["True", "Yule-Walker", "Burg"],
        "phi_1": [ar_params[0], yw_params[0], burg_params[0]],
        "phi_2": [ar_params[1], yw_params[1], burg_params[1]],
        "Residual Mean": [np.nan, yw_resid_mean, burg_resid_mean],
        "Residual Variance": [np.nan, yw_resid_var, burg_resid_var],
        "AIC": [np.nan, yw_aic, burg_aic],
        "BIC": [np.nan, yw_bic, burg_bic],
        "MC phi_1 (mean)": [np.nan, yw_mc_mean[0], burg_mc_mean[0]],
        "MC phi_2 (mean)": [np.nan, yw_mc_mean[1], burg_mc_mean[1]],
        "MC phi_1 (std)": [np.nan, yw_mc_std[0], burg_mc_std[0]],
        "MC phi_2 (std)": [np.nan, yw_mc_std[1], burg_mc_std[1]],
    }
)

print("\nSummary Results:")
print(results_df.to_string(index=False))
