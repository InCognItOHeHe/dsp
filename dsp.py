import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)


# Generate binary random signal (0 or 1)
def generate_binary_signal(length=1000):
    return np.random.randint(0, 2, length)


# Create signal
signal_length = 10000
binary_signal = generate_binary_signal(signal_length)

# ----- Basic Statistics -----
# Compute mean
signal_mean = np.mean(binary_signal)
print(f"Signal Mean: {signal_mean:.4f}")

# ----- Histogram Analysis -----
# Count occurrences of 0s and 1s
hist_values, hist_bins = np.histogram(binary_signal, bins=[0, 0.5, 1.5])
hist_df = pd.DataFrame(
    {"Value": [0, 1], "Count": hist_values, "Frequency": hist_values / signal_length}
)
print("\nHistogram:")
print(hist_df)

# ----- Autocorrelation -----
# Compute autocorrelation
max_lag = 50
autocorr = np.correlate(
    binary_signal - signal_mean, binary_signal - signal_mean, mode="full"
)
autocorr = autocorr[len(autocorr) // 2 : len(autocorr) // 2 + max_lag + 1]
autocorr = autocorr / autocorr[0]  # Normalize

# ----- Power Spectral Density Analysis -----
# Estimate PSD using Welch's method with different window sizes
window_sizes = [128, 256, 512, 1024]
psd_results = {}

for window_size in window_sizes:
    # Calculate PSD using Welch's method
    frequencies, psd = signal.welch(
        binary_signal,
        fs=1.0,
        window="hann",
        nperseg=window_size,
        noverlap=window_size // 2,
        scaling="density",
    )
    psd_results[window_size] = (frequencies, psd)

# ----- Visualization -----
plt.figure(figsize=(14, 10))

# Plot 1: Binary Signal (first 100 samples)
plt.subplot(3, 2, 1)
plt.stem(binary_signal[:100], linefmt="b-", markerfmt="bo", basefmt="r-")
plt.title("Binary Random Signal (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Value (0 or 1)")
plt.grid(True)

# Plot 2: Histogram
plt.subplot(3, 2, 2)
plt.bar(["0", "1"], hist_values, color=["skyblue", "salmon"])
plt.title(f"Histogram (Mean = {signal_mean:.4f})")
plt.xlabel("Value")
plt.ylabel("Count")
for i, v in enumerate(hist_values):
    plt.text(i, v + 50, str(v), ha="center")

# Plot 3: Autocorrelation
plt.subplot(3, 2, 3)
plt.stem(range(len(autocorr)), autocorr, linefmt="g-", markerfmt="go", basefmt="r-")
plt.title("Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)

# Plot 4, 5, 6: PSDs with different window sizes
for i, window_size in enumerate(window_sizes):
    if i < 3:  # First 3 window sizes on main figure
        plt.subplot(3, 2, 4 + i)
        frequencies, psd = psd_results[window_size]
        plt.semilogy(frequencies, psd)
        plt.title(f"PSD (Welch, Window Size = {window_size})")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Power/Frequency")
        plt.grid(True)

# Additional figure for comparing all PSDs
plt.figure(figsize=(10, 6))
for window_size in window_sizes:
    frequencies, psd = psd_results[window_size]
    plt.semilogy(frequencies, psd, label=f"Window Size = {window_size}")

plt.title("PSD Comparison with Different Window Sizes")
plt.xlabel("Normalized Frequency")
plt.ylabel("Power/Frequency (log scale)")
plt.legend()
plt.grid(True)

# Show results
plt.tight_layout()
plt.show()

# Summary statistics for PSD
psd_stats = {}
for window_size in window_sizes:
    frequencies, psd = psd_results[window_size]
    psd_stats[window_size] = {
        "Mean": np.mean(psd),
        "Std": np.std(psd),
        "Max": np.max(psd),
        "Min": np.min(psd),
        "Median": np.median(psd),
    }

# Print PSD statistics
print("\nPSD Statistics for Different Window Sizes:")
for window_size, stats in psd_stats.items():
    print(f"\nWindow Size: {window_size}")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value:.6f}")
