import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def compute_acf(x):
    x = np.asarray(x, dtype=float)
    x_c = x - np.mean(x)
    full = np.correlate(x_c, x_c, mode='full')
    acf = full[full.size // 2:]
    acf /= acf[0]
    return acf

def compute_psd(x, fs_per_hour):
    fs_per_day = fs_per_hour * 24.0
    nperseg = min(1024, len(x))
    freqs, psd = welch(x, fs=fs_per_day, nperseg=nperseg)
    return freqs, psd

def empirical_cdf(x):
    x_sorted = np.sort(x)
    cdf = np.linspace(1/len(x_sorted), 1.0, len(x_sorted))
    return x_sorted, cdf


df_data = pd.read_csv("pv_production_june.csv", parse_dates=["timestamp"])
df_data = df_data.sort_values("timestamp")
df_data.set_index("timestamp", inplace=True)

if df_data.index.tz is None:
    df_data.index = df_data.index.tz_localize("Europe/Berlin")

tz = df_data.index.tz

timestamps = df_data.index.to_series()
dt_hours = 0.25
fs_per_hour = 1.0 / dt_hours

data_series = df_data["pv_production"].values.astype(float)


acf_data = compute_acf(data_series)
lags = np.arange(200)

freqs_data, psd_data = compute_psd(data_series, fs_per_hour)

x_sorted_data, cdf_data = empirical_cdf(data_series)


# ACF
plt.figure(figsize=(12,4))
plt.stem(
    lags,
    acf_data[:200],
    linefmt='b-',
    markerfmt='bo',
    basefmt='k-',
    use_line_collection=True
)
plt.xlabel("Time Lags")
plt.ylabel("ACF")
plt.grid(True)
plt.tight_layout()
plt.show()

# PSD
plt.figure(figsize=(12,4))
plt.semilogy(freqs_data, psd_data, color='blue', label="Data PSD")
plt.xlabel("Frequency (cycles/day)")
plt.ylabel("PSD")
plt.grid(True)
plt.tight_layout()
plt.show()

# CDF
plt.figure(figsize=(6,4))
plt.plot(x_sorted_data, cdf_data, color='blue', label="Data CDF")
plt.xlabel("Normalized Power Production")
plt.ylabel("CDF")
plt.grid(True)
plt.tight_layout()
plt.show()
