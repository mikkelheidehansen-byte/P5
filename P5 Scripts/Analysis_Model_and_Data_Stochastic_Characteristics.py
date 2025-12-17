import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib import location
from scipy.stats import beta
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
dt_hours = (timestamps.diff().dt.total_seconds() / 3600).median()
fs_per_hour = 1.0 / dt_hours



times = df_data.index  

# Location and timezone
latitude = 48.5486
longitude = 12.6925
tz = times.tz

# Sun position
site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(times)

# Panel parameters
panel_tilt = 25         
panel_azimuth = 120     

# Solar radiation
def irradiance_factor(solpos, tilt, azimuth):
    zenith_rad = np.radians(solpos['zenith'])
    azimuth_rad = np.radians(solpos['azimuth'])
    tilt_rad = np.radians(tilt)
    panel_azimuth_rad = np.radians(azimuth)

    cos_theta = (
        np.cos(zenith_rad) * np.cos(tilt_rad) +
        np.sin(zenith_rad) * np.sin(tilt_rad) *
        np.cos(azimuth_rad - panel_azimuth_rad)
    )
    cos_theta = np.clip(cos_theta, 0, None)
    return cos_theta


irr_factor = irradiance_factor(solpos, panel_tilt, panel_azimuth)
Bt = irr_factor


np.random.seed(0)
alpha = 0.7
a_beta, b_beta = 2, 5

nu_t = beta.rvs(a_beta, b_beta, size=len(times))
C_t = np.zeros(len(times))
C_t[0] = nu_t[0]
for t in range(1, len(times)):
    C_t[t] = alpha * C_t[t-1] + (1 - alpha) * nu_t[t]


Pt_model = (1 - C_t) * Bt

df_model = pd.DataFrame(index=times)
df_model["model_pv"] = Pt_model

df_merged = pd.concat([df_data.rename(columns={"pv_production":"data_pv"}), df_model], axis=1)

df_merged = df_merged.dropna(subset=["data_pv", "model_pv"])

df_merged.to_csv("merged_pv.csv", index=True)


data_series = df_merged["data_pv"].values.astype(float)
model_series = df_merged["model_pv"].values.astype(float)

# ACF
acf_data = compute_acf(data_series)
acf_model = compute_acf(model_series)
max_lags = min(200, len(acf_data), len(acf_model))

# PSD 
freqs_data, psd_data = compute_psd(data_series, fs_per_hour)
freqs_model, psd_model = compute_psd(model_series, fs_per_hour)

# CDF
x_sorted_data, cdf_data = empirical_cdf(data_series)
x_sorted_model, cdf_model = empirical_cdf(model_series)


plt.style.use('default')

# Time plot
plt.figure(figsize=(14,5))
plt.plot(df_merged.index, df_merged["model_pv"], label="Model PV", color="red", alpha=0.9)
plt.plot(df_merged.index, df_merged["data_pv"], label="Data PV", color="blue", alpha=0.9)
plt.xlabel("Time")
plt.ylabel("Power Production")
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ACF 
plt.figure(figsize=(12,4))
lags = np.arange(max_lags)

plt.stem(
    lags + 0.15, acf_model[:max_lags],
    linefmt='r-', markerfmt='ro', basefmt='k-',   
    use_line_collection=True
)
plt.stem(
    lags, acf_data[:max_lags],
    linefmt='b-', markerfmt='bo', basefmt='k-',    
    use_line_collection=True
)


plt.xlabel("Time Lags")
plt.ylabel("ACF")
plt.legend(["Model ACF", "Data ACF"])
plt.grid(True)
plt.tight_layout()
plt.show()

# PSD
plt.figure(figsize=(12,4))
plt.semilogy(freqs_data, psd_data, label="Data PSD", color='blue')
plt.semilogy(freqs_model, psd_model, label="Model PSD", color='red')
plt.xlabel("Frequency (cycles/day)")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# CDF
plt.figure(figsize=(6,4))
plt.plot(x_sorted_data, cdf_data, label="Data CDF", color='blue')
plt.plot(x_sorted_model, cdf_model, label="Model CDF", color='red')
plt.xlabel("Normalized Power Production")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

