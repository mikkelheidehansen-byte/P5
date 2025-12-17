import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pvlib import location


latitude = 48.5486
longitude = 12.6925
tz        = 'Europe/Berlin'

panel_tilt    = 25
panel_azimuth = 100
eps = 1e-5
np.random.seed(11)

def irradiance_factor(solpos, tilt_deg, azimuth_deg):
    zen = np.radians(solpos["zenith"])
    azi = np.radians(solpos["azimuth"])
    tilt = np.radians(tilt_deg)
    panel_az = np.radians(azimuth_deg)

    cos_th = (
        np.cos(zen)*np.cos(tilt) +
        np.sin(zen)*np.sin(tilt)*np.cos(azi - panel_az)
    )
    cos_th[cos_th < 0] = 0
    return cos_th

def estimate_params_from_C(C, valid=None):
    C = np.asarray(C)
    if valid is None:
        mask_use = np.ones_like(C, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        mask_use = valid

    C_valid = C[mask_use]
    if len(C_valid) < 10:
        return np.nan, np.nan, np.nan

    meanC = np.mean(C_valid)
    varC  = np.var(C_valid, ddof=1)
    if varC == 0:
        return np.nan, np.nan, np.nan

    if valid is None:
        pair_idx = np.arange(1, len(C))
    else:
        pair_idx = np.where(mask_use[1:] & mask_use[:-1])[0] + 1

    if len(pair_idx) < 5:
        return np.nan, np.nan, np.nan

    x = C[pair_idx]
    y = C[pair_idx - 1]

    n_pairs = len(x)
    gamma1 = np.sum((x - meanC)*(y - meanC)) / (n_pairs - 1)

    alpha_hat = float(np.clip(gamma1 / varC, 0, 0.9999))

    nu_hat = (x - alpha_hat*y) / (1 - alpha_hat)
    nu_hat = np.clip(nu_hat, 0.001, 0.999)

    m = np.mean(nu_hat)
    v = np.var(nu_hat, ddof=1)
    if v <= 0:
        return np.nan, np.nan, alpha_hat

    kappa = m*(1 - m)/v - 1
    if kappa <= 0:
        return np.nan, np.nan, alpha_hat

    a_hat = m * kappa
    b_hat = (1 - m) * kappa
    return a_hat, b_hat, alpha_hat


def empirical_acf(x, max_lag):
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    return acf[:max_lag]


df = pd.read_csv("pv_production_june.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize(tz)
P_real = df["pv_production"].astype(float).values
time_index = df.index


site = location.Location(latitude, longitude, tz='Europe/Berlin')
solpos = site.get_solarposition(time_index)

S_t = irradiance_factor(solpos, panel_tilt, panel_azimuth)
S_t = np.array(S_t)


S_eps = np.maximum(S_t, eps)
C_hat = 1 - P_real / S_eps
C_hat = np.clip(C_hat, 0, 1)

mask = (S_t > eps)


a_hat, b_hat, alpha_hat = estimate_params_from_C(C_hat, valid=mask)

print("--- Estimated parameters from real data ---")
print(f"a_hat     = {a_hat:.4f}")
print(f"b_hat     = {b_hat:.4f}")
print(f"alpha_hat = {alpha_hat:.4f}")


N = len(S_t)

nu_sim = np.random.beta(a_hat, b_hat, size=N)

C_sim = np.zeros(N)
C_sim[0] = nu_sim[0]         
for t in range(1, N):
    C_sim[t] = alpha_hat * C_sim[t-1] + (1 - alpha_hat) * nu_sim[t]

C_sim = np.clip(C_sim, 0, 1)

P_est = (1 - C_sim) * S_t


valid_idx = np.isfinite(P_real) & np.isfinite(P_est)

if np.any(valid_idx):
    mse_all = np.mean((P_real[valid_idx] - P_est[valid_idx])**2)
    rmse_all = np.sqrt(mse_all)

    var_real_all = np.var(P_real[valid_idx], ddof=1) if np.sum(valid_idx) > 1 else np.nan
    nmse_all = mse_all / var_real_all if var_real_all and not np.isnan(var_real_all) else np.nan
else:
    mse_all = np.nan
    rmse_all = np.nan
    nmse_all = np.nan

day_mask = mask 
valid_day = valid_idx & day_mask
if np.any(valid_day):
    mse_day = np.mean((P_real[valid_day] - P_est[valid_day])**2)
    rmse_day = np.sqrt(mse_day)
    var_real_day = np.var(P_real[valid_day], ddof=1) if np.sum(valid_day) > 1 else np.nan
else:
    mse_day = np.nan
    rmse_day = np.nan

print("--- Fit quality ---")
print(f"MSE (all valid)      = {mse_all:.6g}, RMSE = {rmse_all:.6g}")
print(f"MSE (daytime only)   = {mse_day:.6g}, RMSE = {rmse_day:.6g}")

# Time plot
plt.figure(figsize=(17,5))
plt.plot(time_index, P_real, label="Empirical $P_t$", color="orange")
plt.plot(time_index, P_est, label="Estimated $P_t$", color="blue", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Normalized Power Production")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ACF 
max_lag = 200
acf_real = empirical_acf(P_real, max_lag)
acf_est  = empirical_acf(P_est, max_lag)
plt.figure(figsize=(12,4))
lags = np.arange(max_lag)

plt.stem(
lags + 0.15, acf_est[:max_lag],
linefmt='blue', markerfmt='bo', basefmt='k-',
use_line_collection=True)

# ACF 
plt.stem(
lags, acf_real[:max_lag],
linefmt='orange', markerfmt='o', basefmt='k-',
use_line_collection=True)
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend(["ACF of estimated $P_t$", "ACF of empirical $P_t$"])
plt.grid(True)
plt.tight_layout()
plt.show()



# PSD
timestamps = df.index.to_series()
dt_hours = (timestamps.diff().dt.total_seconds()/3600).median()
fs = 1/dt_hours

f_real, psd_real = welch(P_real, fs=fs, nperseg=1024)
f_est,  psd_est  = welch(P_est,  fs=fs, nperseg=1024)

plt.figure(figsize=(12,4))
plt.semilogy(f_real, psd_real, label="PSD of empirical $P_t$", color="orange")
plt.semilogy(f_est,  psd_est,  label="PSD of estimated $P_t$", color="blue")
plt.xlabel("Frequency (cycles/hour)")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# CDF
x_real = np.sort(P_real)
cdf_real = np.linspace(1/len(P_real), 1, len(P_real))

x_est = np.sort(P_est)
cdf_est = np.linspace(1/len(P_est), 1, len(P_est))

plt.figure(figsize=(6,4))
plt.plot(x_real, cdf_real, label="CDF of empirical $P_t$", color="orange")
plt.plot(x_est,  cdf_est,  label="CDF of estimated $P_t$", color="blue")
plt.xlabel("Normalized Power Production")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()