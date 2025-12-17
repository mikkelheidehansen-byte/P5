import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pvlib import location

np.random.seed(14)

latitude = 48.5486
longitude = 12.6925
tz = 'Europe/Berlin'
start_date = '2025-06-01'
end_date   = '2025-06-29'

panel_tilt    = 25
panel_azimuth = 100
eps = 1e-5

# Irregular days
exclude_days = [5, 7, 12, 15, 19, 21, 22, 30]


def irradiance_factor(solpos, tilt_deg, azimuth_deg):
    zen = np.radians(solpos["zenith"])
    azi = np.radians(solpos["azimuth"])
    tilt = np.radians(tilt_deg)
    panel_az = np.radians(azimuth_deg)

    cos_th = (
        np.cos(zen)*np.cos(tilt) +
        np.sin(zen)*np.sin(tilt)*np.cos(azi - panel_az)
    )
    cos_th = np.array(cos_th)  
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
    x = np.asarray(x)
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] if acf[0] != 0 else 1.0)
    return acf[:max_lag]


time_grid = pd.date_range(
    start=start_date,
    end=end_date + " 23:59",
    freq="15min",
    tz=tz
)

P_df = pd.read_csv("pv_production_june.csv", parse_dates=["timestamp"])
P_df = P_df.rename(columns={"pv_production": "P_t"}).set_index("timestamp")


if P_df.index.tz is None:
    P_df.index = P_df.index.tz_localize(tz)
else:
    P_df.index = P_df.index.tz_convert(tz)

P_df = P_df.loc[
    (P_df.index >= pd.Timestamp(start_date, tz=tz)) &
    (P_df.index <  pd.Timestamp("2025-06-30", tz=tz))
]

P_t = (
    P_df["P_t"]
    .resample("15min")
    .mean()
    .reindex(time_grid)
    .interpolate(limit_direction="both")
)


site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(time_grid)   
S_t_full = irradiance_factor(solpos, panel_tilt, panel_azimuth)
S_t_full = np.array(S_t_full)

df_full = pd.DataFrame({"P_t": P_t.values, "S_t": S_t_full}, index=time_grid)
df_full.index.name = "timestamp"


is_excluded_day = df_full.index.day.isin(exclude_days)
df_filtered = df_full[~is_excluded_day].copy()   


df_compact = df_filtered.reset_index(drop=False)   



P_compact = df_compact["P_t"].values
S_compact = df_compact["S_t"].values
time_compact = df_compact["timestamp"] 


S_eps = np.maximum(S_compact, eps)
C_hat = 1 - P_compact / S_eps
C_hat = np.clip(C_hat, 0, 1)


mask_daytime = S_compact > eps


a_hat, b_hat, alpha_hat = estimate_params_from_C(C_hat, valid=mask_daytime)

print("--- Estimated parameters from compacted data ---")
print(f"a_hat     = {a_hat:.4f}")
print(f"b_hat     = {b_hat:.4f}")
print(f"alpha_hat = {alpha_hat:.4f}")


N = len(S_compact)

if np.isnan(a_hat) or np.isnan(b_hat) or np.isnan(alpha_hat) or a_hat <= 0 or b_hat <= 0:
    print("Estimates invalid (NaN or non-positive). Skipping simulation and plotting.")
else:
    nu_sim = np.random.beta(a_hat, b_hat, size=N)
    C_sim = np.zeros(N)
    C_sim[0] = nu_sim[0]          
    for t in range(1, N):
        C_sim[t] = alpha_hat * C_sim[t-1] + (1 - alpha_hat) * nu_sim[t]
    C_sim = np.clip(C_sim, 0, 1)
    P_est = (1 - C_sim) * S_compact



    
    plt.figure(figsize=(17,5))
    plt.plot(P_compact, label="Empirical $P_t$", color="orange")
    plt.plot(P_est, label="Estimated $P_t$", color="blue", alpha=0.7)
    plt.ylabel("Normalized Power Production")
    plt.xlabel("Time")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    max_lag = 200
    acf_real = empirical_acf(P_compact, max_lag)
    acf_est  = empirical_acf(P_est, max_lag)

    plt.figure(figsize=(12,4))
    lags = np.arange(max_lag)

    plt.stem(
        lags + 0.15, acf_est[:max_lag],
        linefmt='b-', markerfmt='bo', basefmt='k-',
        use_line_collection=True
    )

    plt.stem(
        lags, acf_real[:max_lag],
        linefmt='orange', markerfmt='o', basefmt='k-',
        use_line_collection=True
    )

    plt.xlabel("Time Lags")
    plt.ylabel("ACF")
    plt.legend(["ACF of estimated $P_t$", "ACF of empirical $P_t$"])
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    dt_hours = 0.25  
    fs = 1/dt_hours

    valid_idx = np.isfinite(P_compact) & np.isfinite(P_est)
    f_real, psd_real = welch(P_compact[valid_idx], fs=fs, nperseg=1024)
    f_est,  psd_est  = welch(P_est[valid_idx],      fs=fs, nperseg=1024)

    plt.figure(figsize=(12,4))
    plt.semilogy(f_real, psd_real, label="PSD of empirical $P_t$", color="orange")
    plt.semilogy(f_est,  psd_est,  label="PSD of estimated $P_t$", color="blue")
    plt.xlabel("Frequency (cycles/hour)")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    x_real = np.sort(P_compact)
    cdf_real = np.linspace(1/len(x_real), 1, len(x_real))

    x_est = np.sort(P_est)
    cdf_est = np.linspace(1/len(x_est), 1, len(x_est))

    plt.figure(figsize=(6,4))
    plt.plot(x_real, cdf_real, label="CDF of empirical $P_t$", color="orange")
    plt.plot(x_est,  cdf_est,  label="CDF of estimated $P_t$", color="blue")
    plt.xlabel("Normalized Power Production")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

