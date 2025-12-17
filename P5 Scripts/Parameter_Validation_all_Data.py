import numpy as np
import pandas as pd
from pvlib import location


latitude = 48.5486
longitude = 12.6925
tz        = 'Europe/Berlin'

panel_tilt    = 25
panel_azimuth = 100
eps = 1e-5


n_realizations = 1   
seed = 12              

np.random.seed(14)


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
    acf = acf / (acf[0] if acf[0] != 0 else 1.0)
    return acf[:max_lag]



df = pd.read_csv("pv_production_june.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)


df.index = pd.to_datetime(df.index)
if df.index.tz is None:
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

if seed is None:
    rng = np.random.default_rng()
else:
    rng = np.random.default_rng(seed)

P_est_all = np.full((n_realizations, N), np.nan)
mse_all_list = []
rmse_all_list = []
mse_day_list = []
rmse_day_list = []

if np.isnan(a_hat) or np.isnan(b_hat) or np.isnan(alpha_hat) or a_hat <= 0 or b_hat <= 0:
    print("Estimates invalid (NaN or non-positive). Skipper multi-realiseringer.")
else:
    for i in range(n_realizations):
        nu_sim = rng.beta(a_hat, b_hat, size=N)

        C_sim = np.zeros(N)
        C_sim[0] = nu_sim[0]
        for t in range(1, N):
            C_sim[t] = alpha_hat * C_sim[t-1] + (1 - alpha_hat) * nu_sim[t]
        C_sim = np.clip(C_sim, 0, 1)

        P_est = (1 - C_sim) * S_t
        P_est_all[i, :] = P_est

        valid_idx = np.isfinite(P_real) & np.isfinite(P_est)
        if not np.any(valid_idx):
            mse_all_list.append(np.nan)
            rmse_all_list.append(np.nan)
            mse_day_list.append(np.nan)
            rmse_day_list.append(np.nan)
            continue

        mse_all = np.mean((P_real[valid_idx] - P_est[valid_idx])**2)
        rmse_all = np.sqrt(mse_all)

        valid_day = valid_idx & mask
        if np.any(valid_day):
            mse_day = np.mean((P_real[valid_day] - P_est[valid_day])**2)
            rmse_day = np.sqrt(mse_day)
        else:
            mse_day = np.nan
            rmse_day = np.nan

        mse_all_list.append(mse_all)
        rmse_all_list.append(rmse_all)
        mse_day_list.append(mse_day)
        rmse_day_list.append(rmse_day)


def summarize(arr):
    arr = np.array(arr, dtype=float)
    arr_valid = arr[np.isfinite(arr)]
    if arr_valid.size == 0:
        return np.nan, np.nan
    return arr_valid.mean(), arr_valid.std(ddof=1)

mse_all_mean, mse_all_std = summarize(mse_all_list)
rmse_all_mean, rmse_all_std = summarize(rmse_all_list)
mse_day_mean, mse_day_std = summarize(mse_day_list)
rmse_day_mean, rmse_day_std = summarize(rmse_day_list)

print("\n--- Fit quality over realiseringer ---")
print(f"n_realizations = {n_realizations}, seed = {seed}")
print(f"MSE (all) : mean = {mse_all_mean:.6g}, std = {mse_all_std:.6g}")
print(f"RMSE(all) : mean = {rmse_all_mean:.6g}, std = {rmse_all_std:.6g}")
print(f"MSE (day) : mean = {mse_day_mean:.6g}, std = {mse_day_std:.6g}")
print(f"RMSE(day) : mean = {rmse_day_mean:.6g}, std = {rmse_day_std:.6g}")

