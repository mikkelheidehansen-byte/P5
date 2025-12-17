
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib import location


latitude = 48.5486
longitude = 12.6925
tz        = 'Europe/Berlin'

start_date = '2025-06-01'
end_date   = '2025-06-29'

panel_tilt    = 25
panel_azimuth = 100


alpha_true = 0.7
a_true     = 2
b_true     = 5

n_real = 1000000      
eps     = 1e-5     


def irradiance_factor(solpos, tilt_deg, azimuth_deg):
    zen = np.radians(solpos['zenith'])
    azi = np.radians(solpos['azimuth'])
    tilt = np.radians(tilt_deg)
    panel_az = np.radians(azimuth_deg)

    cos_th = (
        np.cos(zen)*np.cos(tilt) +
        np.sin(zen)*np.sin(tilt)*np.cos(azi-panel_az)
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
    N_valid = len(C_valid)
    if N_valid < 10:
        return np.nan, np.nan, np.nan

    meanC = np.mean(C_valid)
    varC = np.var(C_valid, ddof=1)
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
    gamma1 = np.sum((x - meanC) * (y - meanC)) / (n_pairs - 1)

    alpha_hat = gamma1 / varC
    alpha_hat = float(np.clip(alpha_hat, 0.0, 0.9999))

    nu_hat_full = (x - alpha_hat * y) / (1.0 - alpha_hat)

    nu_hat_full = np.clip(nu_hat_full, 0.001, 0.999)

    m = np.mean(nu_hat_full)
    v = np.var(nu_hat_full, ddof=1)
    if v <= 0:
        return np.nan, np.nan, alpha_hat

    kappa = m * (1 - m) / v - 1.0
    if kappa <= 0:
        return np.nan, np.nan, alpha_hat

    a_hat = m * kappa
    b_hat = (1 - m) * kappa

    return a_hat, b_hat, alpha_hat



time_grid = pd.date_range(
    start=start_date,
    end=end_date + " 23:45",
    freq="15min",
    tz=tz
)


site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(time_grid)

S_t = irradiance_factor(solpos, panel_tilt, panel_azimuth)
S_t = np.array(S_t)
N = len(S_t)


a_estimates = np.zeros(n_real)
b_estimates = np.zeros(n_real)
alpha_estimates = np.zeros(n_real)


rng = np.random.default_rng(12345)

for r in range(n_real):

    nu_t = rng.beta(a_true, b_true, size=N)

    C_t = np.zeros(N)
    C_t[0] = nu_t[0]
    for t in range(1, N):
        C_t[t] = alpha_true*C_t[t-1] + (1-alpha_true)*nu_t[t]
    C_t = np.clip(C_t, 0, 1)

    P_t = (1 - C_t)*S_t
    
    S_eps = np.maximum(S_t, eps)
    C_hat = 1 - P_t/(S_eps)
    C_hat = np.clip(C_hat, 0, 1)
    
    mask = (S_t > eps)   
    C_red = C_hat.copy() 

    a_hat, b_hat, alpha_hat = estimate_params_from_C(C_red, valid=mask)


    a_estimates[r] = a_hat
    b_estimates[r] = b_hat
    alpha_estimates[r] = alpha_hat

    if (r+1) % 100 == 0:
        print(f"{r+1}/{n_real} realizations done.")


print("\n------------------------------------------------------")
print("PARAMETER ESTIMATION RESULTS (1000 realisations)")
print("------------------------------------------------------")
print(f"True a = {a_true}, Mean of estimated a = {np.mean(a_estimates):.4f}")
print(f"True b = {b_true}, Mean of estimated b = {np.nanmean(b_estimates):.4f}")
print(f"True α = {alpha_true}, Mean of estimated α = {np.nanmean(alpha_estimates):.4f}")

print("\nStandard deviations:")
print(f"a std = {np.nanstd(a_estimates):.4f}")
print(f"b std = {np.nanstd(b_estimates):.4f}")
print(f"α std = {np.nanstd(alpha_estimates):.4f}")

print("\nRange of estimates")
print(f"a: {np.nanpercentile(a_estimates,0):.4f} – {np.nanpercentile(a_estimates,100):.4f}")
print(f"b: {np.nanpercentile(b_estimates,0):.4f} – {np.nanpercentile(b_estimates,100):.4f}")
print(f"α: {np.nanpercentile(alpha_estimates,0):.4f} – {np.nanpercentile(alpha_estimates,100):.4f}")

def plot_hist_with_info(data, true_value, title, param_name):
    mean_val = np.nanmean(data)
    std_val  = np.nanstd(data)
    min_val  = np.nanmin(data)
    max_val  = np.nanmax(data)

    perc_std = (std_val / true_value) * 100 if true_value != 0 else np.nan

    plt.figure(figsize=(12,4))
    plt.hist(data, bins=40, alpha=0.7)
    plt.axvline(true_value, color='red', linestyle='--',)
    plt.axvline(mean_val, color='green', linestyle='--',)

    
    textstr = '\n'.join((
        f"True value = {true_value:.4f}",
        f"Mean = {mean_val:.4f}",
        f"Std = {std_val:.4f}  ({perc_std:.1f}% of true)",
        f"Min = {min_val:.4f}",
        f"Max = {max_val:.4f}",
        f"Esp = {eps:.5f}"
    ))

    plt.text(0.97, 0.97, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_hist_with_info(a_estimates, a_true, "Distribution of estimated a", "a")
plot_hist_with_info(b_estimates, b_true, "Distribution of estimated b", "b")
plot_hist_with_info(alpha_estimates, alpha_true, "Distribution of estimated alpha", "alpha")
