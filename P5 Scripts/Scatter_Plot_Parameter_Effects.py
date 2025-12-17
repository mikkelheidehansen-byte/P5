import numpy as np
import pandas as pd
from pvlib import location
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Location and timezone ---
latitude = 48.5486  
longitude = 12.6925
tz = 'Europe/Berlin'

start_date = '2025-06-01'
end_date = '2025-06-30'
times = pd.date_range(start=start_date, end=end_date + ' 23:59', freq='15min', tz=tz)
times = times.tz_localize(None)

# Sun position 
site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(times)

# Solar panel parameters
panel_tilt = 25
panel_azimuth = 100

# Solar radiation
def irradiance_factor(solpos, tilt, azimuth):
    zenith_rad = np.radians(solpos['zenith'])
    azimuth_rad = np.radians(solpos['azimuth'])
    tilt_rad = np.radians(tilt)
    panel_azimuth_rad = np.radians(azimuth)
    cos_theta = (
        np.cos(zenith_rad) * np.cos(tilt_rad) +
        np.sin(zenith_rad) * np.sin(tilt_rad) * np.cos(azimuth_rad - panel_azimuth_rad)
    )
    cos_theta[cos_theta < 0] = 0
    return cos_theta

# Power production
irr_factor = irradiance_factor(solpos, panel_tilt, panel_azimuth)
Bt = irr_factor


# Simulation function 
def simulate(alpha, a_beta, b_beta):
    np.random.seed(42)
    nu_t = beta.rvs(a_beta, b_beta, size=len(times))
    C_t = np.zeros(len(times))
    C_t[0] = nu_t[0]
    for t in range(1, len(times)):
        C_t[t] = alpha * C_t[t-1] + (1 - alpha) * nu_t[t]

    
    Pt = (1 - C_t) * Bt 
    corr = np.corrcoef(Bt, Pt)[0, 1]
    return Pt, corr


def plot_group(title, param_name, param_values, fixed_params): 
    plt.figure(figsize=(6, 6))
    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        Pt, corr = simulate(**params)
        plt.scatter(Bt, Pt, s=8, alpha=0.5, label=f"{param_name}={val}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlabel("Sunlight Model $S_t$")
    plt.ylabel("Normalized Power Production $P_t$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# alpha
fixed_1 = {"a_beta": 2, "b_beta": 5}
alpha_values = [0.1, 0.5, 1]
plot_group("Effect of Parameter α", "alpha", alpha_values, fixed_1)

# a
fixed_4 = {"b_beta": 2, "alpha": 0.7}
beta_a_values = [20, 5, 1]
plot_group("Effect of Parameter a", "a_beta", beta_a_values, fixed_4)


# b
fixed_3 = {"a_beta": 2, "alpha": 0.7}
beta_b_values = [20, 5, 1]
plot_group("Effect of Parameter b", "b_beta", beta_b_values, fixed_3)


