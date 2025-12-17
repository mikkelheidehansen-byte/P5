import numpy as np
import pandas as pd
from pvlib import location
import matplotlib.pyplot as plt
from scipy.stats import beta

# Location and timezone
latitude = 48.5486  
longitude = 12.6925
tz = 'Europe/Berlin'

start_date = '2025-06-01'
end_date = '2025-06-30'
times = pd.date_range(start=start_date, end=end_date + ' 23:59', freq='15min', tz=tz)

# Sun position
site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(times)

# Parameters for solar panels
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

# C_t 
alpha = 0.7  
a_beta, b_beta = 2, 5  
np.random.seed(41)

nu_t = beta.rvs(a_beta, b_beta, size=len(times))
C_t = np.zeros(len(times))
C_t[0] = nu_t[0]
for t in range(1, len(times)):
    C_t[t] = alpha * C_t[t-1] + (1 - alpha) * nu_t[t]
                            

# Total production with noise
Pt = (1 - C_t) * Bt 


# DataFrame
df = pd.DataFrame({
    'Timestamp': times,
    'BaselinePower_kWh': Bt,
    'TotalPower_kWh': Pt,
    'CloudFactor_Ct': C_t
})



# Plots
plt.figure(figsize=(7, 6))
plt.scatter(df["BaselinePower_kWh"], df["TotalPower_kWh"],
            alpha=0.6, s=10, color="purple")
plt.plot([0, 1], [0, 1], color="blue", linestyle="-", linewidth=1)
plt.xlabel("Sunlight Model $S_t$")
plt.ylabel("Normalized Power Production $P_t$")
plt.grid(True)

plt.tight_layout()
plt.show()
