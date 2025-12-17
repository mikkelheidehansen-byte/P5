import numpy as np
import pandas as pd
from pvlib import location
import matplotlib.pyplot as plt

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

# DataFrame
data = pd.DataFrame({
    'Timestamp': times,
    'BaselinePower_kWh': Bt,
})

# Plots
plt.figure(figsize=(14, 5))
plt.plot(data['Timestamp'], data['BaselinePower_kWh'], label='$S_t$', color='blue')
plt.ylabel('Normalized Power Production')
plt.xlabel('Time')
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



