import numpy as np
import pandas as pd
from pvlib import location
import matplotlib.pyplot as plt

# Location and timezone
latitude = 48.5486  
longitude = 12.6925
tz = 'Europe/Berlin'
date = '2025-06-01'  

# 15 min interval
times = pd.date_range(start=f'{date} 00:00', end=f'{date} 23:59', freq='15min', tz=tz)

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
data['Timestamp_local'] = data['Timestamp'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
plt.plot(data['Timestamp_local'], data['BaselinePower_kWh'], color='blue')
plt.ylabel('Normalized Power Production')
plt.xlabel('Time')
plt.grid(True)
plt.legend(handles=plt.gca().lines, labels=['$S_t$'])


import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()

