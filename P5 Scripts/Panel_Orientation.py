import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib import location


latitude = 48.5486
longitude = 12.6925
tz = "Europe/Berlin"

start_date = "2025-06-01"
end_date   = "2025-06-30"

panel_tilt     = 25    
panel_azimuth  = 100    


times = pd.date_range(
    start=start_date,
    end=end_date + " 23:59",
    freq="15min",
    tz=tz
)


site = location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(times)

def irradiance_factor(solpos, tilt_deg, azimuth_deg):
    zenith  = np.radians(solpos["zenith"].values)
    azimuth = np.radians(solpos["azimuth"].values)

    tilt    = np.radians(tilt_deg)
    panel_az = np.radians(azimuth_deg)

    cos_theta = (
        np.cos(zenith) * np.cos(tilt)
        + np.sin(zenith) * np.sin(tilt) * np.cos(azimuth - panel_az)
    )

    cos_theta = np.maximum(cos_theta, 0.0)

    return pd.Series(cos_theta, index=times, name="S_t")

S_t = irradiance_factor(solpos, panel_tilt, panel_azimuth)


P_df = pd.read_csv(
    "pv_production_june.csv",
    parse_dates=["timestamp"]
)

P_df = (
    P_df
    .rename(columns={"pv_production": "P_t"})
    .set_index("timestamp")
    .sort_index()
)

if P_df.index.tz is None:
    P_df.index = P_df.index.tz_localize(tz)
else:
    P_df.index = P_df.index.tz_convert(tz)


df = pd.concat([P_df["P_t"], S_t], axis=1).dropna()


plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df["S_t"],
    label="Irradiance factor $S_t$",
    linewidth=2
)

plt.plot(
    df.index,
    df["P_t"],
    label="Normalized PV production $P_t$",
    linewidth=2
)

plt.xlabel("Time")
plt.ylabel("Normalized Power production")
plt.title("Observed PV production vs. irradiance factor (nighttime included)")
plt.legend()
plt.tight_layout()
plt.text(
    0.01, 0.95,
    f"Panel azimuth = {panel_azimuth}°",
    transform=plt.gca().transAxes,
    fontsize=13,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.8,
        boxstyle="round,pad=0.3"
    )
)

plt.show()
