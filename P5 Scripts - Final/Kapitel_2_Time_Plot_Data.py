import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pv_production_june.csv", parse_dates=["timestamp"])

df.set_index("timestamp", inplace=True)

plt.figure(figsize=(15, 5))
plt.plot(df.index, df["pv_production"], label="Power Production", color="orange")
plt.xlabel("Time")
plt.ylabel("Normalized Power Produktion")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
