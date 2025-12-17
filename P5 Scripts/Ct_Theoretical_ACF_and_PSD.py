
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.signal import welch

# Parameters 
a, b = 2.0, 5.0       
alpha = 0.8           
N = 2000              
lags = 40             
random.seed(42)


mu_nu = a / (a + b)
sigma_nu2 = (a * b) / ((a + b)**2 * (a + b + 1))


nu = beta.rvs(a, b, size=N)
nu_centered = nu# - mu_nu        # træk middel ud
C = np.zeros(N)
for t in range(1, N):
    C[t] = alpha * C[t-1] + (1 - alpha) * nu_centered[t]


C_centered = C - np.mean(C)
acf_emp = np.correlate(C_centered, C_centered, mode='full')
acf_emp = acf_emp / acf_emp[N - 1]  
lags_arr = np.arange(-N + 1, N)


k = np.arange(-lags, lags + 1)
sigma_C2 = ((1 - alpha)**2 * sigma_nu2) / (1 - alpha**2)
R_C_theory = sigma_C2 * (alpha ** np.abs(k))
rho_theory = alpha ** np.abs(k)

acf_emp_short = acf_emp[N - 1 - lags : N + lags]

f = np.linspace(-0.5, 0.5, 400)  # normaliseret frekvens [-1,1]
S_C_theory = ((1 - alpha**2) * sigma_C2) / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * f))

f_emp, Pxx_emp = welch(C_centered, nperseg=256, return_onesided=False)
f_emp = np.fft.fftshift(f_emp)
Pxx_emp = np.fft.fftshift(Pxx_emp)

# Plots
plt.figure(figsize=(12, 12))

import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(8,4))
plt.stem(k, R_C_theory, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='Theoretical')
plt.xlabel('Time Lags')
plt.ylabel('ACF')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,4))
plt.stem(k, rho_theory, basefmt=" ", linefmt='C0-', markerfmt='C0o', label='Theoretical')
plt.plot(k, acf_emp_short, 'r--', label='Empirical (simulated)')
plt.xlabel('Time Lags')
plt.ylabel('ρ(k)')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,4))
plt.plot(f, S_C_theory, 'k', label='Theoretical')
plt.plot(f_emp, Pxx_emp, 'r--', label='Empirical')  
plt.xlabel('Normalized frequency f ∈ [-0.5,0.5]')
plt.ylabel('PSD')
plt.legend()
plt.tight_layout()

plt.show()
