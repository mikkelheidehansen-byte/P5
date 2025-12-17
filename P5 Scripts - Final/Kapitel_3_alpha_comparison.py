import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


# Parameters 
a, b = 2.0, 5.0             
alphas = [0,0.2, 0.5, 0.9]    
N = 2000                    
lags = 40                   

mu_nu = a / (a + b)
sigma_nu2 = (a * b) / ((a + b)**2 * (a + b + 1))

plt.figure(figsize=(10, 5))
plt.xlabel("Time Lags")
plt.ylabel(r"$ACF$")

for alpha in alphas:
    nu = beta.rvs(a, b, size=N)
    nu_centered = nu - mu_nu
    C = np.zeros(N)
    for t in range(1, N):
        C[t] = alpha * C[t-1] + (1 - alpha) * nu_centered[t]

    C_centered = C - np.mean(C)
    acf_emp = np.correlate(C_centered, C_centered, mode='full')
    acf_emp = acf_emp / acf_emp[N - 1]
    k = np.arange(-lags, lags + 1)
    acf_emp_short = acf_emp[N - 1 - lags : N + lags]

    sigma_C2 = ((1 - alpha)**2 * sigma_nu2) / (1 - alpha**2)
    R_C_theory = (alpha ** np.abs(k)) * sigma_C2

    plt.plot(k, R_C_theory, label=fr"$\alpha = {alpha}$")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()



plt.figure(figsize=(10, 5))
plt.xlabel("Normalized frequency f ∈ [-0.5,0.5]")
plt.ylabel(r"$PSD$")

for alpha in alphas:
    sigma_C2 = ((1 - alpha)**2 * sigma_nu2) / (1 - alpha**2)
    f = np.linspace(-0.5, 0.5, 400)
    S_C_theory = ((1 - alpha**2) * sigma_C2) / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * f))

    plt.plot(f, S_C_theory, label=fr"$\alpha = {alpha}$")

plt.legend()
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
