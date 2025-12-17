import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters 
alpha = 0.8  
N = 2000     
lags = 40    


beta_params = [(2, 5),(5,2), (15, 15), (2, 2)]  
linestyles = ['-', '-.', ':', '--']  


plt.figure(figsize=(12, 10))


plt.figure(figsize=(10, 5))
plt.xlabel("Time Lags")
plt.ylabel(r"$ACF$")


for i, (a, b) in enumerate(beta_params):

    
    mu_nu = a / (a + b)
    sigma_nu2 = (a * b) / ((a + b)**2 * (a + b + 1))

    nu = beta.rvs(a, b, size=N)
    nu_centered = nu - mu_nu
    C = np.zeros(N)
    for t in range(1, N):
        C[t] = alpha * C[t-1] + (1 - alpha) * nu_centered[t]

    C_centered = C - np.mean(C)
    acf_emp = np.correlate(C_centered, C_centered, mode='full')
    acf_emp = acf_emp / acf_emp[N - 1]
    k = np.arange(-lags, lags + 1)

    sigma_C2 = ((1 - alpha)**2 * sigma_nu2) / (1 - alpha**2)
    R_C_theory = sigma_C2 * (alpha ** np.abs(k))

    plt.plot(k, R_C_theory, linestyles[i % len(linestyles)],
             label=fr"$a={a},\ b={b}$")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()



plt.figure(figsize=(10, 5))
plt.xlabel("Normalized Frequency f ∈ [-0.5,0.5]")
plt.ylabel(r"$PSD$")

for i, (a, b) in enumerate(beta_params):

    mu_nu = a / (a + b)
    sigma_nu2 = (a * b) / ((a + b)**2 * (a + b + 1))

    sigma_C2 = ((1 - alpha)**2 * sigma_nu2) / (1 - alpha**2)

    f = np.linspace(-0.5, 0.5, 400)
    S_C_theory = ((1 - alpha**2) * sigma_C2) / \
                 (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * f))

    plt.plot(f, S_C_theory, linestyles[i % len(linestyles)],
             label=fr"$a={a},\ b={b}$")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()