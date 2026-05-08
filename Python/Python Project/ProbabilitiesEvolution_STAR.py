import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig, inv
from scipy.special import comb
import matplotlib.pyplot as plt

# --- Parameters ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

def f(x, beta):
    return 1.0 / (1.0 + np.exp(np.clip(beta * x, -700, 700)))

def transition_matrix_star(N, a, b, beta):
    dim = 2 * N
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    for k in range(N):
        E[k] = -a
        E[k + N] = a + 2 * b * (2 * k - N + 1)
        
    for k in range(N):
        idx_dn = k
        idx_up = k + N
        
        dU_central = E[idx_up] - E[idx_dn]
        M[idx_up, idx_dn] = f(dU_central, beta)
        M[idx_dn, idx_up] = f(-dU_central, beta)
        
        if k < N - 1:
            M[idx_dn + 1, idx_dn] = (N - 1 - k) * f(0, beta) 
        if k > 0:
            M[idx_dn - 1, idx_dn] = k * f(0, beta)
            
        if k < N - 1:
            dU_up = E[idx_up + 1] - E[idx_up]
            M[idx_up + 1, idx_up] = (N - 1 - k) * f(dU_up, beta)
        if k > 0:
            dU_dn = E[idx_up - 1] - E[idx_up]
            M[idx_up - 1, idx_up] = k * f(dU_dn, beta)

    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M

# --- Configuración del Sistema ---
N = 40
a = a_values[N-2]
b = b_values[N-2]
beta = 1.0

# Condición Inicial
p0 = np.zeros(2 * N)
for k in range(N):
    deg = comb(N - 1, k)
    p0[k] = deg       
    p0[k + N] = deg   
p0 /= np.sum(p0)

M = transition_matrix_star(N, a, b, beta)

# Diagonalización espectral
eigvals, eigvecs = eig(M)
eigvals = np.real(eigvals)  
eigvecs = np.real(eigvecs)

sorted_indices = np.argsort(eigvals)[::-1]
sorted_eigvals = eigvals[sorted_indices]

tau_1 = -1.0 / sorted_eigvals[1]
tau_2 = -1.0 / sorted_eigvals[2]

# --- NUEVO BLOQUE: Evolución temporal y Plot ---
print("Calculando evolución temporal para el plot...")

# Creamos un array de tiempo en escala logarítmica: desde t=0.1 hasta 10 veces tau_1
t_array = np.logspace(-1, np.log10(tau_1 * 10), 300)
p_t = np.zeros((2 * N, len(t_array)))

# Evaluamos p(t) para cada instante
for i, t_val in enumerate(t_array):
    p_t[:, i] = expm(M * t_val) @ p0

plt.figure(figsize=(12, 7))

# Pintamos las probabilidades
for k in range(N):
    # Subespacio DOWN (Línea sólida azul)
    plt.plot(t_array, p_t[k, :], color='blue', linestyle='-', alpha=0.7, lw=2,
             label='Subespacio DOWN' if k == 0 else "")
    # Subespacio UP (Línea punteada roja)
    plt.plot(t_array, p_t[k + N, :], color='red', linestyle='--', alpha=0.7, lw=2,
             label='Subespacio UP' if k == 0 else "")

# Marcamos las escalas temporales clave
plt.axvline(tau_2, color='gray', linestyle=':', lw=2, label=f'$\\tau_2$ (Fin del transitorio)')
plt.axvline(tau_1, color='black', linestyle=':', lw=2, label=f'$\\tau_1$ (Termalización final)')

plt.xscale('log')
plt.ylim(0, np.max(p_t) * 1.1)
plt.xlabel('Tiempo $t$ (Escala Logarítmica)', fontsize=14)
plt.ylabel('Probabilidad $p_k(t)$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4, which='both')
plt.tight_layout()
plt.show()

