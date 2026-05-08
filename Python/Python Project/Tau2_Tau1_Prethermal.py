import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# =============================================================================
# --- 1. PARÁMETROS OPTIMIZADOS DEL STAR MODEL ---
# =============================================================================
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

def f_glauber(delta_E, beta):
    return 1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -700, 700)))

def build_star_matrix(N, a, b, beta=1.0, gamma=1.0):
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
        M[idx_up, idx_dn] = gamma * f_glauber(dU_central, beta)
        M[idx_dn, idx_up] = gamma * f_glauber(-dU_central, beta)
        
        if k < N - 1:
            M[idx_dn + 1, idx_dn] = gamma * (N - 1 - k) * f_glauber(0, beta)
            M[idx_up + 1, idx_up] = gamma * (N - 1 - k) * f_glauber(E[idx_up + 1] - E[idx_up], beta)
        if k > 0:
            M[idx_dn - 1, idx_dn] = gamma * k * f_glauber(0, beta)
            M[idx_up - 1, idx_up] = gamma * k * f_glauber(E[idx_up - 1] - E[idx_up], beta)

    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M

# =============================================================================
# --- 2. EXTRACCIÓN DE ESCALAS TEMPORALES ---
# =============================================================================
N_list = list(range(3, 35))
beta = 1.0

tau1_list = []
tau2_list = []
ratio_list = []

print(f"{'N':<5} | {'tau_2 (Rápido)':<18} | {'tau_1 (Lento)':<18} | {'Ratio tau_1/tau_2':<18}")
print("-" * 65)

for N in N_list:
    a = a_values[N - 2]
    b = b_values[N - 2]
    
    M = build_star_matrix(N, a, b, beta=beta)
    evals = np.sort(np.real(la.eigvals(M)))[::-1]
    
    t1 = -1.0 / evals[1]
    t2 = -1.0 / evals[2]
    
    tau1_list.append(t1)
    tau2_list.append(t2)
    ratio_list.append(t1 / t2)
    
    print(f"{N:<5} | {t2:<18.6f} | {t1:<18.6f} | {t1/t2:<18.2e}")

# =============================================================================
# --- 3. GRÁFICO DE DOBLE EJE ---
# =============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Eje Izquierdo: tau_2 ---
color1 = 'tab:red'
ax1.set_xlabel('Número de espines $N$', fontsize=14)
ax1.set_ylabel(r'Tiempo de formación pretermal $\tau_2$', color=color1, fontsize=14)
line1 = ax1.plot(N_list, tau2_list, 'o-', color=color1, lw=2.5, markersize=8, label=r'$\tau_2$ (Acceso)')
ax1.tick_params(axis='y', labelcolor=color1)

# Creamos un segundo eje Y que comparte el mismo eje X
ax2 = ax1.twinx()  

# --- Eje Derecho: Ratio tau_1 / tau_2 ---
color2 = 'tab:blue'
ax2.set_ylabel(r'Ratio of accessible time $\tau_1 / \tau_2$ (Log scale)', color=color2, fontsize=14)
line2 = ax2.plot(N_list, ratio_list, 's-', color=color2, lw=2.5, markersize=8, label=r'Ratio $\tau_1 / \tau_2$')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log') # Escala logarítmica para ver la divergencia exponencial

# Juntamos las leyendas de ambos ejes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12)

ax1.grid(alpha=0.4)
fig.tight_layout()
plt.show()