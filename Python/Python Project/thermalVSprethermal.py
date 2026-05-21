import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS DEL STAR MODEL ---
# =============================================================================
N_list = np.arange(3, 16) # Desde N=3 hasta N=15

a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977,
            18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872,
            2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537]

T_val = 1.0
dT = 0.001
gamma = 1.0

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -700, 700)))

def build_star_matrix(N, a, b, T):
    dim = 2 * N
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    for k in range(N):
        E[k] = -a
        E[k + N] = a + 2*b*(2*k - N + 1)
        
    for k in range(N):
        idx_dn, idx_up = k, k + N
        if k < N - 1:
            M[idx_dn + 1, idx_dn] = 0.5 * gamma * (N - 1 - k)
            M[idx_up + 1, idx_up] = gamma * (N - 1 - k) * f_glauber(4*b, T)
        if k > 0:
            M[idx_dn - 1, idx_dn] = 0.5 * gamma * k
            M[idx_up - 1, idx_up] = gamma * k * f_glauber(-4*b, T)
        dU_central = E[idx_up] - E[idx_dn]
        M[idx_up, idx_dn] = gamma * f_glauber(dU_central, T)
        M[idx_dn, idx_up] = gamma * f_glauber(-dU_central, T)
        
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

# =============================================================================
# --- 2. CÁLCULO DE FIGURAS DE MÉRITO ---
# =============================================================================
eta_eq_list = []
eta_pre_list = []

print("Calculando eficiencias térmicas...")

for N in N_list:
    a, b = a_values[N-2], b_values[N-2]
    
    # Construcción de matrices
    M_base, E_base = build_star_matrix(N, a, b, T_val)
    M_up, _ = build_star_matrix(N, a, b, T_val + dT)
    M_dn, _ = build_star_matrix(N, a, b, T_val - dT)
    
    # 1. Tiempos de relajación
    evals = np.sort(np.real(la.eigvals(M_base)))[::-1]
    tau_1 = -1.0 / evals[1]
    tau_2 = -1.0 / evals[2]
    
    # 2. Información de Fisher en Equilibrio (Asintótica)
    # P_eq es la distribución de Boltzmann
    P_eq_base = np.exp(-E_base / T_val) / np.sum(np.exp(-E_base / T_val))
    P_eq_up = np.exp(-E_base / (T_val + dT)) / np.sum(np.exp(-E_base / (T_val + dT)))
    P_eq_dn = np.exp(-E_base / (T_val - dT)) / np.sum(np.exp(-E_base / (T_val - dT)))
    
    dP_eq_dT = (P_eq_up - P_eq_dn) / (2 * dT)
    F_eq = np.sum((dP_eq_dT**2) / (P_eq_base + 1e-20))
    
    # 3. Información de Fisher Pretermal
    # Condición Inicial (GS)
    P0 = np.zeros(2*N)
    P0[0] = 1.0  # Empezamos en el estado fundamental (Termómetro frío)
    
    # Evaluamos en la meseta pretermal (t = 5 * tau_2)
    t_pre = 5* tau_2
    
    P_pre_base = la.expm(M_base * t_pre) @ P0
    P_pre_up = la.expm(M_up * t_pre) @ P0
    P_pre_dn = la.expm(M_dn * t_pre) @ P0
    
    dP_pre_dT = (P_pre_up - P_pre_dn) / (2 * dT)
    F_pre = np.sum((dP_pre_dT**2) / (np.abs(P_pre_base) + 1e-20))
    
    # Guardamos las eficiencias (\eta = F / t)
    eta_eq_list.append(F_eq / tau_1)
    eta_pre_list.append(F_pre / tau_2)
    
    print(f"N={N:2d} | eta_pre = {F_pre/tau_2:.2e} | eta_eq = {F_eq/tau_1:.2e}")

# =============================================================================
# --- 3. GRÁFICA PARA EL SUPERVISOR ---
# =============================================================================
plt.figure(figsize=(9, 6))

plt.plot(N_list, eta_pre_list, marker='o', color='purple', lw=3, markersize=8, 
         label=r'Prethermal ($\mathcal{F}_{pre} / \tau_2$)')
plt.plot(N_list, eta_eq_list, marker='s', color='black', lw=3, markersize=8, linestyle='--',
         label=r'Equilibrium ($\mathcal{F}_{eq} / \tau_1$)')

plt.yscale('log')
plt.xlabel('$N$)', fontsize=14)
plt.ylabel('$\eta = \mathcal{F}/t$', fontsize=14)


plt.legend(fontsize=12, loc='upper right')
plt.grid(alpha=0.4, which='both')
plt.tight_layout()
plt.show()