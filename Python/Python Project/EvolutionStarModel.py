import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS Y LISTAS ---
# =============================================================================
N = 8  # ¡Ahora puedes subir N sin miedo!
idx = N - 2 

a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977,
            18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664,
            70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899,
            157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762,
            278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872,
            2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537,
            3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246,
            5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967,
            7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

a_star = a_values[idx]
b_star = b_values[idx]

T_val = 1.0
dT = 0.0008
gamma = 1.0

def f_glauber(dU, T):
    exponent = np.clip(dU / T, -700, 700) # Límite ampliado para N grandes
    return 1.0 / (1.0 + np.exp(exponent))

# =============================================================================
# --- 2. CONSTRUCCIÓN DE MATRIZ (Solo Star Model) ---
# =============================================================================
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
# --- 3. DIAGONALIZACIÓN Y ECUACIONES EXPLÍCITAS ---
# =============================================================================
def get_explicit_equations(M, E):
    D_states = len(E)
    N_spins = D_states // 2
    
    # Condición inicial T=infinito
    P0 = np.zeros(D_states)
    for k in range(N_spins):
        degeneracy = comb(N_spins - 1, k)
        P0[k] = degeneracy           
        P0[k + N_spins] = degeneracy 
    P0 /= np.sum(P0)

    evals, evecs = la.eig(M)
    evals = np.real(evals)
    evecs = np.real(evecs)
    
    C = la.solve(evecs, P0)
    amplitudes = evecs * C[np.newaxis, :]
    
    return evals, amplitudes

def evaluate_probability(t, evals, amplitudes):
    exp_L = np.exp(evals * t)
    return np.dot(amplitudes, exp_L)

# =============================================================================
# --- 4. SIMULACIÓN FISHER ---
# =============================================================================
# 1. Construir matrices
M_base, E_base = build_star_matrix(N, a_star, b_star, T_val)
M_up, E_up     = build_star_matrix(N, a_star, b_star, T_val + dT)
M_dn, E_dn     = build_star_matrix(N, a_star, b_star, T_val - dT)

# 2. Obtener ecuaciones (Diagonalización)
L_base, A_base = get_explicit_equations(M_base, E_base)
L_up, A_up     = get_explicit_equations(M_up, E_up)
L_dn, A_dn     = get_explicit_equations(M_dn, E_dn)

# --- CÁLCULO DINÁMICO DEL TIEMPO ---
# Identificamos el modo de relajación más lento (tau_1) para fijar el t_max
sorted_evals = np.sort(L_base)[::-1]
tau_1 = -1.0 / sorted_evals[1]
print(f"Para N={N}, el tiempo asintótico estimado es tau_1 = {tau_1:.2e}")

# Malla logarítmica de solo 1000 puntos: ultrarrápida y perfecta para escalas grandes
t_array = np.logspace(-1, np.log10(tau_1 * 10), 1000)
F_t = np.zeros(len(t_array))

print("Calculando Información de Fisher dinámica...")
for i, t in enumerate(t_array):
    P_base = evaluate_probability(t, L_base, A_base)
    P_up   = evaluate_probability(t, L_up, A_up)
    P_dn   = evaluate_probability(t, L_dn, A_dn)
    
    dP_dT = (P_up - P_dn) / (2 * dT)
    F_t[i] = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))

# =============================================================================
# --- 5. PLOT ---
# =============================================================================
eta_star = F_t / t_array

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Primer plot: F(t)
ax1.plot(t_array, F_t, label=f'Star Model (N={N})', color='red', lw=2.5)
ax1.set_ylabel(r'$\mathcal{F}(t)$', fontsize=14)
ax1.set_title('Evolución de la Información de Fisher y Precisión', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.4, which='both')

# Segundo plot: eta(t)
ax2.plot(t_array, eta_star, color='red', lw=2.5)
ax2.set_xlabel('Tiempo $t$ (Escala logarítmica)', fontsize=14)
ax2.set_ylabel(r'$\eta = \frac{\mathcal{F}(t)}{t}$', fontsize=14)
ax2.grid(alpha=0.4, which='both')

# Eje X logarítmico compartido
ax2.set_xscale('log')

plt.tight_layout()
plt.show()