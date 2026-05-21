import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS FÍSICOS ---
# =============================================================================
N = 4
T_val = 1.0
dT = 0.001
gamma = 1.0

# Parámetros óptimos encontrados para dinámicas locales (N=3)
B_ata = 0.8869
J_ata = 0.0535

B_free = 0.9659
J_free = 0.0

# Parámetros topológicos para el límite Global de 2 niveles (El límite del Paper)
g0_global = 2               # Degeneración del Ground State
g1_global = (2**N) - g0_global     # Degeneración masiva del estado excitado
E_gap_global = 2.30       # Gap de energía óptimo teórico

# Malla de tiempo centrada en el transitorio corto
t_array = np.linspace(0.01, 25.0, 500)

# =============================================================================
# --- 2. CONSTRUCCIÓN DE SISTEMAS LOCALES (N+1 Macroestados) ---
# =============================================================================
def get_E_macro(n, B_val, J_val):
    S = 2 * n - N
    E_zeeman = -B_val * S
    E_int = -0.5 * J_val * (S**2 - N)
    return E_zeeman + E_int

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_macro_matrix(B_val, J_val, T):
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    for n in range(dim):
        E[n] = get_E_macro(n, B_val, J_val)
        
    for n in range(dim):
        if n < N:
            dU = E[n+1] - E[n]
            M[n+1, n] = gamma * (N - n) * f_glauber(dU, T)
        if n > 0:
            dU = E[n-1] - E[n]
            M[n-1, n] = gamma * n * f_glauber(dU, T)
            
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M

# =============================================================================
# --- 3. CONSTRUCCIÓN DEL SISTEMA GLOBAL (2 Macroestados Colapsados) ---
# =============================================================================
def build_global_3level_matrix(E0, E1, E2, g0, g1, g2, T):
    """Matriz de 3x3 para el modelo global con 3 macro-estados colapsados."""
    M = np.zeros((3, 3))
    
    # Agrupamos en arrays para facilitar el cálculo
    E = np.array([E0, E1, E2])
    g = np.array([g0, g1, g2])
    
    # Calcular transiciones entre todos los pares posibles
    for source in range(3):
        for dest in range(3):
            if source != dest:
                dU = E[dest] - E[source]
                # Tasa de Glauber * degeneración del nivel de DESTINO
                M[dest, source] = gamma * g[dest] * f_glauber(dU, T)
                
    # Conservación de probabilidad en la diagonal (lo que sale resta)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    return M

# =============================================================================
# --- 4. DINÁMICA Y FISHER INFORMATION ---
# =============================================================================
def simulate_local_dynamics(B_val, J_val, t_array):
    M_base = build_macro_matrix(B_val, J_val, T_val)
    M_up   = build_macro_matrix(B_val, J_val, T_val + dT)
    M_dn   = build_macro_matrix(B_val, J_val, T_val - dT)
    
    dim = N + 1
    P0 = np.array([comb(N, n) for n in range(dim)])
    P0 /= np.sum(P0) 

    F_t = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t[i] = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
    return F_t

# Parámetros extraídos de tu optimizador numérico para d=16
E0, E1, E2 = -2.353, -0.864, 0.000
g0, g1, g2 = 4, 1, 11  # Degeneraciones de cada banda

def simulate_global_3level_dynamics(E0, E1, E2, g0, g1, g2, t_array):
    M_base = build_global_3level_matrix(E0, E1, E2, g0, g1, g2, T_val)
    M_up   = build_global_3level_matrix(E0, E1, E2, g0, g1, g2, T_val + dT)
    M_dn   = build_global_3level_matrix(E0, E1, E2, g0, g1, g2, T_val - dT)
    
    # ESTADO INICIAL a T = Infinito
    # La probabilidad se reparte en función del número de microestados (degeneración)
    total_states = g0 + g1 + g2
    P0 = np.array([g0 / total_states, g1 / total_states, g2 / total_states])

    F_t = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t[i] = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
    return F_t

# =============================================================================
# --- 5. EJECUCIÓN Y CÁLCULO ---
# =============================================================================
print(f"Calculando dinámica para Free Spins (Local, N={N})...")
F_free = simulate_local_dynamics(B_free, J_free, t_array)

print(f"Calculando dinámica para All-To-All (Local, N={N})...")
F_ata = simulate_local_dynamics(B_ata, J_ata, t_array)

print("Calculando dinámica para el Modelo Global (Paper Limit)...")
F_global = simulate_global_3level_dynamics(E0, E1, E2, g0, g1, g2, t_array)

# Calcular Eficiencia
eta_free   = F_free / t_array
eta_ata    = F_ata / t_array
eta_global = F_global / t_array

print(f"\nPico Máximo Free Spins: {np.max(eta_free):.4f}")
print(f"Pico Máximo All-To-All: {np.max(eta_ata):.4f}")
print(f"Pico Máximo GLOBAL:     {np.max(eta_global):.4f}")

# =============================================================================
# --- 6. REPRESENTACIÓN GRÁFICA ---
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# --- Plot Superior: Fisher Information F(t) ---
ax1.plot(t_array, F_free, label=f'Local: Free Spins ($B={B_free}$)', color='gray', linestyle='--', lw=2.5)
ax1.plot(t_array, F_ata, label=f'Local: All-To-All ($B={B_ata}, J={J_ata}$)', color='blue', lw=2.5)
ax1.plot(t_array, F_global, label=f'Global: 2-Level Colapso ($\Delta E={E_gap_global}$)', color='red', lw=2.5)
ax1.set_ylabel(r'$\mathcal{F}(t)$', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.4)
ax1.set_title(f'Dinámica Termometral Transitoria (N={N})', fontsize=16)

# --- Plot Inferior: Eficiencia \eta(t) ---
ax2.plot(t_array, eta_free, label='Local: Free Spins', color='gray', linestyle='--', lw=2.5)
ax2.plot(t_array, eta_ata, label='Local: All-To-All', color='blue', lw=2.5)
ax2.plot(t_array, eta_global, label='Global: Límite Teórico Absoluto', color='red', lw=2.5)
ax2.set_xlabel('Tiempo $t$', fontsize=16)
ax2.set_ylabel(r'$\eta(t) = \frac{\mathcal{F}(t)}{t}$', fontsize=16)

ax2.grid(alpha=0.4)

plt.tight_layout()
plt.show()