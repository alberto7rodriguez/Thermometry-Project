import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS FÍSICOS ---
# =============================================================================
N = 3
T_val = 1.0
dT = 0.001
gamma = 1.0

# Parámetros óptimos encontrados para N=3
B_ata = 1.2633
J_ata = 0.0025

# Parámetros óptimos teóricos para Free Spins
B_free = 1.2759
J_free = 0.0

# Malla de tiempo centrada en el transitorio corto
t_array = np.linspace(0.1, 25.0, 500)

# =============================================================================
# --- 2. CONSTRUCCIÓN DEL SISTEMA (MACROESTADOS N+1) ---
# =============================================================================
def get_E_macro(n, B_val, J_val):
    """
    Calcula la energía de un macroestado con 'n' espines hacia arriba.
    Magnetización total S = (n) - (N-n) = 2n - N
    H = - B*S - J/2 * (S^2 - N)
    """
    S = 2 * n - N
    E_zeeman = -B_val * S
    E_int = -0.5 * J_val * (S**2 - N)
    return E_zeeman + E_int

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_macro_matrix(B_val, J_val, T):
    """Construye la matriz de tasas agrupada por macroestados (tamaño N+1)."""
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    # Calcular las energías de cada macroestado
    for n in range(dim):
        E[n] = get_E_macro(n, B_val, J_val)
        
    # Asignar tasas de transición con degeneraciones
    for n in range(dim):
        # Transición n -> n+1 (voltear un espín de abajo a arriba)
        if n < N:
            dU = E[n+1] - E[n]
            M[n+1, n] = gamma * (N - n) * f_glauber(dU, T)
            
        # Transición n -> n-1 (voltear un espín de arriba a abajo)
        if n > 0:
            dU = E[n-1] - E[n]
            M[n-1, n] = gamma * n * f_glauber(dU, T)
            
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

# =============================================================================
# --- 3. DINÁMICA Y FISHER INFORMATION ---
# =============================================================================
def simulate_fisher_dynamics(B_val, J_val, t_array):
    # Matrices a T, T+dT, T-dT
    M_base, _ = build_macro_matrix(B_val, J_val, T_val)
    M_up, _   = build_macro_matrix(B_val, J_val, T_val + dT)
    M_dn, _   = build_macro_matrix(B_val, J_val, T_val - dT)
    
    # ESTADO INICIAL: T=infinito (Distribución Binomial)
    # Todos los microestados son equiprobables, por lo que la probabilidad
    # de un macroestado n viene dada por el número combinatorio.
    
    dim = N + 1
    
    P0 = np.array([comb(N, n) for n in range(dim)])
    P0 /= np.sum(P0) # Normalización equivalente a dividir por 2^N
    '''
    P0 = np.zeros(dim)
    P0[-1] = 1
    '''
    F_t = np.zeros(len(t_array))

    for i, t in enumerate(t_array):
        # Evolución temporal usando la exponencial de la matriz reducida
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        # Derivada numérica de la probabilidad respecto a T
        dP_dT = (P_up - P_dn) / (2 * dT)
        
        # Información de Fisher (suma sobre macroestados)
        # Multiplicamos implícitamente por 1 porque las degeneraciones 
        # ya determinaron las trayectorias de probabilidad de los macroestados.
        F_t[i] = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
    return F_t

# =============================================================================
# --- 4. EJECUCIÓN Y CÁLCULO ---
# =============================================================================
print(f"Calculando dinámica para Free Spins (N={N})...")
F_free = simulate_fisher_dynamics(B_free, J_free, t_array)

print(f"Calculando dinámica para All-To-All (N={N})...")
F_ata = simulate_fisher_dynamics(B_ata, J_ata, t_array)

# Calcular Eficiencia
eta_free = F_free / t_array
eta_ata  = F_ata / t_array

print(f"\nPico Máximo Free Spins: {np.max(eta_free):.4f}")
print(f"Pico Máximo All-To-All: {np.max(eta_ata):.4f}")

# =============================================================================
# --- 5. REPRESENTACIÓN GRÁFICA ---
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- Plot Superior: Fisher Information F(t) ---
ax1.plot(t_array, F_free, label=f'Free Spins ($B={B_free}$)', color='gray', linestyle='--', lw=2.5)
ax1.plot(t_array, F_ata, label=f'All-To-All ($B={B_ata}, J={J_ata}$)', color='blue', lw=2.5)
ax1.set_ylabel(r'$\mathcal{F}(t)$', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.4)
ax1.set_title(f'Dinámica Termometral Transitoria (N={N})', fontsize=16)

# --- Plot Inferior: Eficiencia \eta(t) ---
ax2.plot(t_array, eta_free, label='Free Spins', color='gray', linestyle='--', lw=2.5)
ax2.plot(t_array, eta_ata, label='All-To-All', color='blue', lw=2.5)
ax2.set_xlabel('Tiempo $t$', fontsize=16)
ax2.set_ylabel(r'$\eta(t) = \frac{\mathcal{F}(t)}{t}$', fontsize=16)
ax2.grid(alpha=0.4)

plt.tight_layout()
plt.show()