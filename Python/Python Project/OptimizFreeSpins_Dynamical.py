import numpy as np
import scipy.linalg as la
from scipy.optimize import differential_evolution
from scipy.special import comb
import time

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 1 
T_val = 1.0
dT = 0.0001
gamma = 1.0

# Malla de tiempo corta
t_array = np.linspace(0.01, 5.0, 500) 

# Límites de búsqueda para el único parámetro [B]
bounds_free = [(-5.0, 5.0)]


# =============================================================================
# --- 2. CONSTRUCCIÓN DEL SISTEMA (MACROESTADOS) ---
# =============================================================================
def get_E_free(n, B):
    """
    Energía para un macroestado con 'n' espines hacia arriba en un modelo Free Spins.
    Magnetización total S = (n) - (N-n) = 2n - N
    H = - B*S
    """
    S = 2 * n - N
    return -B * S

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_free_macro_matrix(B, T):
    """Construye la matriz de tasas de tamaño (N+1)x(N+1) para Free Spins."""
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    # Calcular energías
    for n in range(dim):
        E[n] = get_E_free(n, B)
        
    # Calcular tasas con degeneraciones explícitas
    for n in range(dim):
        if n < N:
            dU = E[n+1] - E[n]
            M[n+1, n] = gamma * (N - n) * f_glauber(dU, T)
            
        if n > 0:
            dU = E[n-1] - E[n]
            M[n-1, n] = gamma * n * f_glauber(dU, T)
            
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

# =============================================================================
# --- 3. FUNCIÓN OBJETIVO A MAXIMIZAR ---
# =============================================================================
def objective_function_free_spins(params):
    """Recibe 1 solo parámetro (el campo B global)"""
    B = params[0]
    
    # Construir matrices para derivadas
    M_base, _ = build_free_macro_matrix(B, T_val)
    M_up, _   = build_free_macro_matrix(B, T_val + dT)
    M_dn, _   = build_free_macro_matrix(B, T_val - dT)
    
    
    dim = N + 1
    '''
    P0 = np.array([comb(N, n) for n in range(dim)])
    P0 /= np.sum(P0)
    '''
    P0 = np.zeros(dim)
    P0[-1] = 1
    # Evaluar la dinámica en la malla de tiempo
    eta_max = 0.0
    for t in t_array:
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
        eta_t = F_t / t
        if eta_t > eta_max:
            eta_max = eta_t
            
    return -eta_max # Negativo para minimizar en SciPy

# =============================================================================
# --- 4. EJECUCIÓN DE LA OPTIMIZACIÓN ---
# =============================================================================
print(f"Optimizando Free Spins (Macroestados) para N={N}...")
start_time = time.time()

# Al ser 1D, converge instantáneamente
result_free = differential_evolution(objective_function_free_spins, bounds_free, 
                                     strategy='best1bin', popsize=15, maxiter=50, disp=False)

end_time = time.time()
B_opt_free = result_free.x[0]
eta_pico = -result_free.fun

print("\n--- ¡OPTIMIZACIÓN COMPLETADA! ---")
print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")
print(f"Pico Máximo encontrado: eta_max = {eta_pico:.4f}")
print(f"Campo B óptimo (Free Spins): {B_opt_free:.4f}")