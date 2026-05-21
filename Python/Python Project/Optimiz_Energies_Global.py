import numpy as np
import scipy.linalg as la
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import time

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 5
dim = 2**N  # Número genérico de niveles (equivalente a 2^N)
T_val = 1.0
dT = 0.001
gamma = 1.0

# Malla de tiempo
t_array = np.linspace(0.01, 2.0, 100)

# Límites de búsqueda para las energías [-10, 10]
# Fijamos el nivel 0 a energía E=0. Optimizamos los 7 restantes.
bounds_E = [(-10.0, 10.0)] * (dim - 1)

# =============================================================================
# --- 2. CONSTRUCCIÓN DE LA MATRIZ (ACOPLAMIENTO GLOBAL) ---
# =============================================================================
def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_matrix_global_energies(E_array, T):
    """
    Construye la matriz de Glauber asumiendo ACOPLAMIENTO GLOBAL.
    TODAS las transiciones entre cualquier par de niveles están permitidas.
    """
    M = np.zeros((dim, dim))
    
    for state in range(dim):
        for neighbor in range(dim):
            if state != neighbor:
                dU = E_array[neighbor] - E_array[state]
                # Tasa de Glauber pura para cualquier salto
                M[neighbor, state] = gamma * f_glauber(dU, T)
            
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M

# =============================================================================
# --- 3. FUNCIÓN OBJETIVO (MÁXIMO DE LA EFICIENCIA) ---
# =============================================================================
def objective_global(params):
    E_array = np.zeros(dim)
    E_array[1:] = params
    
    M_base = build_matrix_global_energies(E_array, T_val)
    M_up   = build_matrix_global_energies(E_array, T_val + dT)
    M_dn   = build_matrix_global_energies(E_array, T_val - dT)
    
    # ESTADO INICIAL: T = Infinito (Todos los niveles equiprobables)
    P0 = np.ones(dim) / dim
    
    
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
            
    return -eta_max

# =============================================================================
# --- 4. EJECUCIÓN DEL OPTIMIZADOR ---
# =============================================================================
print(f"Iniciando optimización topológica GLOBAL para {dim} niveles...")
start_time = time.time()

result = differential_evolution(objective_global, bounds_E, 
                                strategy='best1bin', popsize=15, maxiter=100, disp=False)

E_opt = np.zeros(dim)
E_opt[1:] = result.x
eta_pico = -result.fun

print("\n--- ¡OPTIMIZACIÓN COMPLETADA! ---")
print(f"Tiempo de ejecución: {time.time() - start_time:.2f} segundos")
print(f"Pico Máximo encontrado: eta_max = {eta_pico:.4f}")

# =============================================================================
# --- 5. ANÁLISIS DEL ESPECTRO FÍSICO ---
# =============================================================================
E_sorted = np.sort(E_opt)

print("\n--- ESPECTRO DE ENERGÍAS ÓPTIMO (Ordenado) ---")
for i, e in enumerate(E_sorted):
    print(f"Nivel {i}: {e:.4f}")

plt.figure(figsize=(8, 5))
plt.hlines(E_sorted, xmin=0.2, xmax=0.8, colors='red', linewidth=2)
plt.title(f'Espectro Óptimo (Acoplamiento Global, {dim} Niveles)', fontsize=14)
plt.ylabel('Energía', fontsize=12)
plt.xticks([]) 
plt.grid(alpha=0.3, axis='y')
plt.show()