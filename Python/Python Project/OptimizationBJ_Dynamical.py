import numpy as np
import scipy.linalg as la
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 4 
T_val = 1.0
dT = 0.001
gamma = 1.0

# Malla de tiempo corta (centrada cerca de 0, evitando t=0 exacto para no dividir por 0)
t_array = np.linspace(0.01, 2.0, 100) 

# Límites de búsqueda para los parámetros [Min, Max]
# Esto es vital para que el algoritmo no busque en el infinito
bound_B = (-5.0, 5.0)
bound_J = (-5.0, 5.0)

# =============================================================================
# --- 2. CONSTRUCCIÓN GENERAL DEL SISTEMA ---
# =============================================================================
def get_energy(state, B_array, J_matrix):
    """Calcula la energía de un microestado dado (en binario)."""
    # Convertimos el entero 'state' a un array de espines +1/-1
    spins = np.array([1 if (state & (1 << i)) else -1 for i in range(N)])
    
    E = -np.sum(B_array * spins)
    for i in range(N):
        for j in range(i + 1, N):
            E -= J_matrix[i, j] * spins[i] * spins[j]
    return E

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_general_matrix(B_array, J_matrix, T):
    """Construye la matriz de tasas para un Hamiltoniano arbitrario."""
    dim = 2**N
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    for state in range(dim):
        E[state] = get_energy(state, B_array, J_matrix)
        
    for state in range(dim):
        for i in range(N):
            # El estado vecino es el mismo pero con el bit 'i' invertido
            neighbor = state ^ (1 << i)
            dU = E[neighbor] - E[state]
            M[neighbor, state] = gamma * f_glauber(dU, T)
            
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

# =============================================================================
# --- 3. FUNCIÓN OBJETIVO A MAXIMIZAR ---
# =============================================================================
def objective_function(params):
    """
    Recibe un vector de parámetros planos.
    Construye el sistema, calcula eta(t) y devuelve el -MAX(eta).
    (Devuelve negativo porque scipy.optimize siempre MINIMIZA)
    """
    # Desempaquetar parámetros
    B_array = params[:N]
    J_flat = params[N:]
    
    J_matrix = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            J_matrix[i, j] = J_flat[idx]
            J_matrix[j, i] = J_flat[idx]
            idx += 1

    # Construir matrices para derivadas
    M_base, E_base = build_general_matrix(B_array, J_matrix, T_val)
    M_up, _        = build_general_matrix(B_array, J_matrix, T_val + dT)
    M_dn, _        = build_general_matrix(B_array, J_matrix, T_val - dT)
    
    # Condición inicial (Ej: T=infinito, todos los estados equiprobables)
    dim = 2**N
    P0 = np.zeros(dim)
    P0[-1] = 1
    
    # Evaluar la dinámica en la malla de tiempo
    eta_max = 0.0
    for t in t_array:
        # Nota: Para N muy pequeños (N<=5), expm es suficientemente rápido.
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
        eta_t = F_t / t
        if eta_t > eta_max:
            eta_max = eta_t
            
    return -eta_max # Retornamos el negativo para minimizar

# =============================================================================
# --- 4. EJECUCIÓN DE LA OPTIMIZACIÓN ---
# =============================================================================
# Definir límites de las variables
num_J = N * (N - 1) // 2
bounds = [bound_B] * N + [bound_J] * num_J

print(f"Iniciando optimización para N={N} ({len(bounds)} parámetros)...")
# popsize=15 y maxiter=50 es un buen equilibrio entre rapidez y exploración profunda
result = differential_evolution(objective_function, bounds, strategy='best1bin', 
                                popsize=15, maxiter=100, disp=True)

print("\n--- ¡OPTIMIZACIÓN COMPLETADA! ---")
print(f"Pico Máximo de Eficiencia encontrado: eta_max = {-result.fun:.4f}")

B_opt = result.x[:N]
J_opt = result.x[N:]
print("\nCampos B óptimos:", np.round(B_opt, 3))
print("Acoplamientos J óptimos:", np.round(J_opt, 3))