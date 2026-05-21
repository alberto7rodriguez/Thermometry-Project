import numpy as np
import scipy.linalg as la
from scipy.optimize import differential_evolution
from scipy.special import comb
import time

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 4         # ¡Ahora puedes subir esto a 10, 20 o 50 sin problema!
T_val = 1.0
dT = 0.001
gamma = 1.0

# Malla de tiempos para buscar el pico
t_array = np.linspace(0.01, 5.0, 100)

# Límites de búsqueda para [B, J]
bounds = [(-5.0, 5.0), (-2.0, 2.0)] 

# =============================================================================
# --- 2. CONSTRUCCIÓN DEL MODELO ATA (MACROESTADOS) ---
# =============================================================================
def get_E_ata(n, B, J):
    """
    Energía para un macroestado con 'n' espines hacia arriba.
    Magnetización total S = (n) - (N-n) = 2n - N
    H = - B*S - J/2 * (S^2 - N)
    """
    S = 2 * n - N
    E_zeeman = -B * S
    E_int = -0.5 * J * (S**2 - N)
    return E_zeeman + E_int

def f_glauber(dU, T):
    return 1.0 / (1.0 + np.exp(np.clip(dU / T, -100, 100)))

def build_ata_macro_matrix(B, J, T):
    """
    Construye la matriz de transición de tamaño (N+1) x (N+1).
    """
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    # Calcular energías de los macroestados
    for n in range(dim):
        E[n] = get_E_ata(n, B, J)
        
    # Calcular tasas de transición
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
# --- 3. FUNCIÓN OBJETIVO (MAXIMIZAR EL PICO DE ETA) ---
# =============================================================================
def objective_ata(params):
    B, J = params
    
    # Matrices para la derivada numérica
    M_base, _ = build_ata_macro_matrix(B, J, T_val)
    M_up, _   = build_ata_macro_matrix(B, J, T_val + dT)
    M_dn, _   = build_ata_macro_matrix(B, J, T_val - dT)
    
    # ESTADO INICIAL: T = Infinito (Distribución Binomial en macroestados)
    P0 = np.array([comb(N, n) for n in range(N + 1)])
    P0 /= np.sum(P0)
    '''

    # ESTADO INICIAL: Ground State
    P0 = np.zeros(N + 1)
    P0[-1] = 1.0
    '''

    
    eta_max = 0.0
    for t in t_array:
        # Evolución temporal
        P_base = la.expm(M_base * t) @ P0
        P_up   = la.expm(M_up * t) @ P0
        P_dn   = la.expm(M_dn * t) @ P0
        
        # Derivada y Fisher Information
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15))
        
        eta_t = F_t / t
        if eta_t > eta_max:
            eta_max = eta_t
            
    return -eta_max # Negativo para minimizar en SciPy

# =============================================================================
# --- 4. EJECUCIÓN DE LA OPTIMIZACIÓN ---
# =============================================================================
print(f"Iniciando optimización del modelo All-To-All para N={N}...")
start_time = time.time()

# Al ser solo 2D, el optimizador global volará. Usamos una población más alta por seguridad.
result = differential_evolution(objective_ata, bounds, 
                                strategy='best1bin', 
                                popsize=20, maxiter=100, tol=1e-6, disp=False)

end_time = time.time()

B_opt, J_opt = result.x
eta_pico = -result.fun

print("\n--- ¡OPTIMIZACIÓN COMPLETADA! ---")
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
print(f"Pico Máximo encontrado: eta_max = {eta_pico:.4f}")
print(f"Campo B óptimo: {B_opt:.4f}")
print(f"Acoplamiento J óptimo: {J_opt:.4f}")