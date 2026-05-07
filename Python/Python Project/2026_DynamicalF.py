import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS Y LISTAS ---
# =============================================================================
N = 3  # Elegimos N=10 (el máximo común de tus listas)
idx = N - 2 # El índice en las listas (N=2 es index 0, N=10 es index 8)

J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464]
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

J_ata = J_values[idx]
a_star = a_values[idx]
b_star = b_values[idx]

# Espines libres óptimos (B ~ 1.25 maximiza C para N espines a T=1)
h_free = 1.25 

T_val = 1.0
dT = 0.0008
gamma = 1.0

def f_glauber(dU, T):
    exponent = np.clip(dU / T, -100, 100)
    return 1.0 / (1.0 + np.exp(exponent))

# =============================================================================
# --- 2. CONSTRUCCIÓN DE MATRICES ---
# =============================================================================
def build_ata_matrix(N, J, T):
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    for n in range(dim):
        E[n] = J * (-N*(N+1)/2.0 + 2*(n+1)*(N-n))
    for n in range(dim):
        if n < N:
            M[n+1, n] = gamma * (N - n) * f_glauber(E[n+1] - E[n], T)
        if n > 0:
            M[n-1, n] = gamma * n * f_glauber(E[n-1] - E[n], T)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

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

def build_free_matrix(N, h, T):
    dim = N + 1
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    for n in range(dim):
        E[n] = 2 * h * n - h * N
    for n in range(dim):
        if n < N:
            M[n+1, n] = gamma * (N - n) * f_glauber(E[n+1] - E[n], T)
        if n > 0:
            M[n-1, n] = gamma * n * f_glauber(E[n-1] - E[n], T)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, E

# =============================================================================
# --- 3. PASO INTERMEDIO: DIAGONALIZACIÓN Y EXTRACCIÓN DE ECUACIONES ---
# =============================================================================
def get_explicit_equations(M, E, is_star=False):
    """
    Diagonaliza M y devuelve las lambdas y las amplitudes A_{n,i}.
    La probabilidad es: p_n(t) = sum_i A_{n,i} * exp(lambda_i * t)
    """
    from scipy.special import comb
    
    D_states = len(E)
    
    # Condición inicial T=infinito (Máxima entropía, microestados equiprobables)
    if is_star:
        N_spins = D_states // 2
        P0 = np.zeros(D_states)
        for k in range(N_spins):
            degeneracy = comb(N_spins - 1, k)
            P0[k] = degeneracy           # Satélites con Central DOWN
            P0[k + N_spins] = degeneracy # Satélites con Central UP
    else:
        N_spins = D_states - 1
        P0 = np.array([comb(N_spins, n) for n in range(D_states)])
        
    # Normalizamos para asegurar que la suma total de probabilidades sea 1
    P0 /= np.sum(P0)

    # Diagonalización espectral: M = V * L * V^-1
    evals, evecs = la.eig(M)
    
    # Descartamos partes imaginarias numéricas
    evals = np.real(evals)
    evecs = np.real(evecs)
    
    # Resolver los coeficientes iniciales C: V * C = P0  => C = V^-1 * P0
    C = la.solve(evecs, P0)
    
    # Matriz de amplitudes: A[n, i] = evecs[n, i] * C[i]
    amplitudes = evecs * C[np.newaxis, :]
    
    return evals, amplitudes

def print_equations_json(evals, amplitudes, model_name):
    """Imprime las ecuaciones explícitas en formato JSON-like para su inspección"""
    eq_dict = {}
    for n in range(amplitudes.shape[0]):
        terms = []
        for i in range(len(evals)):
            amp = amplitudes[n, i]
            lam = evals[i]
            if np.abs(amp) > 1e-10: # Filtramos términos numéricamente nulos
                terms.append({"amplitude": round(amp, 6), "lambda": round(lam, 6)})
        eq_dict[f"p_{n}(t)"] = terms
    
    print(f"\n--- Ecuaciones Explícitas para {model_name} (T=1.0) ---")
    print(json.dumps(eq_dict, indent=4))

# =============================================================================
# --- 4. EVALUACIÓN DE LAS ECUACIONES Y FISHER ---
# =============================================================================
def evaluate_probability(t, evals, amplitudes):
    """Evalúa p_n(t) usando la fórmula analítica explícita."""
    # exp_L shape: (num_modos,)
    exp_L = np.exp(evals * t)
    # p(t) = Sum_i A_{n,i} * exp(lambda_i * t)
    return np.dot(amplitudes, exp_L)

def simulate_fisher_explicit(build_matrix_func, args, t_array, is_star=False):
    # 1. Construir las 3 matrices (T, T+dT, T-dT)
    M_base, E_base = build_matrix_func(*args, T_val)
    M_up, E_up     = build_matrix_func(*args, T_val + dT)
    M_dn, E_dn     = build_matrix_func(*args, T_val - dT)
    
    # 2. Extraer ecuaciones explícitas para cada temperatura
    L_base, A_base = get_explicit_equations(M_base, E_base, is_star)
    L_up, A_up     = get_explicit_equations(M_up, E_up, is_star)
    L_dn, A_dn     = get_explicit_equations(M_dn, E_dn, is_star)
    
    # Imprimimos las ecuaciones base solo para comprobar que existen y son correctas
    if build_matrix_func.__name__ == 'build_ata_matrix':
        print_equations_json(L_base, A_base, "All-To-All Model")
    
    # 3. Evolución en el tiempo usando las fórmulas analíticas
    F_t = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        if t == 0: continue
            
        P_base = evaluate_probability(t, L_base, A_base)
        P_up   = evaluate_probability(t, L_up, A_up)
        P_dn   = evaluate_probability(t, L_dn, A_dn)
        
        # Derivada F(T+h, t) - F(T-h, t) / 2h
        dP_dT = (P_up - P_dn) / (2 * dT)
        F_t[i] = np.sum((dP_dT**2) / (np.abs(P_base) + 1e-15)) # abs() por seguridad numérica
        
    return F_t

# =============================================================================
# --- 5. EJECUCIÓN PRINCIPAL ---
# =============================================================================
t_array = np.linspace(0.1, 40, 1000)

F_free = simulate_fisher_explicit(build_free_matrix, (N, h_free), t_array)
F_ata  = simulate_fisher_explicit(build_ata_matrix, (N, J_ata), t_array)
F_star = simulate_fisher_explicit(build_star_matrix, (N, a_star, b_star), t_array, is_star=True)

tau_meas = 0 
eta_free = F_free[1:] / (t_array[1:] + tau_meas)
eta_ata  = F_ata[1:]  / (t_array[1:] + tau_meas)
eta_star = F_star[1:] / (t_array[1:] + tau_meas)

# --- PLOT ---
# Cambiamos a 2 filas, 1 columna y activamos sharex=True
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- Primer plot (Superior: Información de Fisher) ---
ax1.plot(t_array, F_free, label='Espines Libres ($J=0$)', color='gray', linestyle='--', lw=2.5)
ax1.plot(t_array, F_ata, label='All-To-All', color='blue', lw=2)
ax1.plot(t_array, F_star, label='Star Model', color='red', lw=2)
# Eliminamos set_xlabel de aquí para que no se duplique
ax1.set_ylabel(r'$\mathcal{F}(t)$', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.4)

# --- Segundo plot (Inferior: Precisión) ---
ax2.plot(t_array[1:], eta_free, label='Espines Libres', color='gray', linestyle='--', lw=2.5)
ax2.plot(t_array[1:], eta_ata, label='All-To-All', color='blue', lw=2)
ax2.plot(t_array[1:], eta_star, label='Star Model', color='red', lw=2)
# El xlabel solo se queda en el plot de abajo
ax2.set_xlabel('Tiempo de medición $t$', fontsize=14)
ax2.set_ylabel(r'$\eta = \frac{\mathcal{F}(t)}{t}$', fontsize=14)
ax2.grid(alpha=0.4)

# Aplicamos la escala personalizada al eje compartido (usando ax2)
ax2.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))

# Ajuste fino para evitar que los títulos o etiquetas se solapen
plt.tight_layout()
plt.show()