import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig, inv
from scipy.special import comb

# --- Parameters ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

def f(x, beta):
    # Clip para evitar overflow en la exponencial si los gaps son muy grandes
    return 1.0 / (1.0 + np.exp(np.clip(beta * x, -700, 700)))

def transition_matrix_star(N, a, b, beta):
    dim = 2 * N
    M = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    # Energías: 0 a N-1 (Central DOWN), N a 2N-1 (Central UP)
    for k in range(N):
        E[k] = -a
        E[k + N] = a + 2 * b * (2 * k - N + 1)
        
    for k in range(N):
        idx_dn = k
        idx_up = k + N
        
        # 1. Saltos del espín central (cambian de manifold)
        dU_central = E[idx_up] - E[idx_dn]
        M[idx_up, idx_dn] = f(dU_central, beta)
        M[idx_dn, idx_up] = f(-dU_central, beta)
        
        # 2. Saltos de los satélites en el manifold DOWN (dU = 0)
        if k < N - 1:
            M[idx_dn + 1, idx_dn] = (N - 1 - k) * f(0, beta) # f(0) es 0.5
        if k > 0:
            M[idx_dn - 1, idx_dn] = k * f(0, beta)
            
        # 3. Saltos de los satélites en el manifold UP
        if k < N - 1:
            dU_up = E[idx_up + 1] - E[idx_up]
            M[idx_up + 1, idx_up] = (N - 1 - k) * f(dU_up, beta)
        if k > 0:
            dU_dn = E[idx_up - 1] - E[idx_up]
            M[idx_up - 1, idx_up] = k * f(dU_dn, beta)

    # Conservación de probabilidad en la diagonal
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M

# --- Configuración del Sistema ---
N = 15
a = a_values[N-2]
b = b_values[N-2]
beta = 1.0
t = 20.0  

# Condición Inicial: Estado de Máxima Entropía (Infinito T)
p0 = np.zeros(2 * N)
for k in range(N):
    deg = comb(N - 1, k)
    p0[k] = deg       # Subespacio DOWN
    p0[k + N] = deg   # Subespacio UP
p0 /= np.sum(p0)

# Construir Matriz y Evolucionar
M = transition_matrix_star(N, a, b, beta)
eMt = expm(M * t)
pt = eMt @ p0

# Diagonalización espectral
eigvals, eigvecs = eig(M)
eigvals = np.real(eigvals)  # Descartamos parte imaginaria residual numérica
eigvecs = np.real(eigvecs)

# --- Imprimir Autovalores Ordenados ---
sorted_indices = np.argsort(eigvals)[::-1] # Orden descendente (de 0 hacia los negativos)
sorted_eigvals = eigvals[sorted_indices]

print("-" * 65)
# Usamos los símbolos λ y τ directamente para evitar el error del backslash
print(f"{'Modo (i)':<10} | {'Eigenvalue (λ_i)':<20} | {'Timescale (τ_i = -1/λ_i)':<25}")
print("-" * 65)

for i, val in enumerate(sorted_eigvals):
    if np.abs(val) < 1e-12:  # Protegemos contra el cero numérico del estado estacionario
        tau_str = "inf (Steady State)"
    else:
        tau = -1.0 / val
        tau_str = f"{tau:.6f}"
    
    # Aquí el backslash de \u03bb sí funciona porque está fuera de las llaves {}
    print(f"{i:<10} | {val:<20.6f} | {tau_str:<25}")
print("-" * 65)

# Proyectar condición inicial
V_inv = inv(eigvecs)
coeffs = V_inv @ p0

# Imprimir ecuaciones explícitas
print("Explicit time dependence of p_k(t):")
print("\n--- Subespacio: Espín Central DOWN ---")
for k in range(N):
    terms = []
    for i in range(2 * N):
        amplitude = eigvecs[k, i] * coeffs[i]
        if np.abs(amplitude) > 1e-6:
            decay = eigvals[i]
            terms.append(f"{amplitude:.3f} * np.exp({decay:.3f} * t)")
    print(f"p(\u2193, {k})(t) = " + " + ".join(terms))

print("\n--- Subespacio: Espín Central UP ---")
for k in range(N):
    terms = []
    for i in range(2 * N):
        idx = k + N
        amplitude = eigvecs[idx, i] * coeffs[i]
        if np.abs(amplitude) > 1e-6:
            decay = eigvals[i]
            terms.append(f"{amplitude:.3f} * np.exp({decay:.3f} * t)")
    print(f"p(\u2191, {k})(t) = " + " + ".join(terms))
print("-" * 50)