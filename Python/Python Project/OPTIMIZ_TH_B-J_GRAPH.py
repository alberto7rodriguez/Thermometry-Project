import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import comb

# =============================================================================
# --- 1. FUNCIÓN PARA CALCULAR CHI EN EL MODELO HOMOGÉNEO (All-To-All) ---
# =============================================================================
def calc_chi_BJ(B, J, N, beta=1.0, gamma=1.0):
    n_levels = N + 1
    
    # 1. Energías y degeneraciones exactas para el All-To-All homogéneo
    E = np.zeros(n_levels)
    g = np.zeros(n_levels)
    
    for k in range(n_levels):
        # Magnetización M = (número de espines up) - (número de espines down)
        M = 2*k - N 
        E[k] = B * M + 0.5 * J * (M**2 - N)
        g[k] = comb(N, k)
        
    # 2. Termodinámica (Capacidad Calorífica C)
    E_shifted = E - np.min(E)
    weights = g * np.exp(-beta * E_shifted)
    Z = np.sum(weights)
    P = weights / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    if C < 1e-12: return 0.0
    
    # 3. Dinámica (Tiempo de Relajación tau)
    M_matrix = np.zeros((n_levels, n_levels))
    for k in range(n_levels - 1):
        delta_E = E[k+1] - E[k]
        
        # Conectividad: desde k, hay (N-k) espines abajo que pueden subir
        K_up = g[k] * (N - k)
        
        Gamma_up = gamma * (K_up / g[k]) * (1.0 / (1.0 + np.exp(beta * delta_E)))
        Gamma_down = gamma * (K_up / g[k+1]) * (1.0 / (1.0 + np.exp(-beta * delta_E)))
        
        M_matrix[k+1, k] = Gamma_up
        M_matrix[k, k+1] = Gamma_down

    np.fill_diagonal(M_matrix, 0)
    np.fill_diagonal(M_matrix, -np.sum(M_matrix, axis=0))
    
    evals = la.eigvals(M_matrix).real
    evals = np.sort(evals)
    lambda_1 = np.abs(evals[-2])
    
    tau = 1.0 / (lambda_1 + 1e-15)
    return C 

# =============================================================================
# --- 2. CONFIGURACIÓN DEL BARRIDO (GRID) ---
# =============================================================================
N_spins = 6
grid_size = 100 # Resolución del gráfico (100x100 píxeles)

# Rangos de exploración
# Solo exploramos B > 0 por simetría (B < 0 daría el mismo mapa exacto)
B_vals = np.linspace(-4.0, 4.0, grid_size) 
J_vals = np.linspace(-1.5, 1.5, grid_size)

B_grid, J_grid = np.meshgrid(B_vals, J_vals)
chi_grid = np.zeros((grid_size, grid_size))

print("Calculando el mapa de calor de chi(B, J)...")

# Llenar la matriz calculando chi para cada punto (B, J)
max_chi = -1
best_B = 0
best_J = 0

for i in range(grid_size):
    for j in range(grid_size):
        chi = calc_chi_BJ(B_grid[i, j], J_grid[i, j], N_spins)
        chi_grid[i, j] = chi
        
        if chi > max_chi:
            max_chi = chi
            best_B = B_grid[i, j]
            best_J = J_grid[i, j]

print(f"¡Cálculo terminado! Máximo encontrado en B = {best_B:.3f}, J = {best_J:.3f} (Chi = {max_chi:.4f})")

# =============================================================================
# --- 3. DIBUJAR EL MAPA DE CALOR ---
# =============================================================================
plt.figure(figsize=(10, 8))

# Generar el contorno de color
contour = plt.contourf(B_grid, J_grid, chi_grid, levels=50, cmap='inferno')
cbar = plt.colorbar(contour)
cbar.set_label(r'Precisión a tiempo finito $\chi = \mathcal{C}/\tau$', fontsize=14)

# Marcar la línea central de interacciones nulas (J = 0)
plt.axhline(0, color='white', linestyle='--', alpha=0.6, label='Espines Libres ($J=0$)')

# Marcar el punto máximo absoluto
plt.plot(best_B, best_J, 'w*', markersize=15, markeredgecolor='black', label=f'Máximo global')

# Etiquetas y estética
plt.title(f'Mapa de Termometría $\chi(B, J)$ para $N={N_spins}$ espines', fontsize=16)
plt.xlabel(r'Campo magnético local $B$', fontsize=14)
plt.ylabel(r'Fuerza de interacción $J$', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()

# Mostrar gráfico
plt.show()