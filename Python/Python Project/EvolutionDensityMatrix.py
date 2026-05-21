import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# --- 1. CONFIGURACIÓN DEL SISTEMA ---
# =============================================================================
N_total = 4  # Cambia a 4 para simular N=4

beta = 1.0
gamma = 1.0  # Acoplamiento base con el baño

if N_total == 3:
    dim = 8
    H = np.zeros((dim, dim))
    # GS (k=0) -> Índice 0
    H[0, 0] = -3.188
    # k=1 (Degeneración 2) -> Índices 1, 2
    H[1, 1] = 0.000
    H[2, 2] = 0.000
    # k=2 (Degeneración 1) -> Índice 3
    H[3, 3] = 3.188
    # Spin off (Degeneración 4) -> Índices 4, 5, 6, 7
    for i in range(4, 8):
        H[i, i] = 0.000

elif N_total == 4:
    dim = 16
    H = np.zeros((dim, dim))
    
    Delta = 1e-3  # Introducimos la pequeña rotura de simetría (imperfección física)
    
    # GS (k=0) -> Índice 0
    H[0, 0] = -4.470
    
    # k=1 (Degeneración 3 rota por Delta) -> Índices 1, 2, 3
    H[1, 1] = -0.894 + 1 * Delta
    H[2, 2] = -0.894 + 2 * Delta
    H[3, 3] = -0.894 + 3 * Delta
    
    # k=2 (Degeneración 3) -> Índices 4, 5, 6
    H[4, 4] = 2.682
    H[5, 5] = 2.682
    H[6, 6] = 2.682
    
    # k=3 (Degeneración 1) -> Índice 7
    H[7, 7] = 6.258
    
    # Spin off (Degeneración 8 rota por Delta) -> Índices 8 a 15
    for i in range(8, 16):
        H[i, i] = -0.894 + i * Delta

# =============================================================================
# --- 2. OPERADORES DE SALTO (BAÑO TÉRMICO COLECTIVO) ---
# =============================================================================
# Usamos la energía base teórica del primer excitado (-0.894 para N=4)
E_GS = H[0, 0]
E_1st_exc_base = -0.894 if N_total == 4 else 0.000 
Delta_E = E_1st_exc_base - E_GS

# Tasas de absorción y emisión
n_th = 1.0 / (np.exp(beta * Delta_E) - 1.0)
Gamma_down = gamma * (n_th + 1.0)
Gamma_up = gamma * n_th

A_down = np.zeros((dim, dim), dtype=complex)
for i in range(1, dim):
    # RELAJAMOS LA TOLERANCIA a 0.5. 
    # Así "atrapa" a todos los estados del manifold aunque tengan el pequeño Delta,
    # pero ignora a los estados k=2 que están a +3.5 de distancia.
    if np.abs(H[i, i] - E_1st_exc_base) < 0.5:
        A_down[0, i] = 1.0 

A_up = A_down.T.conj()

def dissipator(L, rho):
    L_dag = L.T.conj()
    return L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)

# =============================================================================
# --- 3. ECUACIÓN MAESTRA DE LINDBLAD ---
# =============================================================================
def lindblad_rhs(t, rho_flat):
    rho = rho_flat.reshape((dim, dim))
    
    dot_rho = -1j * (H @ rho - rho @ H)
    dot_rho += Gamma_down * dissipator(A_down, rho)
    dot_rho += Gamma_up * dissipator(A_up, rho)
    
    return dot_rho.flatten()

# =============================================================================
# --- 4. SIMULACIÓN DINÁMICA ---
# =============================================================================
# Estado inicial: Termómetro frío (GS puro)
rho0 = np.zeros((dim, dim), dtype=complex)
rho0[0, 0] = 1.0 

t_eval = np.logspace(-1, 6, 100)

print(f"Resolviendo Matriz Densidad ({dim}x{dim}) para N={N_total}...")
sol = solve_ivp(lindblad_rhs, [t_eval[0], t_eval[-1]], rho0.flatten(), 
                t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)

rho_t = sol.y.reshape((dim, dim, len(t_eval)))

# =============================================================================
# --- 5. GRÁFICA DE COHERENCIAS ---
# =============================================================================
plt.figure(figsize=(10, 6))

# Índices (0-based): GS=0. Excitados k=1 son 1, 2 (y 3 para N=4).
if N_total == 3:
    coh_23 = np.abs(rho_t[1, 2, :])
    coh_32 = np.abs(rho_t[2, 1, :])
    
    plt.plot(t_eval, coh_23, label=r'$|\sigma_{23}|$ (Coherencia en manifold $k=1$)', color='blue', lw=2.5)
    plt.plot(t_eval, coh_32, label=r'$|\sigma_{32}|$ (Hermítico conjugado)', color='cyan', linestyle='--', lw=2.5)

elif N_total == 4:
    coh_32 = np.abs(rho_t[2, 1, :])
    coh_34 = np.abs(rho_t[2, 3, :])
    
    plt.plot(t_eval, coh_32, label=r'$|\sigma_{32}|$ (Coherencia entre estados $k=1$)', color='blue', lw=2.5)
    plt.plot(t_eval, coh_34, label=r'$|\sigma_{34}|$ (Coherencia entre estados $k=1$)', color='purple', linestyle='--', lw=2.5)

plt.xscale('log')
plt.xlabel('Tiempo $t$ (Escala Logarítmica)', fontsize=14)
plt.ylabel('Magnitud de Coherencia Inducida', fontsize=14)
plt.title(f'Coherencias Inducidas por el Baño en el Star Model Exacto (N={N_total})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.4, which='both')
plt.tight_layout()
plt.show()