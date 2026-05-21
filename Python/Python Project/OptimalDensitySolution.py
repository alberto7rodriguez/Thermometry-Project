import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# =============================================================================
# --- 1. PARÁMETROS DEL PAPER (Fig. 3a) ---
# =============================================================================
nu = 1.0                 # Salto de energía (frecuencia base)
Delta = 1e-4 * nu        # Pequeño gap (cuasi-degeneración)
beta = 4.0 / nu          # Temperatura inversa (\beta \nu = 4)
gamma = 0.07             # Constante de acoplamiento al baño
dbeta = 1e-4             # Incremento para la derivada numérica

# Lista de número de niveles excitados a simular (N en el paper)
N_exc_list = [1, 2, 3, 4]

# =============================================================================
# --- 2. CONSTRUCCIÓN DEL SUPEROPERADOR DE LIOUVILLE ---
# =============================================================================
def dissipator_super(A):
    """Construye el superoperador disipativo D[A] en forma matricial plana"""
    dim = A.shape[0]
    I = np.eye(dim)
    A_dag = A.T.conj()
    
    # La regla matemática de vectorización: A * rho * B -> kron(B.T, A) * vec(rho)
    term1 = np.kron(A_dag.T, A)                    # A * rho * A_dag
    term2 = -0.5 * np.kron(I, A_dag @ A)           # -0.5 * A_dag * A * rho
    term3 = -0.5 * np.kron((A_dag @ A).T, I)       # -0.5 * rho * A_dag * A
    return term1 + term2 + term3

def get_liouvillian(N_exc, beta_val):
    """Genera la matriz del Superoperador de Lindblad para N_exc excitados"""
    dim = N_exc + 1
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        H[i, i] = nu + (i - 1) * Delta
        
    # Operador de salto COLECTIVO: Todos los excitados decaen al Ground State
    A_down = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        A_down[0, i] = 1.0  
    A_up = A_down.T.conj()
    
    # Tasas de transición (Modelo Bosónico Óhmico del paper)
    n_th = 1.0 / (np.exp(beta_val * nu) - 1.0)
    Gamma_down = 2.0 * gamma * nu * (n_th + 1.0)
    Gamma_up   = 2.0 * gamma * nu * n_th
    
    # Superoperador del Conmutador (Evolución Unitaria)
    I = np.eye(dim)
    L_H = -1j * np.kron(I, H) + 1j * np.kron(H.T, I)
    
    # Superoperador Disipativo
    L_diss = Gamma_down * dissipator_super(A_down) + Gamma_up * dissipator_super(A_up)
    
    return L_H + L_diss

# =============================================================================
# --- 3. CÁLCULO DE LA QUANTUM FISHER INFORMATION (QFI) ---
# =============================================================================
def calculate_qfi(rho, drho_dbeta):
    """Calcula la QFI exacta diagonalizando la matriz densidad"""
    evals, evecs = la.eigh(rho) # eigh es súper rápido porque rho es hermítica
    
    qfi = 0.0
    for i in range(len(evals)):
        for j in range(len(evals)):
            p_i = np.real(evals[i])
            p_j = np.real(evals[j])
            
            # Condición para evitar dividir por cero
            if p_i + p_j > 1e-15:
                v_i = evecs[:, i]
                v_j = evecs[:, j]
                # Elemento de matriz <psi_i | drho | psi_j>
                mat_el = np.vdot(v_i, drho_dbeta @ v_j)
                qfi += (2.0 / (p_i + p_j)) * np.abs(mat_el)**2
    return qfi

# =============================================================================
# --- 4. SIMULACIÓN DINÁMICA ---
# =============================================================================
# Malla temporal logarítmica (ampliada un poco para ver bien la subida final)
t_array = np.logspace(-1, 8, 300) 

# Colores y estilos idénticos a la Fig 3a del paper
colors = {1: '#F6C85F', 2: '#37AFA9', 3: '#455EAA', 4: '#2E2252'}
linestyles = {1: '--', 2: '-', 3: '-', 4: ':'} 

plt.figure(figsize=(7, 5))

for N_exc in N_exc_list:
    print(f"Simulando QFI para N={N_exc} excitados...")
    dim = N_exc + 1
    
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    rho0_flat = rho0.flatten()
    
    L_base = get_liouvillian(N_exc, beta)
    L_up   = get_liouvillian(N_exc, beta + dbeta)
    L_dn   = get_liouvillian(N_exc, beta - dbeta)
    
    F_t = np.zeros(len(t_array))
    
    # QFI de Equilibrio asintótica (Eq. 9)
    F_eq = N_exc * (nu**2) * np.exp(nu * beta) / (N_exc + np.exp(nu * beta))**2
    
    for idx, t in enumerate(t_array):
        rho_base_flat = la.expm(L_base * t) @ rho0_flat
        rho_up_flat   = la.expm(L_up * t) @ rho0_flat
        rho_dn_flat   = la.expm(L_dn * t) @ rho0_flat
        
        rho_base = rho_base_flat.reshape((dim, dim))
        rho_up   = rho_up_flat.reshape((dim, dim))
        rho_dn   = rho_dn_flat.reshape((dim, dim))
        
        drho_dbeta = (rho_up - rho_dn) / (2.0 * dbeta)
        rho_base = (rho_base + rho_base.T.conj()) / 2.0
        
        F_t[idx] = calculate_qfi(rho_base, drho_dbeta)
        
    # Trazamos la evolución temporal (Líneas gruesas)
    plt.plot(t_array, F_t / (nu**2), color=colors[N_exc], linestyle=linestyles[N_exc], 
             lw=3, label=f'$N={N_exc}$')
    
    # Trazamos las asíntotas de equilibrio (Líneas finas discontinuas)
    plt.axhline(F_eq / (nu**2), color=colors[N_exc], linestyle='--', lw=1.5, alpha=0.8)

# =============================================================================
# --- 5. FORMATO DE LA GRÁFICA (Clonación Visual de Fig 3a) ---
# =============================================================================
plt.xscale('log')
# ¡ELIMINADA LA ESCALA LOGARÍTMICA EN Y! Ahora es lineal.
plt.ylim(0, 0.065) 
plt.xlim(0.1, 10**8)

plt.xlabel(r'$\nu t$', fontsize=14)
plt.ylabel(r'$\mathcal{\tilde{F}}(\beta) \nu^{-2}$', fontsize=14)

# Posicionamos la leyenda exactamente igual que en el paper
plt.legend(fontsize=12, loc='lower right', ncol=2, frameon=False)
plt.tight_layout()
plt.show()