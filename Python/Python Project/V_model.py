import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# --- 1. PARÁMETROS DEL PAPER (Reproducción de la Figura 1) ---
# =============================================================================
nu = 1.0                 # Energía \nu
Delta = 1e-4 * nu        # Pequeño gap entre los estados excitados (\Delta)
beta = 4.0 / nu          # Temperatura inversa (\beta \nu = 4)
gamma = 0.07             # Constante de acoplamiento para densidad espectral Óhmica

# Tasas de transición derivadas del baño térmico (Bosónico Óhmico)
# n_B(\nu) es la distribución de Bose-Einstein
n_B = 1.0 / (np.exp(beta * nu) - 1.0) 

# k es la tasa de relajación hacia el ground state
k = 2.0 * gamma * nu * (n_B + 1.0) 

# \phi es una tasa combinada conveniente
phi = k * (1.0 + 2.0 * np.exp(-beta * nu)) 

# =============================================================================
# --- 2. ECUACIONES DE MOVIMIENTO (UQME - Eq. 2 del paper) ---
# =============================================================================
def v_model_dynamics(t, y):
    """
    y = [p, sigma_R, sigma_I]
    Donde:
      p = 1/2 * (\sigma_{22} + \sigma_{33})  (Población promedio excitada)
      sigma_R = \sigma_{32}^R  (Parte real de la coherencia)
      sigma_I = \sigma_{32}^I  (Parte imaginaria de la coherencia)
    """
    p, sig_R, sig_I = y
    
    # Derivadas temporales según la Ecuación 2 del paper
    dp_dt     = -k * sig_R - phi * p + (phi - k) / 2.0
    dsig_R_dt = -k * sig_R - phi * p + Delta * sig_I + (phi - k) / 2.0
    dsig_I_dt = -k * sig_I - Delta * sig_R
    
    return [dp_dt, dsig_R_dt, dsig_I_dt]

# =============================================================================
# --- 3. SIMULACIÓN ---
# =============================================================================
# Condición Inicial: Termómetro preparado en el Ground State puro
# Por tanto, p(0) = 0, y las coherencias son nulas
y0 = [0.0, 0.0, 0.0]

# Malla de tiempo logarítmica para abarcar la enorme separación de escalas temporales
t_eval = np.logspace(-1, 7, 2000)

print("Integrando la dinámica UQME del V-Model...")
sol = solve_ivp(v_model_dynamics, [t_eval[0], t_eval[-1]], y0, 
                t_eval=t_eval, method='Radau', rtol=1e-10, atol=1e-10)

# Extracción de resultados
p_t = sol.y[0]
sig_R_t = sol.y[1]

# Reconstruimos los observables requeridos
# Como la traza de la matriz densidad siempre es 1: \sigma_{11} + \sigma_{22} + \sigma_{33} = 1
# Por tanto, \sigma_{11} = 1 - 2*p
pop_GS = 1.0 - 2.0 * p_t              
pop_ES_avg = p_t                      
coherence_mag = np.abs(sig_R_t)       

# =============================================================================
# --- 4. GRÁFICA ---
# =============================================================================
plt.figure(figsize=(11, 6))

plt.plot(t_eval, pop_GS, label=r'Ground State $\sigma_{11}$', color='black', lw=2.5, linestyle='-')
plt.plot(t_eval, pop_ES_avg, label=r'Avg Excited Pop $p = \frac{1}{2}(\sigma_{22}+\sigma_{33})$', color='red', lw=2.5, linestyle='--')
plt.plot(t_eval, coherence_mag, label=r'Coherence $|\sigma_{32}^R|$', color='blue', lw=2.5, linestyle='-.')

# Escala logarítmica para visualizar el estado pretermal
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6, 1.5)

plt.xlabel(r'Tiempo Adimensional ($\nu t$)', fontsize=14)
plt.ylabel('Poblaciones y Coherencias', fontsize=14)
plt.title(f'Dinámica del V-Model (Basado en Fig. 1 del paper)\nGeneración del Estado Pretermal', fontsize=16)

# Sombrear la zona del estado pretermal (estimación aproximada basada en el plot)
plt.axvspan(1e1, 1e5, color='green', alpha=0.1, label='Prethermal Plateau')

plt.legend(fontsize=12, loc='lower left')
plt.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()