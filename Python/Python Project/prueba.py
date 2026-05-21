import numpy as np
from scipy.optimize import root

def system_of_equations(vars):
    t, B = vars
    
    # Guard against invalid domains (t must be greater than 0)
    if t <= 0:
        return [1e6, 1e6]
        
    # Compute exponential terms safely
    exp_neg_t = np.exp(-t)
    
    # Avoid overflow issues with large 2B values by clipping if necessary
    clamped_2B = np.clip(2 * B, -50, 50)
    exp_2B = np.exp(clamped_2B)
    
    # Core function pieces definitions
    gamma = 1.0 / (1.0 + exp_2B)
    u = 0.5 * exp_neg_t + gamma * (1.0 - exp_neg_t)
    
    # Guard against division by zero in denominators
    if u <= 0 or u >= 1 or exp_neg_t >= 1:
        return [1e6, 1e6]
        
    # Shared terms to keep code clean
    u_denominator = u * (1.0 - u)
    u_numerator = 1.0 - 2.0 * u
    
    # Equation 1: Partial derivative of g with respect to t = 0
    eq1 = ((2.0 * exp_neg_t) / (1.0 - exp_neg_t)) - (1.0 / t) - (exp_neg_t * (gamma - 0.5) * (u_numerator / u_denominator))
    
    # Equation 2: Total derivative with respect to B = 0
    eq2 = 1.0 - 2.0 * B * np.tanh(B) + (B * gamma * (1.0 - gamma) * (1.0 - exp_neg_t) * (u_numerator / u_denominator))
    
    return [eq1, eq2]

# --- Execution ---

# Initial guess for [t, B]. 
# You can tweak these values if you suspect the physical peak lies elsewhere.
initial_guess = [1.0, 0.5]

# Using hybrid solver (default method for root)
result = root(system_of_equations, initial_guess)

if result.success:
    t_optimal, B_optimal = result.x
    print("=" * 40)
    print(" OPTIMIZATION SUCCESSFUL")
    print("=" * 40)
    print(f"Optimal t (Peak location) : {t_optimal:.6f}")
    print(f"Optimal B (Max amplifier) : {B_optimal:.6f}")
    print("-" * 40)
    
    # Verify by calculating the maximum eta value achieved
    gamma_opt = 1.0 / (1.0 + np.exp(2 * B_optimal))
    gamma_prime_opt = (2 * B_optimal * np.exp(2 * B_optimal)) / ((1.0 + np.exp(2 * B_optimal)) ** 2)
    u_opt = 0.5 * np.exp(-t_optimal) + gamma_opt * (1.0 - np.exp(-t_optimal))
    
    g_opt = ((1.0 - np.exp(-t_optimal)) ** 2) / (t_optimal * u_opt * (1.0 - u_opt))
    max_eta = (gamma_prime_opt ** 2) * g_opt
    print(f"Maximum possible eta(B,t) : {max_eta:.6f}")
    print("=" * 40)
else:
    print("Solver failed to converge:", result.message)