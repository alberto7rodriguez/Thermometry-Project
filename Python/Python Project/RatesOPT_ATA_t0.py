import numpy as np
from scipy.optimize import minimize_scalar

def optimize_all_to_all_b(T, alpha, g=1.0):
    """
    Optimizes the all-to-all coupling parameter 'b' to maximize 
    the Fisher Information rate for a given temperature T and Ohmicity alpha.
    """
    
    def objective(b):
        # Avoid division by zero or overflow at b=0
        if b <= 1e-10:
            return 0.0
            
        try:
            exp_bT = np.exp(b / T)
            
            # The exact mathematical ratio: (\dot{\gamma}_{-b})^2 / \gamma_{-b}
            numerator = g * (b**(alpha + 2)) * exp_bT
            denominator = (T**4) * ((exp_bT - 1)**3)
            
            fisher_rate = numerator / denominator
            
            # Scipy minimizes by default, so we return the negative to maximize
            return -fisher_rate 
            
        except OverflowError:
            # Handle math overflows for very large b
            return 0.0

    # Perform a bounded scalar optimization 
    # Searching in a physically reasonable range (e.g., slightly above 0 to 20*T)
    res = minimize_scalar(objective, bounds=(1e-5, 20 * T), method='bounded')
    
    if res.success:
        optimal_b = res.x
        max_fisher_rate = -res.fun
        return optimal_b, max_fisher_rate
    else:
        raise ValueError("Optimization failed to converge.")

# --- System Parameters ---
T = 1.0
alpha =  1 #< We need to define this!
g = 1.0      #< Set to 1.0 by default, only scales the final value, doesn't change optimal b

optimal_b, max_info = optimize_all_to_all_b(T, alpha, g)

print(f"--- Optimization Results ---")
print(f"Temperature (T) : {T}")
print(f"Ohmicity (alpha): {alpha}")
print(f"Optimal coupling (b) : {optimal_b:.20f}")
print(f"Maximized Rate Factor: {max_info:.4f}")