import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def fish_population_model(n, a, K, f):
    """
    Fish population differential equation: dn/dt = an(1 - n/K) - f

    Args:
        n: current population
        a: growth rate parameter
        K: carrying capacity
        f: fishing yield (constant)

    Returns:
        derivative dn/dt
    """
    return a * n * (1 - n / K) - f


def euler_method(f, n0, t_span, h, *args):
    """
    Solve ODE using Euler's method

    Args:
        f: function representing dy/dt = f(y, *args)
        n0: initial condition
        t_span: tuple (t_start, t_end)
        h: step size
        *args: additional parameters for function f

    Returns:
        t: array of time points
        n: array of solution values
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + h, h)
    n = np.zeros_like(t)
    n[0] = n0

    for i in range(1, len(t)):
        n_new = n[i - 1] + h * f(n[i - 1], *args)
        # Prevent negative populations and extreme values
        n[i] = max(0, min(n_new, 10000))

    return t, n


# Parameter variations
a_values = [0.1, 0.3, 0.5]
K_values = [100, 200]
f_values = [5, 15, 25]
initial_conditions = [10, 50, 150]

# Simulation parameters
t_span = (0, 50)
h = 0.1

# Create subplots for better visualization (3x6 grid for 18 combinations)
fig, axes = plt.subplots(3, 6, figsize=(24, 12))
axes = axes.flatten()

plot_idx = 0
colors = ['blue', 'red', 'green', 'orange']

# Create separate plots for each (a, K, f) combination
for a in a_values:
    for K in K_values:
        for f in f_values:
            ax = axes[plot_idx]
            
            # Plot different initial conditions on same subplot
            for i, n0 in enumerate(initial_conditions):
                # Solve using Euler's method
                t, n = euler_method(fish_population_model, n0, t_span, h, a, K, f)
                
                # Plot with different color for each initial condition
                ax.plot(t, n, color=colors[i], label=f"n₀={n0}", linewidth=2)
            
            # Customize each subplot
            ax.set_xlabel("Time")
            ax.set_ylabel("Population n(t)")
            ax.set_title(f"a={a}, K={K}, f={f}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add equilibrium lines for reference
            ax.axhline(y=K, color='black', linestyle='--', alpha=0.5, label=f'K={K}')
            
            plot_idx += 1

# Overall title
fig.suptitle("Fish Population Model: ṅ = an(1 - n/K) - f\nSolutions using Euler's Method", fontsize=16)
plt.tight_layout()
plt.show()

# Also create a focused comparison plot
plt.figure(figsize=(12, 8))
for i, f in enumerate([5, 15, 25]):
    for j, n0 in enumerate([10, 50, 150]):
        # Fixed parameters for comparison
        a, K = 0.3, 100
        t, n = euler_method(fish_population_model, n0, t_span, h, a, K, f)
        
        # Use different line styles and colors
        linestyle = ['-', '--', ':'][i]
        color = ['blue', 'red', 'green'][j]
        plt.plot(t, n, linestyle=linestyle, color=color, linewidth=2, 
                label=f"f={f}, n₀={n0}", alpha=0.8)

plt.xlabel("Time")
plt.ylabel("Population n(t)")
plt.title(f"Comparison Plot: a=0.3, K=100\nEffect of fishing yield (f) and initial conditions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Print summary of parameters used
print("Parameter Analysis Summary:")
print(f"Growth rates (a): {a_values}")
print(f"Carrying capacities (K): {K_values}")
print(f"Fishing yields (f): {f_values}")
print(f"Initial conditions n(0): {initial_conditions}")
print(f"Step size: {h}")
print(f"Time span: {t_span}")
print(
    f"Total solution curves: {len(a_values) * len(K_values) * len(f_values) * len(initial_conditions)}"
)
