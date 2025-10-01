import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, n0, t_start, t_end, h):
    """
    Abstract Euler's method implementation

    Parameters:
    f: function representing dn/dt = f(t, n)
    n0: initial condition
    t_start: start time
    t_end: end time
    h: step size

    Returns:
    t_values: array of time values
    n_values: array of population values
    """
    t_values = np.arange(t_start, t_end + h, h)
    n_values = np.zeros(len(t_values))
    n_values[0] = n0

    for i in range(1, len(t_values)):
        n_values[i] = n_values[i - 1] + h * f(t_values[i - 1], n_values[i - 1])

    return t_values, n_values


def rabbit_population_derivative(t, n):
    """dn/dt = -0.1n + 20"""
    return -0.1 * n + 20


# Plot 1: Impact of step size on accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
step_sizes = [0.5, 1.0, 2.0, 5.0, 20]
colors = ["blue", "red", "green", "orange", "purple"]
t_end = 50


for i, h in enumerate(step_sizes):
    t_vals, n_vals = euler_method(rabbit_population_derivative, 251, 0, t_end, h)
    plt.plot(t_vals, n_vals, colors[i], marker="o", markersize=3, label=f"h = {h}")

plt.xlabel("Time (months)")
plt.ylabel("Population")
plt.title("Impact of Step Size on Euler's Method Accuracy")
plt.legend()
plt.grid(True)

# Plot 2: Single series with small step size and equilibrium line
plt.subplot(1, 2, 2)
h_small = 0.1
t_end = 50

# Single series with n(0) = 251
t_vals, n_vals = euler_method(rabbit_population_derivative, 251, 0, t_end, h_small)
plt.plot(t_vals, n_vals, "blue", linewidth=2, label="n(0) = 251")

# Equilibrium line at n = 200
plt.axhline(
    y=200, color="black", linestyle="--", linewidth=2, label="Equilibrium (n = 200)"
)

# Shaded 1% interval around equilibrium
plt.axhspan(198, 202, alpha=0.2, color="gray", label="±1% of equilibrium")

plt.xlabel("Time (months)")
plt.ylabel("Population")
plt.title("Rabbit Population with Small Step Size (h = 0.1)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Calculate when population reaches within 1% of limiting value (200)
def find_time_within_percent(n0, target, percent=0.01, h=0.1, max_time=100):
    """Find when population is within given percent of target value"""
    tolerance = target * percent
    t_vals, n_vals = euler_method(rabbit_population_derivative, n0, 0, max_time, h)

    for i, n in enumerate(n_vals):
        if abs(n - target) <= tolerance:
            return t_vals[i]
    return None


# For n(0) = 251
time_within_1_percent = find_time_within_percent(251, 200, 0.01)
print(f"Starting from n(0) = 251:")
print(
    f"Population reaches within 1% of limiting value (200) after {time_within_1_percent:.1f} months"
)
