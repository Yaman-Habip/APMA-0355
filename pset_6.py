import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt('/Users/yamanhabip/Downloads/Cooling_Data.txt')
x = data[:, 0]
y = data[:, 1]

# Linear regression
m, b = np.polyfit(x, y, 1)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='blue', alpha=0.6)
plt.plot(x, m*x + b, 'r-', label=f'Trendline: y = {m:.4f}x + {b:.4f}')

plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Cooling Data with Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
