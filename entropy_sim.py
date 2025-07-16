# examples/simulations/entropy_sim.py

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.1, 20, 100)
phi = 1.618
b = np.log(phi) / (np.pi / 2)
delta_s = b * np.log(t) + np.sin(2 * np.pi * t / 11.3)

plt.plot(t, delta_s)
plt.title("Entropy Pulse Simulation (ΔS ~ b ln t + sin(2π t / 11.3))")
plt.xlabel("t")
plt.ylabel("ΔS")
plt.grid(True)
plt.show()

# examples/simulations/entropy_sim.py

import numpy as np
import matplotlib.pyplot as plt

# Parameters from cosmic inspiration (golden ratio phi, period 11.3)
phi = 1.618  # Fibonacci-related
b = np.log(phi) / (np.pi / 2)
period = 11.3  # Black hole analog period

# Time array
t = np.linspace(0.1, 20, 100)

# Entropy delta simulation
delta_s = b * np.log(t) + np.sin(2 * np.pi * t / period)

# Plot
plt.plot(t, delta_s, label='ΔS ~ b ln t + sin(2π t / 11.3)')
plt.title("Entropy Pulse Simulation (Black Hole Analog)")
plt.xlabel("Time t")
plt.ylabel("ΔS (Entropy Change)")
plt.grid(True)
plt.legend()
plt.savefig('entropy_simulation.png')  # Save for repo/docs
plt.show()

# Optional: Simulate with variation (e.g., different periods)
for p in [5.0, 11.3, 20.0]:
    delta_s_var = b * np.log(t) + np.sin(2 * np.pi * t / p)
    print(f"Avg ΔS for period {p}: {np.mean(delta_s_var):.2f}")

print("This simulation mirrors TFNPLayer's entropy handling in noisy data processing.")