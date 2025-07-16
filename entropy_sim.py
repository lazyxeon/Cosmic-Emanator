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
