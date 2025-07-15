#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   ackley_func_plot.py
# Time    :   2025/06/16 11:25:23
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Ackley function with input (x, y, z)
def ackley_3d(x, y, z, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2 + z**2) / 3))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y) + np.cos(c * z)) / 3)
    return term1 + term2 + a + np.exp(1)

# Prepare figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Meshgrid for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Initial surface (z = -5)
Z = ackley_3d(X, Y, -5)
surf = [ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')]

# Set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y, z)')
ax.set_zlim(0, 20)
ax.set_title('Ackley Function with Varying z')

# Animation update function
def update(frame):
    # Remove the previous surface
    surf[0].remove()
    Z = ackley_3d(X, Y, frame)
    surf[0] = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Ackley Function Slice at z = {frame:.2f}")
    return (surf[0],)

# Create animation
z_vals = np.linspace(-32.768, 32.768, 180)
ani = animation.FuncAnimation(fig, update, frames=z_vals, interval=100)

# Save as GIF
gif_path = "Multi-Objective Optimisation/Benchmark/Package Module-III/Ackley_Function/ackley_3d_slice.gif"
ani.save(gif_path, writer='pillow', fps=10, dpi=120)

print(f"Animation saved as {gif_path}")

