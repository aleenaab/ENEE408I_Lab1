import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(0, 2 * np.pi, 1000)
plt.figure()
plt.plot(x_vals, np.sin(x_vals))
plt.xlabel("radians")
plt.ylabel("sin(x)")
plt.title("Sine Function")


X, Y = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
Z = np.sin(np.sqrt(X**2 + Y**2))
graph = plt.figure().add_subplot(111, projection="3d")
graph.plot_surface(X, Y, Z, cmap='plasma')
graph.set_xlabel("x")
graph.set_ylabel("y")
graph.set_zlabel("z")
graph.set_title("3d Graph")
plt.show()