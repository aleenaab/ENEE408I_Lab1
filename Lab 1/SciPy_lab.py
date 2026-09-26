import numpy as np
from scipy import linalg
from scipy import optimize
from scipy import fft
import matplotlib.pyplot as plt

A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

solution = linalg.solve(A, b)

print("Linear system:")
print("x =", round(solution[0], 2))
print("y =", round(solution[1], 2))

def function(x):
    return x**2 + 2*x

result = optimize.minimize_scalar(function)

print("\nMinimum of y = x^2 + 2x:")
print("x =", round(result.x, 2))
print("y =", round(result.fun, 2))

x = np.linspace(0, 1, 1000, endpoint=False)

f = np.sin(100 * np.pi * x) + 0.5 * np.sin(160 * np.pi * x)

F = fft.fft(f)
frequencies = fft.fftfreq(len(x), x[1] - x[0])

positive = frequencies >= 0

plt.plot(frequencies[positive], np.abs(F[positive]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform")
plt.grid()
plt.show()