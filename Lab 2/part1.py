import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, x = wavfile.read("human_voice.wav")
if x.ndim > 1:
    x = x[:, 0]
x = x.astype(np.float64)

print("Original sampling frequency:", fs, "Hz")
print("Original number of samples:", len(x))

t = np.arange(len(x)) / fs
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.title(f"Original Signal ({fs} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

target_fs = 8000
ratio = fs / target_fs
n_out = int(len(x) / ratio)
y = np.zeros(n_out)

for i in range(n_out):
    pos = i * ratio
    k = int(pos)
    frac = pos - k
    if k + 1 < len(x):
        y[i] = x[k] * (1 - frac) + x[k + 1] * frac
    else:
        y[i] = x[k]

print("Downsampled sampling frequency:", target_fs, "Hz")
print("Number of samples after downsampling:", len(y))

wavfile.write("human_voice_8k.wav", target_fs, y.astype(np.int16))

t_down = np.arange(len(y)) / target_fs
plt.figure(figsize=(10, 4))
plt.plot(t_down, y, color="tab:orange")
plt.title(f"Downsampled Signal ({target_fs} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
