import numpy as np
from scipy.io import wavfile

def rms (audio):
    return np.sqrt(np.mean(audio**2))

n1, audio_M1 = wavfile.read("M1.wav")
n2, audio_M2 = wavfile.read("M2.wav")
n3, audio_M3 = wavfile.read("M3.wav")

audio_M1 = audio_M1.astype(np.float32)
audio_M2 = audio_M2.astype(np.float32)
audio_M3 = audio_M3.astype(np.float32)

rms_M1 = rms (audio_M1)
rms_M2 = rms (audio_M2)
rms_M3 = rms (audio_M3)

print("RMS for M1:", rms_M1)
print("RMS for M2:", rms_M2)
print("RMS for M3:", rms_M3)

def correlation(x, y):
    N = min(len(x), len(y))
    x = x[:N] - np.mean(x[:N])
    y = y[:N] - np.mean(y[:N])
    lags = range(-(N - 1), N)
    Rxy = []
    for m in lags:
        correlation = 0.0
        for n in range(N):
            index_y = n - m
            if 0 <= index_y < N:
                correlation += x[n] * y[index_y]
        Rxy.append(correlation)
    return Rxy, list(lags)

Rxy, lags = correlation(audio_M1, audio_M2)
max_index = 0

for i in range(1, len(Rxy)):
    if Rxy[i] > Rxy[max_index]:
        max_index = i

time_delay = (lags[max_index]) / n1

print("Time delay:", time_delay, "seconds or", time_delay * 1000, "ms")

