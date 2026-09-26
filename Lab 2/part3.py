import wave
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#1

folder = Path(__file__).resolve().parent
audio_path = folder / "Cafe_with_noise.wav"

with wave.open(str(audio_path), "rb") as wav_file:
    sample_rate = wav_file.getframerate()
    number_of_frames = wav_file.getnframes()
    number_of_channels = wav_file.getnchannels()
    audio_data = wav_file.readframes(number_of_frames)

audio = np.frombuffer(audio_data, dtype=np.int16)

if number_of_channels > 1:
    audio = audio.reshape(-1, number_of_channels)
    audio = audio.mean(axis=1)

audio = audio.astype(float)
audio = audio / np.max(np.abs(audio))

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 4))
plt.plot(time, audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Cafe_with_noise.wav - Audio Signal")
plt.grid()
plt.tight_layout()
plt.show()


#2

N = len(audio)

window = np.hanning(N)
windowed_audio = audio * window

fft_result = np.fft.rfft(windowed_audio)
frequencies = np.fft.rfftfreq(N, 1 / sample_rate)

magnitude = np.abs(fft_result)
magnitude = magnitude / np.max(magnitude)

plt.figure(figsize=(12, 5))
plt.plot(frequencies, magnitude)

plt.axvspan(
    85,
    4000,
    color="green",
    alpha=0.2,
    label="Human voice region"
)

plt.axvspan(
    1400,
    1600,
    color="red",
    alpha=0.3,
    label="Noise region (1200-1800 Hz)"
)

plt.xlim(0, 4000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency-Domain Analysis")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


#3

lower_cutoff = 1400
upper_cutoff = 1600

fft_result = np.fft.rfft(audio)
frequencies = np.fft.rfftfreq(N, 1 / sample_rate)

filtered_fft = np.copy(fft_result)

filtered_fft[
    (frequencies >= lower_cutoff) &
    (frequencies <= upper_cutoff)
] = 0

filtered_audio = np.fft.irfft(filtered_fft, n=N)

filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))

plt.figure(figsize=(12, 4))
plt.plot(time, filtered_audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Filtered Voice Signal")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, audio)
plt.title("Original Audio Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, filtered_audio)
plt.title("Filtered Voice Signal (1400-1600 Hz Removed)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

filtered_fft_check = np.fft.rfft(filtered_audio)
filtered_magnitude = np.abs(filtered_fft_check)
filtered_magnitude = filtered_magnitude / np.max(filtered_magnitude)

plt.figure(figsize=(12, 5))
plt.plot(frequencies, magnitude, label="Original")
plt.plot(frequencies, filtered_magnitude, label="Filtered")

plt.axvspan(
    1400,
    1600,
    color="red",
    alpha=0.2,
    label="Removed region"
)

plt.xlim(0, 4000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Original vs Filtered Frequency Spectrum")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

output_path = folder / "Cafe_filtered.wav"

filtered_audio_int16 = np.int16(
    filtered_audio / np.max(np.abs(filtered_audio)) * 32767
)

with wave.open(str(output_path), "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(filtered_audio_int16.tobytes())

print("Filtered audio saved to:", output_path)
