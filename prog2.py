from pydub import silence, AudioSegment import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import time
wavFile1 = "source1.wav"
mp3File1 = "heavy.mp3"
flacFile1 = "heavy.flac"

wavFile2 = "sourcesmall.wav" 
mp3File2 = "light.mp3" 
flacFile2 = "light.flac"

audio = AudioSegment.from_file(wavFile1)
a = time.time()
audio.export(flacFile1, format="flac")
b = time.time()
print("time from wav to flac: ", (b-a))

audio = AudioSegment.from_file(wavFile1) 
a = time.time()
audio.export(mp3File1, format="mp3")
b = time.time()
print("time from wav to mp3: ", (b-a))

audio = AudioSegment.from_file(wavFile2) 
a = time.time()
audio.export(flacFile2, format="flac")
b = time.time()
print("Light file, time from wav to flac: ", (b-a))

audio = AudioSegment.from_file(wavFile2)
a = time.time()
audio.export(mp3File2, format="mp3")
b = time.time()
print("Light file, time from wav to mp3: ", (b-a))

print("Example of lossy compression algorithm:")
data, samplerate = sf.read(wavFile2)
print("Sample rate : {} Hz".format(samplerate)) 
n = len(data)
Fs = samplerate
ch1 = np.array([data[i][0] for i in range(n)]) 
ch2 = np.array([data[i][1] for i in range(n)])
ch1_Fourier = np.fft.fft(ch1)
abs_ch1_Fourier = np.absolute(ch1_Fourier[:n//2])
plt.plot(np.linspace(0, Fs / 2, n//2), abs_ch1_Fourier) 
plt.ylabel('Spectrum')
plt.xlabel('$f$ (Hz)')
plt.savefig("image1.png")
eps = 1e-5
frequenciesToRemove = (1 - eps) * np.sum(abs_ch1_Fourier) < np.cumsum(abs_ch1_Fourier)
f0 = (len(frequenciesToRemove) - np.sum(frequenciesToRemove) )* (Fs / 2) / (n / 2)
print("f0 : {} Hz".format(int(f0)))
plt.axvline(f0, color='r')
plt.plot(np.linspace(0, Fs / 2, n//2), abs_ch1_Fourier)
plt.ylabel('Spectrum')
plt.xlabel('$f$ (Hz)')
plt.savefig("image2.png")
wavCompressedFile = "result.wav" D = int(Fs / f0)
print("Downsampling factor : {}".format(D))
new_data = data[::D, :]
sf.write(wavCompressedFile, new_data, int(Fs / D), 'PCM_16')
