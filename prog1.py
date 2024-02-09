import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt import sounddevice as sd
plt.rcParams['figure.figsize'] = [12,8] 
plt.rcParams.update({'font.size': 18})

dt = 0.001
t = np.arange(0,2,dt)
print(t)
f0 = 60 f1 = 300 tl = 2
f = np.sin(2*np.pi*t*(f0 + (f1-f0)*np.power(t,2)/(3*tl**2)))
fs = 1/dt sd.play(2*f, fs)
plt.specgram(f, NFFT=128, Fs=1/dt, noverlap=120, cmap='jet_r') 
plt.colorbar()
plt.show()
n = len(t)
fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(2, np.floor(n/2),dtype='int')
fig,axs = plt.subplots(2,1)
plt.sca(axs[0]) plt.plot(t,f,color='c',label='signal form') 
plt.xlim(t[0],t[-1])
plt.legend()
plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c',label='spectrum') 
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()
plt.show()
