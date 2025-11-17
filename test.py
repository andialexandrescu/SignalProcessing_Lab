# %% [markdown]
# Ex1
# 
# a) numarul de masini e masurat din ora in ora, deci rata de esantionare inseamna o esantionare pe ora, adica fs=1/3600
# 
# b) 18288 esantioane, un esantion pe ora => 18288 de ore, adica 18288/24=762 zile
# 
# c) Nyquist: fs>=2*frecv_max, iar fs=1/3600 => frecv_max ... 1/(3600**2), 1/7200

# %%
# d)
import numpy as np
import matplotlib.pyplot as plt

x = np.genfromtxt('Train.csv', delimiter=',')
N = len(x)
fs = 1/3600
X = np.fft.fft(x)
X = np.abs(X/N)
X = X[:N//2] # simetrie
f = fs*np.linspace(0, N/2, N//2)/N

plt.figure(figsize=(12, 6))
plt.plot(f, X)
plt.xlabel('frecv')
plt.ylabel('magnitude')
plt.title('fourier trafic')
plt.grid(True)
plt.show()


