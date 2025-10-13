# %%
# conda activate test_env
# python lab1.py

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("./lab1_plots", exist_ok=True)

# %% [markdown]
# Ex1

# %%
os.makedirs("lab1_plots/ex1", exist_ok=True)
t = np.linspace(0, 0.03, int(0.03*1/0.0005)) # SAU t = np.arange(0, 0.0300, 0.0005)
print(*[x for x in t])

# %%
x = np.cos(520*np.pi*t+np.pi/3)
print(x)

# %%
y = np.cos(280*np.pi*t-np.pi/3)
z = np.cos(120*np.pi*t+np.pi/3)

# %%

fig, axs = plt.subplots(3, figsize=(20, 12))
L = [x, y, z]
for ax, i in zip(axs, L):
    ax.plot(t, i)
    #ax.stem(t, i)
    ax.set_xlabel('t')
    ax.grid(True)
axs[0].set_ylabel('x(t)')
axs[1].set_ylabel('y(t)')
axs[2].set_ylabel('z(t)')
plt.savefig(fname="./lab1_plots/ex1/semnale_xyz.pdf", format="pdf")
plt.show()

# %%
durata = 0.03 # aleg 2 secunde SAU 0.03 secunde (la a), dar la 0.03 secunde am avea numai 6 esantioane, insuficient pt a observa semnalul
frecv = 200
n_esantioane = durata*frecv # 200 Hz pe secunda, cu 2 secunde
t_n = np.linspace(0, durata, int(n_esantioane))
print(n_esantioane)
print(*[x for x in t_n], end="\n\n")

T = 1/frecv # perioada de esantionare
dif = np.diff(t_n) # diferente intre puncte t_n consecutive
print("T = ", T, "; Diferente individuale:", dif) # toate diferentele ar trebui sa fie egale cu T sau foarte apropiate

# %%
x_n = np.cos(520*np.pi*t_n+np.pi/3)
y_n = np.cos(280*np.pi*t_n-np.pi/3)
z_n = np.cos(120*np.pi*t_n+np.pi/3)

# %%
fig, axs = plt.subplots(3, figsize=(20, 12))

L = [x_n, y_n, z_n]
for ax, i in zip(axs, L):
    ax.plot(t_n, i)
    ax.stem(t_n, i)
    ax.set_xlabel('t_n')
    ax.grid(True)
axs[0].set_ylabel('x(t_n)')
axs[1].set_ylabel('y(t_n)')
axs[2].set_ylabel('z(t_n)')
plt.savefig(fname="./lab1_plots/ex1/semnale_esantionate_xyz_frecventa_200_secunde_2.pdf", format="pdf")
plt.show()

# %% [markdown]
# Ex2

# %%
os.makedirs("lab1_plots/ex2", exist_ok=True)

# general: t = np.linspace(0, durata, n_esantioane)
# ampl * np.sin(2 * np.pi * frecv * t + faza_phi)

n_esantioane = 1600
frecv = 400
durata = n_esantioane/frecv # 1600 esantioane la 400 Hz => 4 secunde
t = np.linspace(0, durata, int(n_esantioane))
w = np.sin(2*np.pi*frecv*t)
plt.plot(t, w)
plt.xlabel('t')
plt.ylabel('w(t)')
#plt.xlim([0, 0.5]) # se vad bine reprezentarile esantioanelor la val mult mai mici decat 4
plt.grid(True)
plt.savefig(fname="./lab1_plots/ex2/semnal_sinusoidal_frecventa_400_esantioane_1600.pdf", format="pdf")
plt.show()

# %%
frecv = 800
durata = 3
n_esantioane = durata*frecv
t = np.linspace(0, durata, int(n_esantioane))
u = 2*np.sin(2*np.pi*frecv*t + np.pi/4)
plt.plot(t, u)
plt.xlabel('t')
plt.ylabel('u(t)')
plt.grid(True)
plt.savefig(fname="./lab1_plots/ex2/semnal_sinusoidal_frecventa_800_durata_3.pdf", format="pdf")
plt.show()


# %%
#from scipy import signal

frecv = 240
t = np.linspace(0, 0.04, 1000) # shh n-am pus n_esantioane shh
#u = signal.sawtooth(2 * np.pi * 5 * t)
ampl = 1
v = ampl*(frecv*t-np.floor(frecv*t))
plt.plot(t, v)
plt.xlabel('t')
plt.ylabel('v(t)')
#plt.xlim([0, 0.0042])
plt.grid(True)
plt.savefig(fname="./lab1_plots/ex2/semnal_sawtooth_frecventa_240.pdf", format="pdf")
plt.show()

# %%
frecv = 300
t = np.linspace(0, 12, 190) # shh n-am pus n_esantioane shh
ampl = 1
m = ampl*np.sign(np.sin(2*np.pi*frecv*t))

plt.figure(figsize=(10, 5))
plt.plot(t, m)
plt.xlabel('t')
plt.ylabel('m(t)')
plt.grid(True)
plt.savefig(fname="./lab1_plots/ex2/semnal_square_frecventa_300.pdf", format="pdf")
plt.show()

# %%
n = np.random.rand(128, 128)

plt.imshow(n)
plt.savefig(fname="./lab1_plots/ex2/zgomot_random.pdf", format="pdf")
plt.show()

# %%
m = np.zeros((128, 128))

for i in range(128):
    for j in range(128):
        if j%2:
            m[i][j] = np.arctan(np.pi*(j/128))
        elif i%2:
            m[i][j] = np.sin(np.pi*(i/128))
            

plt.imshow(m)
plt.savefig(fname="./lab1_plots/ex2/zgomot1_.pdf", format="pdf")
plt.show()

# %% [markdown]
# Ex3
# 
# a) T = 1/frecv = 1/2000 = 0.0005s
# 
# b) 2000 esantione/ sec (definitie frecv) => 2000x3600 = 7200000 esantioane/ ora
# 
# 1 esantion/ 4 biti => 4x7200000 = 28800000 biti/ ora = 28800000/8 = 3600000 bytes/ ora


