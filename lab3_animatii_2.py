import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, Math, HTML
from matplotlib.animation import FuncAnimation

frecv = 20
rata_esantionare = 1000
durata = 0.2
T = 1/rata_esantionare # perioada de esantionare
N = durata*rata_esantionare # nr_esantioane din exercitiile din lab2
n = np.arange(0, int(N)) # n = 0, 1, ... N-1

tn = n*T # momentele discrete de timp
x_n = 0.7*np.sin(2*np.pi*frecv*tn+np.pi/3)

m = np.arange(0, int(N)) # m = 0, 1, ... N-1
omega0 = (2*np.pi)/(N*T) # frecv de infasurare
omegam = m*omega0
#print(omega0, omegam, sep="\n")
M = 1 # frecv discreta fixata pentru a conecta puncte discrete consecutive intre ele din cauza faptului ca X_discret are dimensiunea NxN
sm_n = np.exp(-1j*omegam[M]*tn)

# ceva f f important, practic in laborator scrie o formula foarte simpla in care nici nu e specificat omega tocmai pt ca omegam[1] == omega0

# %%
# b)

selectie_omegam = np.random.choice(omegam, size=6, replace=False)
# %%
# pt animatii, ele sunt numai in fisierul python, nu merg in jupyternotebook
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
X_selectie_omegam = []
for omega in selectie_omegam:
    sm_n = np.exp(-1j*omega*tn)
    X_selectie_omegam.append(x_n*sm_n)

plt.ion()

for i in range(len(X_selectie_omegam[0])):
    for ax_index, ax in enumerate(axes):
        X_discret = X_selectie_omegam[ax_index]

        ax.set_xlim(X_discret.real.min() * 1.1, X_discret.real.max() * 1.1)
        ax.set_ylim(X_discret.imag.min() * 1.1, X_discret.imag.max() * 1.1)
        ax.set_xlabel("real")
        ax.set_ylabel("imaginar")
        ax.grid(True)
        ax.set_title(f"omega={selectie_omegam[ax_index]:.2f}")
        if i > 0:
            ax.plot(X_discret.real[:i+1], X_discret.imag[:i+1], color='blue')
        ax.scatter(X_discret.real[:i+1], X_discret.imag[:i+1], c=np.abs(X_discret[:i+1]), cmap='viridis')

    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
