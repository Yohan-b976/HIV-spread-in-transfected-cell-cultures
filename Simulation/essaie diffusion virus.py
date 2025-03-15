# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:35:33 2024

@author: yohan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.fft import fft2, ifft2

# États de la cellule
VIDE = 0
NON_INFECTEE = 1
INFECTEE = 2
INFECTIEUSE = 3
MORTE = 4

# Paramètres du modèle
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 100  # Nombre moyen de particules virales produites par virion
s = 0.8  # Probabilité d'infection par contact
k = 2  # Nombre moyen de contacts par particule virale
Rmax = 100  # Valeur de coupure maximale pour le taux de réplication
mu = 0 # Probabilité de mort d'une cellule infectieuse
timesteps = 40  # Nombre d'étapes temporelles
D = 0.1  # Coefficient de diffusion
dt = 0.1  # Pas de temps
dx = 1.0  # Pas spatial

# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)

# Fonction de diffusion utilisant la FFT
def diffusion(virus_density, D, dt, dx):
    N = virus_density.shape[0]  # Taille de la grille
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2))  # Noyau de diffusion en espace de Fourier

    # FFT de la densité virale
    virus_density_fft = fft2(virus_density)
    # Convolution en espace de Fourier
    virus_density_fft_new = virus_density_fft * F_hat
    # Inverse FFT pour revenir à l'espace réel
    virus_density_new = np.real(ifft2(virus_density_fft_new))
    return virus_density_new

# Simulation du processus de Markov avec diffusion
def simulate_infection(grid_size, N_initial):
    state = np.full((grid_size, grid_size), 0)
    state[10,10] = 3
    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    replication_rate = np.zeros((grid_size, grid_size))

    # Définir les couleurs pour chaque état
    colors = ['mistyrose' ,'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)

    for t in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()

        # Simulation de l'infection et de la réplication virale
        for i in range(grid_size):
            for j in range(grid_size):
                if state[i, j] == NON_INFECTEE:
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                elif state[i, j] == INFECTEE:
                    compte_avant_infectiosité[i, j] += 1
                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE
                elif state[i, j] == INFECTIEUSE:
                    Z_i = np.random.poisson(virus_density[i, j] * k)
                    replication_rate[i, j] = min(Z_i * V, Rmax)
                    new_virus_density[i, j] += replication_rate[i, j]
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE

        # Diffusion de la densité virale
        for l in range (1):
            virus_density = diffusion(new_virus_density, D, dt, dx)
            plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
            plt.title('Densité virale')
            plt.colorbar()
            plt.show()

        state = new_state

       
        # Afficher l'évolution avec les couleurs fixes pour chaque état
        plt.imshow(state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])  # Positions des étiquettes
        cbar.set_ticklabels(['Vide','Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels personnalisés
        plt.pause(0.1)

    plt.show()
    return state, virus_density

# Paramètres d'entrée
grid_size = 20
N_initial = 1.5
# Simuler l'infection
final_state, final_virus_density = simulate_infection(grid_size, N_initial)

# Afficher la densité virale finale
plt.imshow(final_virus_density, cmap='inferno', interpolation='nearest')
plt.title('Densité virale finale')
plt.colorbar()
plt.show()
