# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:56:02 2024

@author: yohan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# États de la cellule
NON_INFECTEE = 0
INFECTEE = 1
INFECTIEUSE = 2
MORTE = 3

# Paramètres du modèle
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 1  # Nombre moyen de particules virales produites par virion
s = 0.001  # Probabilité d'infection par contact
k = 2  # Nombre moyen de contacts par particule virale
Rmax = 10  # Valeur de coupure maximale pour le taux de réplication
mu = 0.0001  # Probabilité de mort d'une cellule infectieuse
timesteps = 96  # Nombre d'étapes temporelles

# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)

# Simulation du processus de Markov
def simulate_infection(grid_size, N_initial):
    state = np.full((grid_size, grid_size), NON_INFECTEE)
    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    replication_rate = np.zeros((grid_size, grid_size))

    # Définir les couleurs pour chaque état
    colors = ['mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)

    for t in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()

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

        state = new_state
        virus_density = new_virus_density

        # Afficher l'évolution avec les couleurs fixes pour chaque état
        plt.imshow(state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5])  # Positions des étiquettes
        cbar.set_ticklabels(['Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels personnalisés
        plt.pause(0.1)

    plt.show()
    return state, virus_density

# Paramètres d'entrée
grid_size = 100
N_initial = 1

# Simuler l'infection
final_state, final_virus_density = simulate_infection(grid_size, N_initial)

# Afficher le résultat final avec les couleurs fixes et labels textuels
# plt.imshow(final_state, cmap=cmap, norm=norm, interpolation='nearest')
# plt.title('État final des cellules')
# cbar = plt.colorbar()
# cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
# cbar.set_ticklabels(['Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels textuels
# plt.show()

plt.imshow(final_virus_density, cmap='inferno', interpolation='nearest')
plt.title('Densité virale finale')
plt.colorbar()
plt.show()

