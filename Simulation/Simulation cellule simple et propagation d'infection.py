# -*- coding: utf-8 -*-

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
timesteps = 100  # Nombre d'étapes temporelles

# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    # Calcul du taux d'infection λ_i
    lambda_i = s * N_i * k
    # Probabilité d'infection (au moins un succès dans une distribution de Poisson)
    return 1 - np.exp(-lambda_i)

# Simulation du processus de Markov
def simulate_infection(grid_size, N_initial):
    # Initialiser l'état des cellules (grille de taille grid_size x grid_size)
    state = np.full((grid_size, grid_size), NON_INFECTEE)
    # Carte de densité virale
    virus_density = np.full((grid_size, grid_size), N_initial)
    # Taux de réplication
    replication_rate = np.zeros((grid_size, grid_size))
    
    # Suivre la progression dans le temps
    for t in range(timesteps):
        new_state = state.copy()  # Pour stocker les nouveaux états
        new_virus_density = virus_density.copy()
        
        for i in range(grid_size):
            for j in range(grid_size):
                if state[i, j] == NON_INFECTEE:
                    # Calculer la probabilité d'infection pour la cellule non infectée
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE  # Cellule devient infectée

                elif state[i, j] == INFECTEE:
                    # Compte à rebours vers l'état infectieux
                    if t >= m:
                        new_state[i, j] = INFECTIEUSE  # Cellule devient infectieuse

                elif state[i, j] == INFECTIEUSE:
                    # Cellule infectieuse produit des virus
                    Z_i = np.random.poisson(virus_density[i, j] * k)
                    replication_rate[i, j] = min(Z_i * V, Rmax)
                    # Libération des virions dans l'environnement local
                    new_virus_density[i, j] += replication_rate[i, j]

                    # Mort possible de la cellule infectieuse
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE  # Cellule devient morte

                # Une cellule morte reste morte
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE

        # Mettre à jour les états et la densité virale
        state = new_state
        virus_density = new_virus_density
        
        # Optionnel : Afficher l'évolution des états (facultatif)
        plt.imshow(state, cmap='viridis', interpolation='nearest')
        plt.title(f'Timestep {t}')
        plt.colorbar()
        plt.pause(0.1)
    
    plt.show()
    return state, virus_density

# Paramètres d'entrée
grid_size = 100  # Taille de la grille (20x20 cellules)
N_initial = 1  # Densité initiale de virus dans chaque cellule

# Simuler l'infection
final_state, final_virus_density = simulate_infection(grid_size, N_initial)

# Afficher le résultat final
plt.imshow(final_state, cmap='viridis', interpolation='nearest')
plt.title('État final des cellules')
plt.colorbar()
plt.show()

plt.imshow(final_virus_density, cmap='inferno', interpolation='nearest')
plt.title('Densité virale finale')
plt.colorbar()
plt.show()