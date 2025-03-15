# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# États de la cellule
NON_INFECTEE = 0
INFECTEE = 1
INFECTIEUSE = 2
MORTE = 3

# Paramètres du modèle
m = 5  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 2  # Nombre moyen de particules virales produites par virion
s = 0.01  # Probabilité d'infection par contact
k = 3  # Nombre moyen de contacts par particule virale
Rmax = 100  # Valeur de coupure maximale pour le taux de réplication
mu = 0.001  # Probabilité de mort d'une cellule infectieuse
timesteps = 96  # Nombre d'étapes temporelles
T_d = 22  # Temps de doublement des cellules saines
growth_reduction_factor = 0.5  # Facteur de réduction de croissance pour les cellules infectées

# Fonction de croissance exponentielle
def croissance_exponentielle(N0, t, Td):
    return N0 * 2 ** (t / Td)

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
    # Densité de population des cellules
    cell_population = np.full((grid_size, grid_size), N_initial)
    
    # Suivre la progression dans le temps
    for t in range(timesteps):
        new_state = state.copy()  # Pour stocker les nouveaux états
        new_virus_density = virus_density.copy()
        new_cell_population = cell_population.copy()
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Mise à jour de la densité cellulaire
                if state[i, j] == NON_INFECTEE:
                    # Les cellules non infectées croissent exponentiellement
                    new_cell_population[i, j] = croissance_exponentielle(cell_population[i, j], t, T_d)
                    # Calculer la probabilité d'infection
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE  # Cellule devient infectée

                elif state[i, j] == INFECTEE:
                    # Compte à rebours vers l'état infectieux
                    if t >= m:
                        new_state[i, j] = INFECTIEUSE  # Cellule devient infectieuse
                    # Réduction de la croissance des cellules infectées
                    new_cell_population[i, j] *= growth_reduction_factor

                elif state[i, j] == INFECTIEUSE:
                    # Cellule infectieuse produit des virus
                    Z_i = np.random.poisson(virus_density[i, j] * k)
                    replication_rate = min(Z_i * V, Rmax)
                    # Libération des virions dans l'environnement local
                    new_virus_density[i, j] += replication_rate

                    # Mort possible de la cellule infectieuse
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE  # Cellule devient morte

                # Une cellule morte reste morte
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
                    # La population de cellules mortes ne change pas

        # Mettre à jour les états, la densité virale et la population cellulaire
        state = new_state
        virus_density = new_virus_density
        cell_population = new_cell_population
        
        # Optionnel : Afficher l'évolution des états (facultatif)
        plt.imshow(state, cmap='viridis', interpolation='nearest')
        plt.title(f'Timestep {t}')
        plt.colorbar()
        plt.pause(0.1)
    
    plt.show()
    return state, virus_density, cell_population

# Paramètres d'entrée
grid_size = 20  # Taille de la grille (20x20 cellules)
N_initial = 100  # Densité initiale de cellules

# Simuler l'infection
final_state, final_virus_density, final_cell_population = simulate_infection(grid_size, N_initial)

# Afficher le résultat final
plt.imshow(final_state, cmap='viridis', interpolation='nearest')
plt.title('État final des cellules')
plt.colorbar()
plt.show()

plt.imshow(final_virus_density, cmap='inferno', interpolation='nearest')
plt.title('Densité virale finale')
plt.colorbar()
plt.show()

plt.imshow(final_cell_population, cmap='plasma', interpolation='nearest')
plt.title('Densité cellulaire finale')
plt.colorbar()
plt.show()