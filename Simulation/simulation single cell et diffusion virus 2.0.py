# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 00:07:09 2024

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
V = 90  # Nombre moyen de particules virales produites par virion
s = 0.0014  # Probabilité d'infection par contact
k = 4  # Nombre moyen de contacts par particule virale
Rmax = 700  # Valeur de coupure maximale pour le taux de réplication
mu = 0 # Probabilité de mort d'une cellule infectieuse
timesteps = 76 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 24 # temps necessaire pour le doublementd'une cellule

# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)





# Fonction de diffusion avec un pas de temps adaptable
def diffusion(virus_density, D, dx, t, n):
    # Calcul du pas de temps basé sur le temps total et le nombre de subdivisions
    dt = t / n
    N = virus_density.shape[0]
    
    # Calcul des fréquences spatiales pour la transformée de Fourier
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky)

    # Calcul du noyau de diffusion en espace de Fourier
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2))

    # Transformée de Fourier de la densité virale
    virus_density_fft = fft2(virus_density)

    # Convolution dans l'espace de Fourier
    virus_density_fft_new = virus_density_fft * F_hat

    # Retour à l'espace réel via la transformée de Fourier inverse
    virus_density_new = np.real(ifft2(virus_density_fft_new))

    return virus_density_new






# Simulation du processus de Markov avec diffusion
def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    state = np.full((grid_size, grid_size), 0)
    state[10,10] = 3
    state[1,1] = 3
    number_virion_infection = np.full((grid_size, grid_size), 0)
    number_virion_infection[10,10] = 3
    number_virion_infection[[1,1]] = 2
    
    virus_density = np.full((grid_size, grid_size), N_initial)

    replication_rate = np.zeros((grid_size, grid_size))

    # Définir les couleurs pour chaque état
    # Définir les couleurs pour chaque état
    colors = ['mistyrose' ,'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)

    for t_step in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()
        new_number_virion_infection = number_virion_infection.copy()
        replication_rate = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                # if state[i, j] == NON_INFECTEE:
                #     p_infection = infection_probability(virus_density[i, j])
                #     if np.random.rand() < p_infection:
                #         new_state[i, j] = INFECTEE
                        
                # elif state[i, j] == INFECTEE:
                #     compte_avant_infectiosité[i, j] += 1
                #     if compte_avant_infectiosité[i, j] >= m:
                #         new_state[i, j] = INFECTIEUSE
                        
                        
                if state[i, j] == INFECTIEUSE:
                    
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    print(f'nombre_virions{nombre_virions}')
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                        print(f'nombre_virions dans la cell {nombre_virions}')


                    
                    replication_rate[i, j] = min(number_virion_infection[i, j] * V, Rmax)
                    new_virus_density[i, j] += replication_rate[i, j]
                    print(f'replication_rate {replication_rate[i, j]}')
                    
                    
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                        
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE

        # Appliquer la diffusion avec un pas de temps variable
        for rep in range(n):
            
            new_virus_density = diffusion(new_virus_density, D, dx, t, n)
            
        
        virus_density = new_virus_density
        plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
        plt.title('Densité virale')
        plt.colorbar()
        plt.show()

        state = new_state
        
        number_virion_infection = new_number_virion_infection

        # Afficher l'évolution avec les couleurs fixes pour chaque état
        plt.imshow(state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t_step}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])  # Positions des étiquettes
        cbar.set_ticklabels(['Vide','Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels personnalisés
        plt.pause(0.1)
        


    plt.show()
    return state, virus_density

# Paramètres d'entrée
grid_size = 20
N_initial = 1.5
dx =  7e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  # Nombre de subdivisions temporelles (donc dt = t/n)

# Simuler l'infection avec diffusion
final_state, final_virus_density = simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow
