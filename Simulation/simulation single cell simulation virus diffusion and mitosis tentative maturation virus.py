# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:14:19 2024

@author: yohan
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.fft import fft2, ifft2
import random as rd

# États de la cellule
VIDE = 0
NON_INFECTEE = 1
INFECTEE = 2
INFECTIEUSE = 3
MORTE = 4

# Paramètres du modèle
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 5  # Nombre moyen de particules virales produites par virion
s = 0.08  # Probabilité d'infection par contact
k = 0.5  # Nombre moyen de contacts par particule virale
Rmax = 100  # Valeur de coupure maximale pour le taux de réplication
mu = 0.05  # Probabilité de mort d'une cellule infectieuse
timesteps = 78 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 24 # temps necessaire pour le doublementd'une cellule


# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)

#Replication de cellule
def mitose_cell(i,j,new_state,croissance_cells,state):
    voisins, voisins_libre = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)], []
    for k in voisins:
        
        if k[0] >= 0 and k[0] < grid_size and k[1] >= 0 and k[1] < grid_size and state[k] == VIDE:
            voisins_libre +=[k]
        
        if len(voisins_libre)==0:
            croissance_cells[i,j] = 0
        else:
            
            choix_aleatoire = int(rd.random()*len(voisins_libre))
            new_state[voisins_libre[choix_aleatoire]] = NON_INFECTEE
            croissance_cells[i,j] = 0

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
    #Initiation des matrices 
    state = np.full((grid_size, grid_size), VIDE)
    cells = np.random.choice(np.arange(grid_size*grid_size), size = int(coef_occupation_cells*grid_size*grid_size), replace = False) 
    croissance_cells = np.zeros((grid_size, grid_size)) # suit l'avancé de la réplication des cellules
    number_virion_infection = np.zeros((grid_size, grid_size))
    number_virion_maturing = np.zeros((grid_size, grid_size))
    
    for index in cells:
       state[index // grid_size, index % grid_size] = NON_INFECTEE
       croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random()) # attribution des valeurs de p
    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
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
        new_number_virion_maturing =  np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    
                    if croissance_cells[i, j] >=po:
                        mitose_cell(i,j,new_state,croissance_cells,state)
                        
                    p_infection = infection_probability(virus_density[i, j])
                    
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                        
                elif state[i, j] == INFECTEE:
                    compte_avant_infectiosité[i, j] += 1
                    
                    p_infection = infection_probability(virus_density[i, j])
                    
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                        
                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE
                        
                        
                elif state[i, j] == INFECTIEUSE:
                    p_infection = infection_probability(virus_density[i, j])
                    
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                        
                    replication_rate[i, j] = min(number_virion_infection[i, j] * V, Rmax)
                    new_number_virion_maturing[i, j] += replication_rate[i, j]
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                        
                        
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE

        # Appliquer la diffusion avec un pas de temps variable
        for rep in range(n):
            new_virus_density = diffusion(new_virus_density, D, dx, t, n)
            
        #Diffusion et affichage de la map de virus
        plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
        plt.title(f'Densité virale {t_step}')
        plt.colorbar()
        plt.show()

        state = new_state
        
        virus_density = new_virus_density + number_virion_maturing
        number_virion_maturing = new_number_virion_maturing
        
        
        if t_step == 24 or t_step == 48 or t_step == 72 or t_step == 95:
            num_infectieuses = np.sum(state == INFECTIEUSE)
            pourcentage_infectieuses = (num_infectieuses / (np.sum(state == INFECTEE)+np.sum(state == NON_INFECTEE) + np.sum(state == INFECTIEUSE))) * 100
            print(f'Timestep {t_step}: {pourcentage_infectieuses:.2f}% de cellules infectieuses')


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
grid_size = 100
N_initial = 1.5
total_pixels = grid_size**2
dx = 1.5e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  # Nombre de subdivisions temporelles (donc dt = t/n)

# Simuler l'infection avec diffusion
final_state, final_virus_density = simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow

