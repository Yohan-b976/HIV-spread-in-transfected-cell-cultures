# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:50:33 2024

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
NON_INFECTEE_TRANSFECTE = 2

INFECTEE = 3
INFECTEE_TRANSFECTE = 4

INFECTIEUSE = 5
INFECTIEUSE_TRANSFECTE = 6

MORTE = 7



# Paramètres du modèle
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 90  # Nombre moyen de particules virales produites par virion
s = 0.0014  # Probabilité d'infection par contact
k = 4  # Nombre moyen de contacts par particule virale
Rmax = 700  # Valeur de coupure maximale pour le taux de réplication
mu = 0.044 # Probabilité de mort d'une cellule infectieuse
timesteps = 121 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 24 # temps necessaire pour le doublementd'une cellule
coef_de_transfectées = 1 # coefficient de transfectées
# Paramètres d'entrée
grid_size = 100
N_initial = 2.2
total_pixels = grid_size**2
dx = 7.0e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  
""""""""""""
#------------------------------------------------------------------------------
# Fonctions nécessaire à la simulation


# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)


#Replication de cellule
def mitose_cell(i,j,new_state,croissance_cells,state, valeur_cell):
    voisins, voisins_libre = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)], []
    for k in voisins:
        
        if k[0] >= 0 and k[0] < grid_size and k[1] >= 0 and k[1] < grid_size and state[k] == VIDE:
            voisins_libre +=[k]
        
        if len(voisins_libre)==0:
            croissance_cells[i,j] = 0
        else:
            
            choix_aleatoire = int(rd.random()*len(voisins_libre))
            new_state[voisins_libre[choix_aleatoire]] = state[i, j]
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


# Fonction de diffusion avec les virions matures et immatures
def diffusion_coupled(virus_density, virion_maturing, D, dx, t, n):
    # Fusionner les deux matrices pour la diffusion
    total_virus_density = virus_density + virion_maturing
    
    # Appliquer la diffusion à la matrice totale
    for rep in range(n):
        total_virus_density = diffusion(total_virus_density, D, dx, t, n)
    
    # Après diffusion, répartir à nouveau les virions matures et immatures
    # (ici, on suppose que la proportion de virions matures et immatures reste la même après diffusion)
    
    total_initial = virus_density + virion_maturing
    ratio_mature = np.divide(virus_density, total_initial, out=np.zeros_like(virus_density), where=total_initial!=0)
    ratio_maturing = 1 - ratio_mature
    
    new_virus_density = total_virus_density * ratio_mature
    new_virion_maturing = total_virus_density * ratio_maturing
    
    return new_virus_density, new_virion_maturing

#------------------------------------------------------------------------------
# Simulation du processus de Markov avec diffusion

def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    # Initiation des matrices 
    state = np.full((grid_size, grid_size), VIDE)
    
    cells = np.random.choice(np.arange(grid_size*grid_size), size=int(coef_occupation_cells*grid_size*grid_size), replace=False) 
    croissance_cells = np.zeros((grid_size, grid_size))  # Suit l'avancée de la réplication des cellules
    
    number_virion_infection = np.zeros((grid_size, grid_size))
    number_virion_maturing = np.zeros((grid_size, grid_size))  # Virions immatures (non infectieux mais diffusant)
    
    for index in cells:
        nature = rd.random()
        if nature < coef_de_transfectées:
            state[index // grid_size, index % grid_size] = NON_INFECTEE_TRANSFECTE
            croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())  # Attribution des valeurs de p
        else:
            state[index // grid_size, index % grid_size] = NON_INFECTEE
            croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())




    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    replication_rate = np.zeros((grid_size, grid_size))

    # Définir les couleurs pour chaque état
    colors = ['mistyrose', 'mediumblue', 'blue', 'seagreen', 'lime', 'tomato', 'crimson', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = BoundaryNorm(bounds, cmap.N)

    for t_step in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()
        new_number_virion_infection = number_virion_infection.copy()
        new_number_virion_maturing = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state, state[i, j])
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                        
                elif state[i, j] == NON_INFECTEE_TRANSFECTE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state, state[i, j])
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE_TRANSFECTE
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
                        
                elif state[i, j] == INFECTEE_TRANSFECTE:
                    compte_avant_infectiosité[i, j] += 1
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions

                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE_TRANSFECTE

                elif state[i, j] == INFECTIEUSE:
                    # Pas de production de virus
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                    
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                
                elif state[i, j] == INFECTIEUSE_TRANSFECTE:
                    # Production de virus
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_infection[i, j] += nombre_virions
                    
                    replication_rate[i, j] = min(number_virion_infection[i, j] * V, Rmax)
                    new_number_virion_maturing[i, j] += replication_rate[i, j]  # Les virions produits sont immatures
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE

                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
                    
        # Ajout des virus ayant fini leur maturation
        new_virus_density = new_virus_density + number_virion_maturing

        # Diffusion couplée des virions matures et immatures
        virus_density, number_virion_maturing = diffusion_coupled(new_virus_density, new_number_virion_maturing, D, dx, t, n)
        
        number_virion_infection = new_number_virion_infection
        
        if t_step == 24 or t_step == 48 or t_step == 72 or t_step == 95 or t_step == 121 or t_step == 160:
            num_infectieuses = np.sum(state == INFECTIEUSE) + np.sum(state == INFECTIEUSE_TRANSFECTE)
            pourcentage_infectieuses = (num_infectieuses / (total_pixels - np.sum(state == VIDE) - np.sum(state == MORTE)))* 100
            print(f'Timestep {t_step}: {pourcentage_infectieuses:.2f}% de cellules infectieuses')

        if t_step == 0:
            virus_density_ini = virus_density
        
        if t_step == 24:
            virus_density = virus_density - virus_density_ini
            
            
        # Affichage de la densité virale à chaque étape
        plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
        plt.title(f'Densité virale {t_step}')
        plt.colorbar()
        plt.show()

        # Affichage des états cellulaires
        plt.imshow(new_state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t_step}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5,5.5,6.5,7.5])
        cbar.set_ticklabels(['Vide', 'Non infecté', 'Non infecté T',  'Infecté', 'Infecté T','Infectieux', 'Infectieux T', 'Mort'])
        plt.pause(0.1)
        
        # num_cells = np.sum(state == INFECTEE)+np.sum(state == NON_INFECTEE) + np.sum(state == INFECTIEUSE)
        # num_cells_tot = total_pixels - np.sum(state == VIDE)
        # print(num_cells)
        # print(num_cells_tot)
        # print(1)

        state = new_state

    plt.show()
    return state, virus_density

#------------------------------------------------------------------------------
# Paramètres d'entrée
grid_size = 100
N_initial = 2.2
total_pixels = grid_size**2
dx = 7.0e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  # Nombre de subdivisions temporelles (donc dt = t/n)



#------------------------------------------------------------------------------
# Simuler l'infection avec diffusion
final_state, final_virus_density = simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow











