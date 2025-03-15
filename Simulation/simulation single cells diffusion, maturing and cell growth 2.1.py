# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:11:48 2024

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
V = 90  # Nombre moyen de particules virales produites par virion
s = 0.0014  # Probabilité d'infection par contact
k = 4  # Nombre moyen de contacts par particule virale
Rmax = 700  # Valeur de coupure maximale pour le taux de réplication
mu = 0.044 # Probabilité de mort d'une cellule infectieuse
timesteps = 121 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 24 # temps necessaire pour le doublementd'une cellule


#------------------------------------------------------------------------------
# Fonctions nécessaire à la simulation


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


#Fonction de diffusion des virus

def diffusion_dif_pop_virus(number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
    number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
        number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
            number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                    number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24,\
                        number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4,\
                            D, dx, t, n):
    
    number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
        number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
            number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                    number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                        number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24 = \
                            number_virion_maturing_4, number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
                                number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
                                    number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                                        number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                                            number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                                                number_virion_infection_21, number_virion_infection_22, number_virion_infection_23
    
    number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4 = number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3
    
    # Appliquer la diffusion à la matrice totale
    for rep in range(n):
        #Diffusion des virus matures et infectieux
        number_virion_infection_1 = diffusion(number_virion_infection_1, D, dx, t, n)
        number_virion_infection_2 = diffusion(number_virion_infection_2, D, dx, t, n)
        number_virion_infection_3 = diffusion(number_virion_infection_3, D, dx, t, n)
        number_virion_infection_4 = diffusion(number_virion_infection_4, D, dx, t, n)
        number_virion_infection_5 = diffusion(number_virion_infection_5, D, dx, t, n)
        number_virion_infection_6 = diffusion(number_virion_infection_6, D, dx, t, n)
        number_virion_infection_7 = diffusion(number_virion_infection_7, D, dx, t, n)
        number_virion_infection_8 = diffusion(number_virion_infection_8, D, dx, t, n)
        number_virion_infection_9 = diffusion(number_virion_infection_9, D, dx, t, n)
        number_virion_infection_10 = diffusion(number_virion_infection_10, D, dx, t, n)
        number_virion_infection_11 = diffusion(number_virion_infection_11, D, dx, t, n)
        number_virion_infection_12 = diffusion(number_virion_infection_12, D, dx, t, n)
        number_virion_infection_13 = diffusion(number_virion_infection_13, D, dx, t, n)
        number_virion_infection_14 = diffusion(number_virion_infection_14, D, dx, t, n)
        number_virion_infection_15 = diffusion(number_virion_infection_15, D, dx, t, n)
        number_virion_infection_16 = diffusion(number_virion_infection_16, D, dx, t, n)
        number_virion_infection_17 = diffusion(number_virion_infection_17, D, dx, t, n)
        number_virion_infection_18 = diffusion(number_virion_infection_18, D, dx, t, n)
        number_virion_infection_19 = diffusion(number_virion_infection_19, D, dx, t, n)
        number_virion_infection_20 = diffusion(number_virion_infection_20, D, dx, t, n)
        number_virion_infection_21 = diffusion(number_virion_infection_21, D, dx, t, n)
        number_virion_infection_22 = diffusion(number_virion_infection_22, D, dx, t, n)
        number_virion_infection_23 = diffusion(number_virion_infection_23, D, dx, t, n)
        number_virion_infection_24 = diffusion(number_virion_infection_24, D, dx, t, n)
        
        #Diffusion des virus en cours de maturation
        number_virion_maturing_2 = diffusion(number_virion_maturing_2, D, dx, t, n)
        number_virion_maturing_3 = diffusion(number_virion_maturing_3, D, dx, t, n)
        number_virion_maturing_4 = diffusion(number_virion_maturing_4, D, dx, t, n)
        
    #retour valeur matrice de maturation t=1
    number_virion_maturing_1 = np.zeros((grid_size, grid_size))
    
    return number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
        number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
            number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                    number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                        number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24,\
                            number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4








 # Soustraction des cirus intégrées dans les cellules 
def soustraction(virus_density, new_virus_density,Matrice_infection):
    """Réattribuer à chaque matrices les qte de virus en fonction de la franction de départ"""
    return new_virus_density * Matrice_infection/virus_density



def soustraction_dif_pop_virus(virus_density, new_virus_density,number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
    number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
        number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
            number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                    number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24):
    """Applique la fonction soustraction à toutes les matrices de virus"""
    number_virion_infection_1 = soustraction(virus_density, new_virus_density,number_virion_infection_1)
    number_virion_infection_1 = soustraction(virus_density, new_virus_density, number_virion_infection_1)
    number_virion_infection_2 = soustraction(virus_density, new_virus_density, number_virion_infection_2)
    number_virion_infection_3 = soustraction(virus_density, new_virus_density, number_virion_infection_3)
    number_virion_infection_4 = soustraction(virus_density, new_virus_density, number_virion_infection_4)
    number_virion_infection_5 = soustraction(virus_density, new_virus_density, number_virion_infection_5)
    number_virion_infection_6 = soustraction(virus_density, new_virus_density, number_virion_infection_6)
    number_virion_infection_7 = soustraction(virus_density, new_virus_density, number_virion_infection_7)
    number_virion_infection_8 = soustraction(virus_density, new_virus_density, number_virion_infection_8)
    number_virion_infection_9 = soustraction(virus_density, new_virus_density, number_virion_infection_9)
    number_virion_infection_10 = soustraction(virus_density, new_virus_density, number_virion_infection_10)
    number_virion_infection_11 = soustraction(virus_density, new_virus_density, number_virion_infection_11)
    number_virion_infection_12 = soustraction(virus_density, new_virus_density, number_virion_infection_12)
    number_virion_infection_13 = soustraction(virus_density, new_virus_density, number_virion_infection_13)
    number_virion_infection_14 = soustraction(virus_density, new_virus_density, number_virion_infection_14)
    number_virion_infection_15 = soustraction(virus_density, new_virus_density, number_virion_infection_15)
    number_virion_infection_16 = soustraction(virus_density, new_virus_density, number_virion_infection_16)
    number_virion_infection_17 = soustraction(virus_density, new_virus_density, number_virion_infection_17)
    number_virion_infection_18 = soustraction(virus_density, new_virus_density, number_virion_infection_18)
    number_virion_infection_19 = soustraction(virus_density, new_virus_density, number_virion_infection_19)
    number_virion_infection_20 = soustraction(virus_density, new_virus_density, number_virion_infection_20)
    number_virion_infection_21 = soustraction(virus_density, new_virus_density, number_virion_infection_21)
    number_virion_infection_22 = soustraction(virus_density, new_virus_density, number_virion_infection_22)
    number_virion_infection_23 = soustraction(virus_density, new_virus_density, number_virion_infection_23)
    number_virion_infection_24 = soustraction(virus_density, new_virus_density, number_virion_infection_24)
    
    
    


#------------------------------------------------------------------------------
# Simulation du processus de Markov avec diffusion

def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    # Initiation des matrices des cellules
    state = np.full((grid_size, grid_size), VIDE)
    cells = np.random.choice(np.arange(grid_size*grid_size), size=int(coef_occupation_cells*grid_size*grid_size), replace=False) 
    croissance_cells = np.zeros((grid_size, grid_size))  # Suit l'avancée de la réplication des cellules
    

    #Initiation de la culture avec mise en place des cellules
    for index in cells:
        state[index // grid_size, index % grid_size] = NON_INFECTEE
        croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())  # Attribution des valeurs de p
        
    #Mise en place des matrices  des virus
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    replication_rate = np.zeros((grid_size, grid_size))
    number_virion_incell = np.zeros((grid_size, grid_size))
    
    number_virion_infection_1 = np.full((grid_size, grid_size), N_initial)
    number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
        number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
            number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                    number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                        number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24 = \
                            np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)), \
                                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)), \
                                    np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)),\
                                        np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)),\
                                            np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)), \
                                                np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) 
        
    number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4 = \
        np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)) , np.zeros((grid_size, grid_size)) 
    

    # Définir les couleurs pour chaque état du cycle cellulaire
    colors = ['mistyrose', 'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)

    for t_step in range(timesteps):
        
        #copie de l'état cellulaire
        new_state = state.copy()
        
        #matrice des virus infectieux
        virus_density = sum([number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
            number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
                number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                    number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                        number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                            number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24])
        new_virus_density = virus_density.copy()

        
        #Virus entrée dans les cellules spécifique à chaque cellule
        new_number_virion_incell = number_virion_incell.copy()

        #parcours de la culture de cellule
        for i in range(grid_size):
            for j in range(grid_size):
                
                #Partie non infectées
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state)
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(new_virus_density[i, j])
                    nombre_virions = int(new_virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_incell[i, j] += nombre_virions
                        
                #partie infectée
                elif state[i, j] == INFECTEE:
                    compte_avant_infectiosité[i, j] += 1
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_incell[i, j] += nombre_virions


                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE

                elif state[i, j] == INFECTIEUSE:
                    # Virions en maturation (produits mais pas encore infectieux)
                    p_infection = infection_probability(virus_density[i, j])
                    nombre_virions = int(virus_density[i, j] * p_infection) + 1
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= nombre_virions
                        new_number_virion_incell[i, j] += nombre_virions


                    
                    replication_rate[i, j] = min(number_virion_incell[i, j] * V, Rmax)
                    number_virion_maturing_1[i, j] += replication_rate[i, j]  # Les virions produits sont immatures
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE

                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
                    
        # Calcul des nouvelles matrices de virus
        soustraction_dif_pop_virus(virus_density, new_virus_density,number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
            number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
                number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                    number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                        number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                            number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24)
        
        number_virion_incell = new_number_virion_incell

        # Diffusion des matrices de virus
        number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
            number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
                number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                    number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                        number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                            number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24,\
                                number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4 = diffusion_dif_pop_virus(number_virion_infection_1, number_virion_infection_2, number_virion_infection_3, number_virion_infection_4, \
            number_virion_infection_5, number_virion_infection_6, number_virion_infection_7, number_virion_infection_8, \
                number_virion_infection_9, number_virion_infection_10, number_virion_infection_11, number_virion_infection_12, \
                    number_virion_infection_13, number_virion_infection_14, number_virion_infection_15, number_virion_infection_16, \
                        number_virion_infection_17, number_virion_infection_18, number_virion_infection_19, number_virion_infection_20, \
                            number_virion_infection_21, number_virion_infection_22, number_virion_infection_23, number_virion_infection_24,\
                                number_virion_maturing_1, number_virion_maturing_2, number_virion_maturing_3, number_virion_maturing_4,\
                                    D, dx, t, n)
        
        if t_step == 24 or t_step == 48 or t_step == 72 or t_step == 95 or t_step == 120 or t_step == 160:
            num_infectieuses = np.sum(state == INFECTIEUSE)
            pourcentage_infectieuses = (num_infectieuses / (np.sum(state == INFECTEE)+np.sum(state == NON_INFECTEE) + np.sum(state == INFECTIEUSE))) * 100
            print(f'Timestep {t_step}: {pourcentage_infectieuses:.2f}% de cellules infectieuses')

            
        # Affichage de la densité virale à chaque étape
        plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
        plt.title(f'Densité virale {t_step}')
        plt.colorbar()
        plt.show()

        # Affichage des états cellulaires
        plt.imshow(new_state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t_step}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_ticklabels(['Vide', 'Non infecté', 'Infecté', 'Infectieux', 'Mort'])
        plt.pause(0.1)
        
        # num_cells = np.sum(state == INFECTEE)+np.sum(state == NON_INFECTEE) + np.sum(state == INFECTIEUSE)
        # num_cells_tot = total_pixels - np.sum(state == VIDE)
        # print(num_cells)
        # print(num_cells_tot)
        # print(1)

        state = new_state

    plt.show()

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
simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow