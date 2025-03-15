# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:11:15 2024

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

###############################################################################
# PARAMETRAGE
###############################################################################
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 150  # Nombre moyen de particules virales produites par virion
s = 0.0014  # Probabilité d'infection par contact
k = 4  # Nombre moyen de contacts par particule virale
Rmax = 700  # Valeur de coupure maximale pour le taux de réplication
mu = 0.044 # Probabilité de mort d'une cellule infectieuse
timesteps = 121 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 24 # temps necessaire pour le doublementd'une cellule
grid_size = 100
N_initial = 2.2
total_pixels = grid_size**2
dx = 7.0e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  # Nombre de subdivisions temporelles (donc dt = t/n)
temps_maturation_virus = 4
alpha = 0.09 # Coefficient de perte à chaque propagation virus mature 

coef_de_transfectees = 1 # coefficient de transfectées
pouvoir_replicatif_virus = 0 # fraction du pouvoir réplicatif du virus par rapport à un virus WT dans un cellule WT qd pas de transfection
###############################################################################
######################## FONCTIONS 
###############################################################################
#------------------------------------------------------------------------------


# Fonction de simulation de l'infection d'une cellule
def infection_probability(N_i):
    lambda_i = s * N_i * k
    return 1 - np.exp(-lambda_i)

#------------------------------------------------------------------------------
#Replication de cellule
def mitose_cell(i,j,new_state,croissance_cells,state, NAT):
    voisins, voisins_libre = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)], []
    for k in voisins:
        
        if k[0] >= 0 and k[0] < grid_size and k[1] >= 0 and k[1] < grid_size and state[k] == VIDE:
            voisins_libre +=[k]
        
        if len(voisins_libre)==0:
            croissance_cells[i,j] = 0
        else:
            
            choix_aleatoire = int(rd.random()*len(voisins_libre))
            new_state[voisins_libre[choix_aleatoire]] = NAT
            

#------------------------------------------------------------------------------
# Fonction de diffusion avec un pas de temps adaptable
def diffusion(virus_density, D, dx, t, n):
    # Calcul du pas de temps
    dt = t / n
    N = virus_density.shape[0]
    
    # Vérification de la condition de stabilité (si nécessaire)
    if dt > dx**2 / (4 * D):
        print("Attention : dt trop grand, instabilité possible.")

    # Fréquences spatiales pour la FFT
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky)

    # Noyau de diffusion dans l'espace de Fourier
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2) - alpha*dt)

    # Transformée de Fourier de la densité virale
    virus_density_fft = fft2(virus_density)

    # Convolution en espace de Fourier
    virus_density_fft_new = virus_density_fft * F_hat

    # Retour en espace réel
    virus_density_new = np.real(ifft2(virus_density_fft_new))
    
    return virus_density_new

def diffusion_maturation(virus_density, D, dx, t, n):
    # Calcul du pas de temps
    dt = t / n
    N = virus_density.shape[0]
    
    # Vérification de la condition de stabilité (si nécessaire)
    if dt > dx**2 / (4 * D):
        print("Attention : dt trop grand, instabilité possible.")

    # Fréquences spatiales pour la FFT
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky)

    # Noyau de diffusion dans l'espace de Fourier
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2) )

    # Transformée de Fourier de la densité virale
    virus_density_fft = fft2(virus_density)

    # Convolution en espace de Fourier
    virus_density_fft_new = virus_density_fft * F_hat

    # Retour en espace réel
    virus_density_new = np.real(ifft2(virus_density_fft_new))
    
    return virus_density_new


#diffusion des virus à l'echelle des pops
def diffusion_dif_pop_virus(virus_density, m_virus, D, dx, t, n):

    virus_density = m_virus[-1] +virus_density


    # Décale les matrices de diffusion pour maturings
    m_virus = [np.zeros((grid_size, grid_size))] + m_virus[:-1]

    # Appliquer la diffusion à toutes les matrices
    for rep in range(n):
        virus_density = diffusion(virus_density, D, dx, t, n) 
        m_virus = [diffusion_maturation(mat, D, dx, t, n) for mat in m_virus]
        
    return virus_density, m_virus




#------------------------------------------------------------------------------
# Simulation du processus de Markov avec diffusion

def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    # Initiation des matrices 
    state = np.full((grid_size, grid_size), VIDE)
    cells = np.random.choice(np.arange(grid_size*grid_size), size=int(coef_occupation_cells*grid_size*grid_size), replace=False) 
    croissance_cells = np.zeros((grid_size, grid_size))  # Suit l'avancée de la réplication des cellules
    
    for index in cells:
        nature = rd.random()
        if nature < coef_de_transfectees:
            state[index // grid_size, index % grid_size] =  NON_INFECTEE_TRANSFECTE
            croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())  # Attribution des valeurs de p
        else:
            state[index // grid_size, index % grid_size] = NON_INFECTEE
            croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())

    maturation_virus =  [np.zeros((grid_size, grid_size)) for _ in range(temps_maturation_virus)]
    number_virion_incell = np.zeros((grid_size, grid_size))
    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    

    # Définir les couleurs pour chaque état
    colors = ['mistyrose', 'mediumblue', 'blue', 'seagreen', 'lime', 'tomato', 'crimson', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = BoundaryNorm(bounds, cmap.N)

    for t_step in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()
        new_number_virion_incell = number_virion_incell.copy()
        replication_rate = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):

# Partie des non-infectées-----------------------------------------------------
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state, NON_INFECTEE)
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                        new_virus_density[i, j] -= 1
                        new_number_virion_incell[i, j] += 1
                        
                elif state[i, j] == NON_INFECTEE_TRANSFECTE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state, NON_INFECTEE_TRANSFECTE)
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE_TRANSFECTE
                        new_virus_density[i, j] -= 1
                        new_number_virion_incell[i, j] += 1

# Partie des infectées-----------------------------------------------------                        
                elif state[i, j] == INFECTEE:
                    compte_avant_infectiosité[i, j] += 1
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= 1
                        new_number_virion_incell[i, j] += 1


                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE
                        
                elif state[i, j] == INFECTEE_TRANSFECTE:
                    compte_avant_infectiosité[i, j] += 1
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= 1
                        new_number_virion_incell[i, j] += 1


                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE_TRANSFECTE

# Partie des infectieuses-----------------------------------------------------
                elif state[i, j] == INFECTIEUSE:
                    # Virions en maturation (produits mais pas encore infectieux)
                    replication_rate[i, j] = min(pouvoir_replicatif_virus*number_virion_incell[i, j] * V, pouvoir_replicatif_virus*Rmax)
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                
                elif state[i, j] == INFECTIEUSE_TRANSFECTE:
                    # Virions en maturation (produits mais pas encore infectieux)
                    replication_rate[i, j] = min(number_virion_incell[i, j] * V, Rmax)
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                        
# Partie des mortes-----------------------------------------------------
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
                    
# Partie gestion matrices et visualisation-------------------------------------

        # Copie des matrices            
        new_maturation_virus = maturation_virus.copy()
        new_maturation_virus[0] = replication_rate
        
        
        #Diffusion des virus dans l'espace
        new_virus_density, new_maturation_virus = diffusion_dif_pop_virus(new_virus_density, new_maturation_virus, D, dx, t, n)
        
        # Réattribution des nouvelles valeurs
        maturation_virus = new_maturation_virus
        virus_density = new_virus_density
        state = new_state
        number_virion_incell = new_number_virion_incell

        
        if t_step == 24 or t_step == 48 or t_step == 72 or t_step == 95 or t_step == 121 or t_step == 160:
           num_infectieuses = np.sum(state == INFECTIEUSE) + np.sum(state == INFECTIEUSE_TRANSFECTE)
           pourcentage_infectieuses = (num_infectieuses / (total_pixels - np.sum(state == VIDE) - np.sum(state == MORTE)))* 100
           print(f'Timestep {t_step}: {pourcentage_infectieuses:.2f}% de cellules infectieuses')

            
        # Affichage de la densité virale à chaque étape
        plt.figure(1)
        plt.clf()
        plt.imshow(virus_density, cmap='inferno', interpolation='nearest')
        plt.title(f'Densité virale {t_step}')
        plt.colorbar()
        plt.show()
        plt.pause(0.1)

        # Affichage des états cellulaires
        plt.figure(2)
        plt.clf()
        plt.imshow(new_state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t_step}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5,5.5,6.5,7.5])
        cbar.set_ticklabels(['Vide', 'Non infecté', 'Non infecté T',  'Infecté', 'Infecté T','Infectieux', 'Infectieux T', 'Mort'])
        plt.pause(0.1)
        
        
        

    plt.show()


###############################################################################
# LANCER LA SIMULATION
###############################################################################

simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow





