# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:28 2024

@author: yohan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.fft import fft2, ifft2
import random as rd


# États de la cellule
VIDE = 0
NON_INFECTEE = 1
INFECTEE = 2
INFECTIEUSE = 3
MORTE = 4
# Paramètres verifié et réels : !
# Paramètres du modèle
m = 23  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse !
V = 5000  # Nombre moyen de particules virales produites par virion incell !
s = 0.00014  # Probabilité d'infection par contact
k = 6  # Nombre moyen de contacts par particule virale
Rmax = 15000  # Valeur de coupure maximale pour le taux de réplication !
mu = 0.056 # Probabilité de mort d'une cellule infectieuse ! (tt meurt à t = 54h après infection)
timesteps = 98 # Nombre d'étapes temporelles
D = 4.48e-12  # Coefficient de diffusion !
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule !
po = 24 # temps necessaire pour le doublementd'une cellule !
perte = 0.55 # Coefficient de perte à chaque propagation virus mature  (avoir 30% de perte sur la pop)
perte_mat = 0.00008 # Coefficient de perte à chaque propagation virus non mature
virus_entrant = 1
temps_vie_virus = 24 # !  
temps_maturation_virus = 5 # en réalité n-1 !
regulation_CD4 = 18  # Échelle de temps pour la régulation à la baisse des CD4 !
ksj = 0.3  # Paramètre composite pour simplifier le modèle
tps_prod_GFP = 15 # délai avant l'apparition d'une fluorescence !
coef_quiescence = 0.2 #Diminution du métabolisme après une période de temps
coef_tps_cell_quiescente = 2

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
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2) - perte*dt)

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
    F_hat = np.exp(-4 * np.pi**2 * D * dt * (KX**2 + KY**2)- perte_mat*dt )

    # Transformée de Fourier de la densité virale
    virus_density_fft = fft2(virus_density)

    # Convolution en espace de Fourier
    virus_density_fft_new = virus_density_fft * F_hat

    # Retour en espace réel
    virus_density_new = np.real(ifft2(virus_density_fft_new))

    return virus_density_new


#diffusion des virus à l'echelle des pops

def diffusion_dif_pop_virus(v_infectieux, m_virus, D, dx, t, n):
    
    # Décale les matrices de diffusion pour infections
    v_infectieux = [m_virus[-1]] + v_infectieux[:-1]

    # Décale les matrices de diffusion pour maturings
    m_virus = [np.zeros((grid_size, grid_size))] + m_virus[:-1]

    # Appliquer la diffusion à toutes les matrices
    for rep in range(n):
        v_infectieux = [diffusion(inf, D, dx, t, n) for inf in v_infectieux]
        m_virus = [diffusion_maturation(mat, D, dx, t, n) for mat in m_virus]

    return v_infectieux, m_virus

#------------------------------------------------------------------------------
# Enlever les virus entrées dans les cellules

def soustraction(virus_density, new_virus_density,Matrice):
    """Réattribuer à chaque matrices les qte de virus en fonction de la franction de départ"""
    return new_virus_density * Matrice/virus_density


def soustraction_dif_pop_virus(virus_density,new_virus_density,virus_infectieux):
    
    virus_infectieux = [soustraction(virus_density, new_virus_density,inf) for inf in virus_infectieux]
    
    return virus_infectieux

#------------------------------------------------------------------------------
#Régulation par CD4

# Fonction pour la régulation des CD4 en fonction du temps écoulé depuis l'infection
def cd4_downregulation(j, a):
    return np.exp(-j / a)  # Fraction de CD4 restants

# Définition des taux de succès en fonction du temps depuis l'infection
def success_rate(j, a, s):
    return s * cd4_downregulation(j, a)

# Calcul du nombre de virions qui infectent une cellule donnée
def infection_events(N_j, j, a, s):
    bs_j = success_rate(j, a, s)  # Taux de succès dépendant du temps
    lam = N_j * ksj * bs_j  # Taux de l'événement Poisson
    return 1 - np.exp(-lam)

###############################################################################
#SIMULATION
###############################################################################
#------------------------------------------------------------------------------
# Simulation du processus de Markov avec diffusion

def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    
    
    # Initialisation des matrices des cellules
    state = np.full((grid_size, grid_size), VIDE)
    cells = np.random.choice(np.arange(grid_size*grid_size), size=int(coef_occupation_cells*grid_size*grid_size), replace=False) 
    croissance_cells = np.zeros((grid_size, grid_size))  # Suit l'avancée de la réplication des cellules
    
    for index in cells:
        state[index // grid_size, index % grid_size] = NON_INFECTEE
        croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random())  # Attribution des valeurs de p
        
    #Initialisation des matrices de virus
    virus_infectieux = [np.zeros((grid_size, grid_size)) for _ in range(temps_vie_virus)]
    virus_infectieux[0] = np.full((grid_size, grid_size), N_initial)
    
    maturation_virus =  [np.zeros((grid_size, grid_size)) for _ in range(temps_maturation_virus)]
    
    #Suivi de l'état des cellules
    number_virion_incell = np.zeros((grid_size, grid_size))
    compte_avant_infectiosité = np.full((grid_size, grid_size), 0)
    
    #Initialisation de la matrice GFP+
    GFP_pos = np.zeros((grid_size, grid_size))
    

    # Définir les couleurs pour chaque état
    colors = ['mistyrose', 'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    #Définir les couleurs pour la GFP
    gfp = mcolors.LinearSegmentedColormap.from_list("FluoGreenToBlack", ["black","#00FF00" ])
    norm_gfp = mcolors.Normalize(vmin=0, vmax=1)

    for t_step in range(timesteps):
        
        new_state = state.copy()
        virus_density = np.sum(virus_infectieux, axis=0)
        new_virus_density = virus_density.copy()
        new_number_virion_incell = number_virion_incell.copy()
        replication_rate = np.zeros((grid_size, grid_size))
        

        for i in range(grid_size):
            for j in range(grid_size):
                
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >= po:
                        mitose_cell(i, j, new_state, croissance_cells, state)
                    
                    # Calcul de l'infection uniquement par les virions matures
                    p_infection = infection_probability(virus_density[i, j])
                    if np.random.rand() < p_infection:
                        new_state[i, j] = INFECTEE
                        new_virus_density[i, j] -= virus_entrant
                        new_number_virion_incell[i, j] += virus_entrant


                elif state[i, j] == INFECTEE:
                    compte_avant_infectiosité[i, j] += 1
                    
                    #si infectée depuis moins de 6 heures
                    if compte_avant_infectiosité[i, j] < 6 :
                        p_infection = infection_probability(virus_density[i, j])
                        if np.random.rand() < p_infection:
                            new_virus_density[i, j] -= virus_entrant
                            new_number_virion_incell[i, j] += virus_entrant
                            
                    #si infectée depuis 6 heures ou plus, regulation des CD4
                    else:
                        tps_norm_infection = compte_avant_infectiosité[i, j] - 6
                        p_infection = infection_events(virus_density[i, j], tps_norm_infection, regulation_CD4, s)
                        if np.random.rand() < p_infection:
                            new_virus_density[i, j] -= virus_entrant
                            new_number_virion_incell[i, j] += virus_entrant
                            
                    #Production de gfp
                    if compte_avant_infectiosité[i, j] == tps_prod_GFP:
                        GFP_pos[i, j] = 1

                    #Passage à l'état infectieux
                    if compte_avant_infectiosité[i, j] >= m:
                        new_state[i, j] = INFECTIEUSE

                elif state[i, j] == INFECTIEUSE:
                    if croissance_cells[i, j] >= po*coef_tps_cell_quiescente:
                        replication_rate[i, j] += min(coef_quiescence*number_virion_incell[i, j] * V, coef_quiescence*Rmax)
                    else:
                        replication_rate[i, j] += min(number_virion_incell[i, j] * V, Rmax)
                        

                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                        GFP_pos[i, j] = 0

                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
                    
                    
        # Copie des listes de matrices
        new_virus_infectieux = virus_infectieux.copy()
        new_maturation_virus = maturation_virus.copy()
        new_maturation_virus[0] = replication_rate
        
        # Application des entrées de virus dans les cellules
        if (new_virus_density != virus_density).any(): #(evite les divisions par zéros lorsqu'il n'y a pas de virus
            new_virus_infectieux = soustraction_dif_pop_virus(virus_density,new_virus_density, new_virus_infectieux)
        
        #Diffusion des virus dans l'espace
        new_virus_infectieux, new_maturation_virus = diffusion_dif_pop_virus(new_virus_infectieux, new_maturation_virus, D, dx, t, n)
        
        # Réattribution des nouvelles valeurs
        virus_infectieux, maturation_virus = new_virus_infectieux, new_maturation_virus
        state = new_state
        number_virion_incell = new_number_virion_incell

        # # Affichages
        # plt.imshow(state, cmap=cmap, norm=norm, interpolation='nearest')
        # plt.title(f'Timestep {t_step}')
        # cbar = plt.colorbar()
        # cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])  # Positions des étiquettes
        # cbar.set_ticklabels(['Vide','Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels personnalisés
        # plt.pause(0.1)
        
        # plt.imshow(GFP_pos, cmap=gfp, norm = norm_gfp, interpolation='nearest')
        # plt.colorbar()
        # plt.title(f'Timestep {t_step}, cellules GFP+')
        # plt.show()
        
        # plt.imshow(np.sum(virus_infectieux, axis=0), cmap='inferno', interpolation='nearest')
        # plt.title(f'Timestep {t_step},Densité virale')
        # plt.colorbar()
        # plt.show()
        
        
        #Affichage des pourcentages de cellules pour les temps interessants
        if t_step == 24 or t_step == 48 or t_step == 72 or t_step == 96 or t_step == 120 or t_step == 144:
            num_gfp_pos = np.sum(GFP_pos == 1)
            pourcentage_gfp_pos = (num_gfp_pos / (np.sum(state == INFECTEE)+np.sum(state == NON_INFECTEE) + np.sum(state == INFECTIEUSE))) * 100
            print(f'Timestep {t_step}: {pourcentage_gfp_pos:.2f}% de cellules GFP+')
        
    plt.show()
    



###############################################################################
# PARAMETRAGE
###############################################################################
# Paramètres d'entrée
grid_size = 100
N_initial = 22
total_pixels = grid_size**2
dx =  2e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 30  # Nombre de subdivisions temporelles (donc dt = t/n)


###############################################################################
# LANCER LA SIMULATION
###############################################################################

# Simuler l'infection avec diffusion
simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow

                    
