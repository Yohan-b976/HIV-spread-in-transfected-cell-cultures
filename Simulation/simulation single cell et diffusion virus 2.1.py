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
V = 100  # Nombre moyen de particules virales produites par virion
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



#diffusion des virus à l'echelle des pops

def diffusion_dif_pop_virus(v_infectieux, m_virus, D, dx, t, n):
    
    # Décale les matrices de diffusion pour infections
    v_infectieux = [m_virus[-1]] + v_infectieux[:-1]

    # Décale les matrices de diffusion pour maturings
    m_virus = [np.zeros((grid_size, grid_size))] + m_virus[:-1]

    # Appliquer la diffusion à toutes les matrices
    for rep in range(n):
        v_infectieux = [diffusion(inf, D, dx, t, n) for inf in v_infectieux]
        m_virus = [diffusion(mat, D, dx, t, n) for mat in m_virus]

    return v_infectieux, m_virus

def soustraction(virus_density, new_virus_density,Matrice):
    """Réattribuer à chaque matrices les qte de virus en fonction de la franction de départ"""
    return new_virus_density * Matrice/virus_density


def soustraction_dif_pop_virus(virus_density,new_virus_density,virus_infectieux):
    
    virus_infectieux = [soustraction(virus_density, new_virus_density,inf) for inf in virus_infectieux]
    
    return virus_infectieux




# Simulation du processus de Markov avec diffusion
def simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n):
    state = np.full((grid_size, grid_size), 0)
    state[10,10] = 3
    state[1,1] = 3
    number_virion_incell = np.full((grid_size, grid_size), 0)
    number_virion_incell[10,10] = 3
    number_virion_incell[[1,1]] = 2
    
    virus_density = np.full((grid_size, grid_size), N_initial)

    replication_rate = np.zeros((grid_size, grid_size))
    
    virus_infectieux = [np.zeros((grid_size, grid_size)) for _ in range(24)]
    virus_infectieux[0] = np.full((grid_size, grid_size), N_initial)
    
    maturation_virus =  [np.zeros((grid_size, grid_size)) for _ in range(7)]


    # Définir les couleurs pour chaque état
    colors = ['mistyrose' ,'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)

    for t_step in range(timesteps):
        new_state = state.copy()
        
        virus_density = np.sum(virus_infectieux, axis=0)
        new_virus_density = virus_density.copy()
        
        
        new_number_virion_incell = number_virion_incell.copy()
        replication_rate = np.zeros((grid_size, grid_size))
        

        for i in range(grid_size):
            for j in range(grid_size):
      
                if state[i, j] == INFECTIEUSE:
                    
                    p_infection = infection_probability(virus_density[i, j])
    
                    
                    if np.random.rand() < p_infection:
                        new_virus_density[i, j] -= 1
                        new_number_virion_incell[i, j] += 1
                        


                    
                    replication_rate[i, j] = min(number_virion_incell[i, j] * V, Rmax)
                    
                    
                    
                    if np.random.rand() < mu:
                        new_state[i, j] = MORTE
                        
                elif state[i, j] == MORTE:
                    new_state[i, j] = MORTE
        
        # Copie des listes de matrices
        new_virus_infectieux = virus_infectieux.copy()
        new_maturation_virus = maturation_virus.copy()
        new_maturation_virus[0] = replication_rate
        
        # Application des entrées de virus dans les cellules
        new_virus_infectieux = soustraction_dif_pop_virus(virus_density,new_virus_density, new_virus_infectieux)
        #Diffusion des virus dans l'espace
        new_virus_infectieux, new_maturation_virus = diffusion_dif_pop_virus(new_virus_infectieux, new_maturation_virus, D, dx, t, n)
        
        # Réattribution des nouvelles valeurs
        virus_infectieux, maturation_virus = new_virus_infectieux, new_maturation_virus
        state = new_state
        number_virion_incell = new_number_virion_incell

        # Affichages
        plt.imshow(np.sum(virus_infectieux, axis=0), cmap='inferno', interpolation='nearest')
        plt.title('Densité virale')
        plt.colorbar()
        plt.show()
        
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
dx =  6e-6  # Taille des cellules sur la grille
t = 1.0  # Temps total de simulation
n = 10  # Nombre de subdivisions temporelles (donc dt = t/n)

# Simuler l'infection avec diffusion
final_state, final_virus_density = simulate_infection_with_diffusion(grid_size, N_initial, dx, t, n)

# Afficher la densité virale finale
plt.imshow
