# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:30:58 2024

@author: yohan
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import random as rd
from matplotlib.animation import FuncAnimation

# États de la cellule
VIDE = 0
NON_INFECTEE = 1
INFECTEE = 2
INFECTIEUSE = 3
MORTE = 4


# Paramètres du modèle
coef_occupation_cells = 0.225 # fraction de pixel occupé par une cellule
po = 25 # temps necessaire pour le doublementd'une cellule
m = 24  # Nombre d'étapes avant qu'une cellule infectée devienne infectieuse
V = 4  # Nombre moyen de particules virales produites par virion
s = 0.008  # Probabilité d'infection par contact
k = 2  # Nombre moyen de contacts par particule virale
Rmax = 40  # Valeur de coupure maximale pour le taux de réplication
mu = 0.0008  # Probabilité de mort d'une cellule infectieuse
timesteps = 80  # Nombre d'étapes temporelles

data_cell = []
nbr_cell = []
#Replication de cellule
def mitose_cell(i,j,new_state,croissance_cells,state):
    voisins, voisins_libre = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)], []
    for k in voisins:
        
        if k[0] >= 0 and k[0] < 100 and k[1] >= 0 and k[1] < 100 and state[k] == VIDE:
            voisins_libre +=[k]
        
        if len(voisins_libre)==0:
            croissance_cells[i,j] = 0
        else:
            
            choix_aleatoire = int(rd.random()*len(voisins_libre))
            new_state[voisins_libre[choix_aleatoire]] = NON_INFECTEE
            croissance_cells[i,j] = 0


# Simulation du processus de Markov
def simulate_infection(grid_size, N_initial, data_cell,nbr_cell ):
    state = np.full((grid_size, grid_size), VIDE)
    cells = np.random.choice(np.arange(grid_size*grid_size), size = int(coef_occupation_cells*grid_size*grid_size), replace = False) 
    croissance_cells = np.zeros((grid_size, grid_size)) # suit l'avancé de la réplication des cellules
    
    for index in cells:
        state[index // grid_size, index % grid_size] = NON_INFECTEE
        croissance_cells[index // grid_size, index % grid_size] = int(po * rd.random()) # attribution des valeurs de p
        
        
    #ne sert pas encore___________
    virus_density = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité = np.zeros((grid_size, grid_size))
    replication_rate = np.zeros((grid_size, grid_size))
    #-----------------------------------------------------------
    
    # Définir les couleurs pour chaque état
    colors = ['mistyrose' ,'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    
    for t in range(timesteps):
        new_state = state.copy()
        new_virus_density = virus_density.copy()
        
        for i in range(grid_size):
            for j in range(grid_size):
                if state[i, j] == NON_INFECTEE:
                    croissance_cells[i, j] += 1
                    if croissance_cells[i, j] >=po:
                        mitose_cell(i,j,new_state,croissance_cells,state)
        data_cell += [new_state]
        state = new_state
        virus_density = new_virus_density
    # Afficher l'évolution avec les couleurs fixes pour chaque état
        plt.imshow(state, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Timestep {t}')
        cbar = plt.colorbar()
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])  # Positions des étiquettes
        cbar.set_ticklabels(['Vide','Non infecté', 'Infecté', 'Infectieux', 'Mort'])  # Labels personnalisés
        plt.pause(0.1)
        num_cells = np.sum(state == NON_INFECTEE)
        print(num_cells)
        nbr_cell +=[num_cells]

    plt.show()
    return state, virus_density



# Paramètres d'entrée
grid_size = 100
N_initial = 1.5
total_pixels = grid_size**2

# Simuler l'infection
final_state, final_virus_density = simulate_infection(grid_size, N_initial,data_cell,nbr_cell )

# Définir les couleurs pour chaque état
colors = ['mistyrose', 'mediumblue', 'seagreen', 'tomato', 'black']  # Couleurs pour VIDE, NON_INFECTEE, INFECTEE, INFECTIEUSE, MORTE
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5]
norm = BoundaryNorm(bounds, cmap.N)
    
    
# Création de la figure et de la première image
fig, ax = plt.subplots()
cax = ax.imshow(data_cell[0], cmap = cmap, norm=norm, interpolation='nearest')
ax.set_title('Timestep 0')

# Création de la colorbar avec les ticks personnalisés
cbar = plt.colorbar(cax, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
cbar.set_ticklabels(['Vide', 'Non infecté', 'Infecté', 'Infectieux', 'Mort'])

# Fonction d'animation
def update(frame):
    # Met à jour les données
    cax.set_array(data_cell[frame])

    # Met à jour le titre avec l'étape temporelle
    ax.set_title(f'Timestep {frame}')
    
    return [cax]

# Création de l'animation
ani = FuncAnimation(fig, update, frames=len(data_cell), interval=600, blit=True)

# Sauvegarde en vidéo (mp4)
ani.save('simulation_cell_growth_80.mp4', writer='ffmpeg')

plt.show()


time = np.linspace(0, 80, 80)
reel =  [coef_occupation_cells*grid_size*grid_size*math.exp(np.log(2) * (t / 22)) for t in time]

plt.plot(time, nbr_cell, label='Simulated growth of the cells')
plt.plot(time, reel, label='Theoretical growth of the cells')
plt.xlabel('Temps (en heure)')
plt.ylabel('Nombre de cellule')
plt.title('Evolution de la population de cellule de la simulation en comparaison de la réalité')
plt.legend()
plt.grid(True)

# Affichage du graphique
plt.show()
