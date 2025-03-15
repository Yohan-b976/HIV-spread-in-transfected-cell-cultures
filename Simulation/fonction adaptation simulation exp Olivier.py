# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:01:22 2024

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

# objectif est de réaliser un renouveau de la culture tt les 48h avec une nouvelle concentration et les memes cell

def repiquage(state, maturation_virus, number_virion_incell, virus_density,compte_avant_infectiosité):
    state_new_medium = np.full((grid_size, grid_size), VIDE)
    maturation_virus_new_medium =  [np.zeros((grid_size, grid_size)) for _ in range(temps_maturation_virus)]
    number_virion_incell_new_medium = np.zeros((grid_size, grid_size))
    virus_density_new_medium = np.full((grid_size, grid_size), N_initial)
    compte_avant_infectiosité_new_medium = np.full((grid_size, grid_size), 0)
    
    
    cells_new_medium = np.random.choice(np.arange(grid_size*grid_size), size=int(coef_occupation_cells*grid_size*grid_size), replace=False)
    Choix_cells = np.random.choice(np.arange(grid_size*grid_size), size=int(grid_size*grid_size), replace=False)
    cellinculture = 0
    while celllinculture < densite_repiquage:
         for index_cell in Choix_cells:
             if state[index_cell // grid_size, index_cell % grid_size] != MORTE and state[index_cell // grid_size, index_cell % grid_size] != VIDE:
                 index = cells_new_medium[cellinculture]
                 state_new_medium[index // grid_size, index % grid_size] = state[index_cell // grid_size, index_cell % grid_size]
                 number_virion_incell_new_medium[index // grid_size, index % grid_size] = number_virion_incell[index_cell // grid_size, index_cell % grid_size]
                 compte_avant_infection_new_medium[index // grid_size, index % grid_size] = compte_avant_infection[index_cell // grid_size, index_cell % grid_size]
                 cellinculture += 1
            
                