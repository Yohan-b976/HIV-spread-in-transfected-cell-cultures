# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:11:54 2024

@author: yohan
"""

import numpy as np

# Paramètres
a = 40  # Échelle de temps pour la régulation à la baisse des CD4
sA, sB = 0.5, 0.5  # Taux de succès initiaux pour les virions de types A et B
ksj = 0.1  # Paramètre composite pour simplifier le modèle

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
    return 1 - np.random.poisson(lam)

# Exemples d'utilisation pour différents temps écoulés depuis l'infection
j_values = [5, 10, 20, 30, 40]  # Différents temps en heures depuis la première infection
N_j = 100  # Nombre de virions
results = {}

# Simulation du nombre de virions qui infectent pour chaque temps `j`
for j in j_values:
    Zi_A = infection_events(N_j, j, a, sA)
    Zi_B = infection_events(N_j, j, a, sB)
    results[j] = (Zi_A, Zi_B)

# Afficher les résultats
for j, (Zi_A, Zi_B) in results.items():
    print(f"Temps depuis l'infection (j = {j}): Virions type A = {Zi_A}, Virions type B = {Zi_B}")
