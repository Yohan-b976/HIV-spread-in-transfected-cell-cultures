# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:45:14 2024

@author: yohan
"""

import numpy as np
from scipy.fftpack import fft2, ifft2

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

    # Optionnel : gestion des bords (réfléchissants)
    virus_density_new[0, :] += virus_density_new[-1, :]
    virus_density_new[-1, :] += virus_density_new[0, :]
    virus_density_new[:, 0] += virus_density_new[:, -1]
    virus_density_new[:, -1] += virus_density_new[:, 0]

    return virus_density_new
