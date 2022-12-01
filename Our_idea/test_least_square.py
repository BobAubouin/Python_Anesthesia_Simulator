#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:11:26 2022

@author: aubouinb
"""

import sys
import os 
path = os.getcwd()
path_root = path[:-7]
sys.path.append(str(path_root))
from src.PAS import Patient, disturbances, metrics

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


George = Patient.Patient(35, 170, 70, 0, BIS_param = [None]*6, Te = 5)

cer = np.linspace(0, 8, 100)
cep = np.linspace(0, 8, 100)
cer, cep = np.meshgrid(cer, cep)
BIS = George.BisPD.compute_bis(cep, cer)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(cer, cep, BIS, cmap=cm.jet, linewidth=0.1, alpha = 0.5)
ax.set_xlabel('Remifentanil')
ax.set_ylabel('Propofol')
ax.set_zlabel('BIS')
fig.colorbar(surf, shrink=0.5, aspect=8)
ax.view_init(12, -72)


Cep = []
Cer = []
bis = []
for i,j in product(range(100), range(100)):
    if BIS[i,j]<70 and BIS[i,j]>30:
        Cep.append(cep[i,j])
        Cer.append(cer[i,j])
        bis.append(BIS[i,j])

A = np.array([Cep, Cer, np.ones(len(Cer))]).T
b = bis
sol = np.linalg.lstsq(A, b, rcond=None)
sol = sol[0]

new_formula = lambda x,y: sol[0]*x + sol[1]*y + sol[2]

BIS2 = new_formula(cep, cer)
ax.plot_surface(cer, cep, BIS2, cmap=cm.jet, linewidth=0.1)