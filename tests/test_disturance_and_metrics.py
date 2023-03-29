#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:28:04 2023

@author: aubouinb
"""

import control
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from src.python_anesthesia_simulator import simulator, pd_models, disturbances, metrics

ts = 1
age = 35
weight = 70
height = 170
gender = 0

# %%

George_1 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)
George_2 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)
George_3 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)

Ap = George_1.propo_pk.continuous_sys.A
Ar = George_1.remi_pk.continuous_sys.A

Bp = George_1.propo_pk.continuous_sys.B
Br = George_1.remi_pk.continuous_sys.B
A_nom = block_diag(Ap, Ar)
B_nom = block_diag(Bp, Br)
C = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
D = np.array([[0, 0]])

continuous_sys = control.ss(A_nom, B_nom, C, D)
discret_sys = continuous_sys.sample(ts)
A_nom = discret_sys.A
B_nom = discret_sys.B

# %% Simulation

N_simu = int(50 * 60/ts)

Time = np.arange(0, N_simu)*ts/60
uP, uR, uD, uS, uA = 0.1, 0.5, 0, 0, 0
Time_change = [50, 20, 30, 40, 50]
x = np.zeros((11, N_simu+1))

for index in range(N_simu):
    Dist_1 = disturbances.compute_disturbances(index*ts, dist_profil='realistic')
    George_1.one_step(uP, uR, Dist=Dist_1, noise=False)
    x[:, index+1] = A_nom @ x[:, index] + B_nom @ np.array([uP, uR])
    Dist_2 = disturbances.compute_disturbances(index*ts, dist_profil='simple')
    George_2.one_step(uP, uR, Dist=Dist_2, noise=False)
    Dist_3 = disturbances.compute_disturbances(index*ts, dist_profil='step')
    George_3.one_step(uP, uR, Dist=Dist_3, noise=False)

fig, axs = plt.subplots(8, figsize=(14, 16))
for i in range(4):
    axs[i].plot(George_1.dataframe['x_propo_' + str(i+1)], '-')
    axs[i].plot(x[i, :], '-')
    axs[i].set(xlabel='t', ylabel='$xp_' + str(i+1) + '$')
    plt.grid()
    axs[i+4].plot(George_1.dataframe['x_remi_' + str(i+1)], '-')
    axs[i+4].plot(x[i+4, :], '-')
    axs[i+4].set(xlabel='t', ylabel='$xr_' + str(i+1) + '$')
    plt.grid()

plt.show()


fig, ax = plt.subplots(3)
ax[0].plot(Time, George_1.dataframe['u_propo'])
ax[1].plot(Time, George_1.dataframe['u_remi'])
ax[2].plot(Time, George_1.dataframe['u_nore'])

ax[0].set_ylabel("Propo")
ax[1].set_ylabel("Remi")
ax[2].set_ylabel("Nore")


plt.show()


fig, ax = plt.subplots(4)

ax[0].plot(Time, George_1.dataframe['BIS'])
ax[1].plot(Time, George_1.dataframe['MAP'])
ax[2].plot(Time, George_1.dataframe['CO'])
ax[3].plot(Time, George_1.dataframe['TOL'])
ax[0].plot(Time, George_2.dataframe['BIS'])
ax[1].plot(Time, George_2.dataframe['MAP'])
ax[2].plot(Time, George_2.dataframe['CO'])
ax[3].plot(Time, George_2.dataframe['TOL'])
ax[0].plot(Time, George_3.dataframe['BIS'])
ax[1].plot(Time, George_3.dataframe['MAP'])
ax[2].plot(Time, George_3.dataframe['CO'])
ax[3].plot(Time, George_3.dataframe['TOL'])

ax[0].set_ylabel("BIS")
ax[1].set_ylabel("MAP")
ax[2].set_ylabel("CO")
ax[3].set_ylabel("TOL")
ax[3].set_xlabel("Time (min)")
plt.show()
