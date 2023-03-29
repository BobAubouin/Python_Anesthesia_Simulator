#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:32:20 2023

@author: aubouinb
"""

import numpy as np
import matplotlib.pyplot as plt
from src.python_anesthesia_simulator import simulator


Ts = 60
George_1 = simulator.Patient([18, 170, 60, 0], ts=Ts, model_propo="Eleveld", model_remi="Eleveld", co_update=True)
George_1.tol_pd.plot_surface()

# %% Simulation

N_simu = int(360 * 60/Ts)

George_1.initialized_at_maintenance(bis_target=50, tol_target=0.9, map_target=80)
uP, uR, uN = George_1.u_propo_eq, George_1.u_remi_eq, George_1.u_nore_eq
Bis, Co, Map, Tol = George_1.one_step(u_propo=uP, u_remi=uR, u_nore=uN, noise=False)
uP, uR, uN = George_1.find_equilibrium(bis_target=40, tol_target=0.95, map_target=85)
for index in range(N_simu):
    Bis, Co, Map, Tol = George_1.one_step(u_propo=uP, u_remi=uR, u_nore=uN, noise=False)


fig, ax = plt.subplots(3)
Time = George_1.dataframe['Time']/60
ax[0].plot(Time, George_1.dataframe['u_propo'])
ax[1].plot(Time, George_1.dataframe['u_remi'])
ax[2].plot(Time, George_1.dataframe['u_nore'])

ax[0].set_ylabel("Propo")
ax[1].set_ylabel("Remi")
ax[2].set_ylabel("Nore")
for i in range(3):
    ax[i].grid()

plt.show()

fig, ax = plt.subplots(1)

ax.plot(Time, George_1.dataframe['x_propo_4'], label="Propofol")
ax.plot(Time, George_1.dataframe['x_remi_4'], label="Remifentanil")
ax.plot(Time, George_1.dataframe['c_blood_nore'], label="Norepinephrine")
plt.title("Hypnotic effect site Concentration")
ax.set_xlabel("Time (min)")
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(4)

ax[0].plot(Time, George_1.dataframe['BIS'])
ax[1].plot(Time, George_1.dataframe['MAP'])
ax[2].plot(Time, George_1.dataframe['CO'])
ax[3].plot(Time, George_1.dataframe['TOL'])

ax[0].set_ylabel("BIS")
ax[1].set_ylabel("MAP")
ax[2].set_ylabel("CO")
ax[3].set_ylabel("TOL")
ax[3].set_xlabel("Time (min)")
for i in range(4):
    ax[i].grid()
plt.ticklabel_format(style='plain')
plt.show()
