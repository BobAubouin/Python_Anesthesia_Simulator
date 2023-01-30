#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:28:04 2023

@author: aubouinb
"""


import numpy as np
import matplotlib.pyplot as plt
from src.python_anesthesia_simulator import patient, disturbances


Ts = 5
George = patient.Patient(age=18, height=170, weight=60, gender=0, model_propo="Eleveld", Ts=Ts)

N_simu = int(5 * 60/Ts)

BIS = np.zeros(N_simu)
MAP = np.zeros(N_simu)
CO = np.zeros(N_simu)
NMB = np.zeros(N_simu)
RASS = np.zeros(N_simu)
Time = np.arange(0, N_simu)*Ts
uP, uR, uA = 0, 0, 0
for index in range(N_simu):
    Dist = disturbances.compute_disturbances(index * Ts, 'realistic')
    Bis, Co, Map, Nmb, Rass = George.one_step(uP, uR, Dist=Dist, noise=False)
    BIS[index] = Bis
    MAP[index] = Map
    CO[index] = Co
    NMB[index] = Nmb
    RASS[index] = Rass

    if index == 0:
        uP, uR, uA = 20, 20, 20
    else:
        uP, uR, uA = 0, 0, 0


fig, ax = plt.subplots(5)

ax[0].plot(Time, BIS)
ax[1].plot(Time, MAP)
ax[2].plot(Time, CO)
ax[3].plot(Time, NMB)
ax[4].plot(Time, RASS)

ax[0].set_ylabel("BIS")
ax[1].set_ylabel("MAP")
ax[2].set_ylabel("CO")
ax[3].set_ylabel("NMB")
ax[4].set_ylabel("RASS")

plt.show()
