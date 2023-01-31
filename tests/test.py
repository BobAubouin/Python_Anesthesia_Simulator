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

N_simu = int(25 * 60/Ts)

BIS = np.zeros(N_simu)
MAP = np.zeros(N_simu)
CO = np.zeros(N_simu)
NMB = np.zeros(N_simu)
RASS = np.zeros(N_simu)
Time = np.arange(0, N_simu)*Ts/60
uP, uR, uD, uS, uA = 0, 0, 0, 0, 0
up, ur, ud, us, ua = np.zeros(N_simu), np.zeros(N_simu), np.zeros(N_simu), np.zeros(N_simu), np.zeros(N_simu)
for index in range(N_simu):
    Dist = disturbances.compute_disturbances(index * Ts, 'null')
    Bis, Co, Map, Rass, Nmb = George.one_step(uP=uP, uR=uR, uA=uA, uS=uS, uD=uD, Dist=Dist, noise=False)
    BIS[index] = Bis
    MAP[index] = Map
    CO[index] = Co
    NMB[index] = Nmb
    RASS[index] = Rass
    up[index], ur[index], ud[index], us[index], ua[index] = uP, uR, uD, uS, uA

    if index >= 0 and index <= 5*60/Ts:
        uP = 1
        uR, uS, uD, uA = 0, 0, 0, 0

    elif index >= 5*60/Ts and index <= 10*60/Ts:
        uR = 1
        uP, uS, uD, uA = 0, 0, 0, 0

    elif index >= 10*60/Ts and index <= 15*60/Ts:
        uD = 1
        uR, uP, uS, uA = 0, 0, 0, 0

    elif index >= 15*60/Ts and index <= 20*60/Ts:
        uS = 1
        uR, uD, uP, uA = 0, 0, 0, 0

    elif index >= 20*60/Ts and index <= 25*60/Ts:
        uA = 1
        uR, uS, uD, uP = 0, 0, 0, 0


fig, ax = plt.subplots(5)

ax[0].plot(Time, up)
ax[1].plot(Time, ur)
ax[2].plot(Time, ud)
ax[3].plot(Time, us)
ax[4].plot(Time, ua)

ax[0].set_ylabel("Propo")
ax[1].set_ylabel("Remi")
ax[2].set_ylabel("Dop")
ax[3].set_ylabel("Snp")
ax[4].set_ylabel("AtrD")

plt.show()

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
