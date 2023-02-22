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
George = patient.Patient(age=18, height=170, weight=60, gender=0, Ts=Ts, model_propo="Eleveld", model_remi="Eleveld")


# %% Simulation

N_simu = int(50 * 60/Ts)

Time = np.arange(0, N_simu)*Ts/60
uP, uR, uD, uS, uA = 0, 0, 0, 0, 0
Time_change = [50, 20, 30, 40, 50]
for index in range(N_simu):
    Dist = disturbances.compute_disturbances(index * Ts, 'null')
    George.one_step(uP=uP, uR=uR, uA=uA, uS=uS, uD=uD, Dist=Dist, noise=False)

    if index >= 0 and index <= Time_change[0]*60/Ts:
        uP = 1
        uR, uS, uD, uA = 0, 0, 0, 0

    elif index >= Time_change[0]*60/Ts and index <= Time_change[1]*60/Ts:
        uR = 1
        uP, uS, uD, uA = 0, 0, 0, 0

    elif index >= Time_change[1]*60/Ts and index <= Time_change[2]*60/Ts:
        uD = 1
        uR, uP, uS, uA = 0, 0, 0, 0

    elif index >= Time_change[2]*60/Ts and index <= Time_change[3]*60/Ts:
        uS = 1
        uR, uD, uP, uA = 0, 0, 0, 0

    elif index >= Time_change[3]*60/Ts:
        uA = 1
        uR, uS, uD, uP = 0, 0, 0, 0


fig, ax = plt.subplots(5)

ax[0].plot(Time, George.dataframe['u_propo'])
ax[1].plot(Time, George.dataframe['u_remi'])
ax[2].plot(Time, George.dataframe['u_dopamine'])
ax[3].plot(Time, George.dataframe['u_snp'])
ax[4].plot(Time, George.dataframe['u_atracium'])

ax[0].set_ylabel("Propo")
ax[1].set_ylabel("Remi")
ax[2].set_ylabel("Dop")
ax[3].set_ylabel("Snp")
ax[4].set_ylabel("AtrD")

plt.show()

fig, ax = plt.subplots(1)

ax.plot(Time, George.dataframe['x_propo_4'], label="Propofol")
ax.plot(Time, George.dataframe['x_remi_4'], label="Remifentanil")

ax.set_ylabel("Concentration")
ax.set_xlabel("Time (min)")
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(1)

ax.plot(Time, George.dataframe['x_hemo_c_propo'], label="Propofol")
ax.plot(Time, George.dataframe['x_hemo_c_remi'], label="Remifentanil")

ax.set_ylabel("Concentration MAP")
ax.set_xlabel("Time (min)")
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(5)

ax[0].plot(Time, George.dataframe['BIS'])
ax[1].plot(Time, George.dataframe['MAP'])
ax[2].plot(Time, George.dataframe['CO'])
ax[3].plot(Time, George.dataframe['NMB'])
ax[4].plot(Time, George.dataframe['RASS'])

ax[0].set_ylabel("BIS")
ax[1].set_ylabel("MAP")
ax[2].set_ylabel("CO")
ax[3].set_ylabel("NMB")
ax[4].set_ylabel("RASS")
ax[4].set_xlabel("Time (min)")
plt.show()
