#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:32:20 2023

@author: aubouinb
"""

import matplotlib.pyplot as plt
from python_anesthesia_simulator import simulator


Ts = 120
George_1 = simulator.Patient([18, 170, 60, 0], ts=Ts, model_propo="Eleveld", model_remi="Eleveld", co_update=True)
George_2 = simulator.Patient([18, 170, 60, 0], ts=Ts, model_propo="Eleveld", model_remi="Eleveld", co_update=False)


# %% Simulation

N_simu = int(720 * 60/Ts)

bis_target_1 = 50
tol_target_1 = 0.9
map_target_1 = 80

George_1.initialized_at_maintenance(bis_target=50, tol_target=0.9, map_target=80)
uP, uR, uN = George_1.u_propo_eq, George_1.u_remi_eq, George_1.u_nore_eq
George_1.one_step(u_propo=uP, u_remi=uR, u_nore=uN, noise=False)

bis_target_2 = 40
tol_target_2 = 0.95
map_target_2 = 85

up_2, ur_2 = George_2.find_bis_equilibrium_with_ratio(bis_target=bis_target_1, rp_ratio=2)

George_2.one_step(u_propo=up_2, u_remi=ur_2, u_nore=0, noise=False)

uP, uR, uN = George_1.find_equilibrium(bis_target=40, tol_target=0.95, map_target=85)
for index in range(N_simu):
    George_1.one_step(u_propo=uP, u_remi=uR, u_nore=uN, noise=False)
    George_2.one_step(u_propo=up_2, u_remi=ur_2, u_nore=0, noise=False)
# %% plot
if __name__ == '__main__':
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
    ax.plot(Time, George_1.dataframe['x_nore'], label="Norepinephrine")
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

    # plot input and bis for patient 2
    Time = George_2.dataframe['Time']/60
    fig, ax = plt.subplots(2)
    ax[0].plot(Time, George_2.dataframe['u_propo'], label="Propofol")
    ax[0].plot(Time, George_2.dataframe['u_remi'], label="Remifentanil")
    ax[0].plot(Time, George_2.dataframe['u_nore'], label="Norepinephrine")
    ax[0].set_ylabel("Input")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(Time, George_2.dataframe['BIS'])
    ax[1].set_ylabel("BIS")
    ax[1].set_xlabel("Time (min)")
    ax[1].grid()
    plt.show()

# %% test
# Verify that the equilibrium is reached at the beginning and at the end of the simulation
assert abs(George_1.dataframe['BIS'].iloc[0]-bis_target_1) < 5e-1
assert abs(George_1.dataframe['BIS'].iloc[-1]-bis_target_2) < 1
assert abs(George_1.dataframe['TOL'].iloc[0]-tol_target_1) < 1e-2
assert abs(George_1.dataframe['TOL'].iloc[-1]-tol_target_2) < 1e-2
assert abs(George_1.dataframe['MAP'].iloc[0]-map_target_1) < 1e-2
assert abs(George_1.dataframe['MAP'].iloc[-1]-map_target_2) < 1e-1
assert abs(George_2.dataframe['BIS'].iloc[-1]-bis_target_1) < 1

print('test ok')
