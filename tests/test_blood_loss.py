#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:32:20 2023

@author: aubouinb
"""
# %% Import
import matplotlib.pyplot as plt
from python_anesthesia_simulator import simulator

# %% Initialization patient
ts = 5
age, height, weight, gender = 74, 164, 88, 1
George = simulator.Patient([age, height, weight, gender], ts=ts,
                           model_propo="Eleveld", model_remi="Eleveld", co_update=True)

# %% Simulation

N_simu = int(60 * 60/ts)

uP, uR, uN = 0.15, 0.1, 2
George.initialized_at_given_input(u_propo=uP, u_remi=uR, u_nore=uN)
blood_loss_rate = 200  # ml/min
blood_gain_rate = 50  # ml/min
for index in range(N_simu):
    if index > 1*60/ts and index < 11*60/ts:
        Bis, Co, Map, Tol = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                            blood_rate=- blood_loss_rate, noise=False)
    elif index > 15*60/ts and index < 55*60/ts:
        Bis, Co, Map, Tol = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                            blood_rate=blood_gain_rate, noise=False)
    else:
        Bis, Co, Map, Tol = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                            blood_rate=0, noise=False)
# %% plot
if __name__ == '__main__':
    fig, ax = plt.subplots(3)
    Time = George.dataframe['Time']/60
    ax[0].plot(Time, George.dataframe['u_propo'])
    ax[1].plot(Time, George.dataframe['u_remi'])
    ax[2].plot(Time, George.dataframe['u_nore'])

    ax[0].set_ylabel("Propo")
    ax[1].set_ylabel("Remi")
    ax[2].set_ylabel("Nore")
    for i in range(3):
        ax[i].grid()
    plt.ticklabel_format(style='plain')
    plt.show()

    fig, ax = plt.subplots(1)

    ax.plot(Time, George.dataframe['x_propo_4']*10, label="Propofol (x10)")
    ax.plot(Time, George.dataframe['x_remi_4']*10, label="Remifentanil(x10)")
    ax.plot(Time, George.dataframe['x_nore'], label="Norepinephrine")
    plt.title("Hypnotic effect site Concentration")
    ax.set_xlabel("Time (min)")
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(5)

    ax[0].plot(Time, George.dataframe['BIS'])
    ax[1].plot(Time, George.dataframe['MAP'])
    ax[2].plot(Time, George.dataframe['CO'])
    ax[3].plot(Time, George.dataframe['TOL'])
    ax[4].plot(Time, George.dataframe['blood_volume'])

    ax[0].set_ylabel("BIS")
    ax[1].set_ylabel("MAP")
    ax[2].set_ylabel("CO")
    ax[3].set_ylabel("TOL")
    ax[4].set_ylabel("blood volume")
    ax[4].set_xlabel("Time (min)")
    for i in range(5):
        ax[i].grid()
    plt.ticklabel_format(style='plain')
    plt.show()

# %% test
index_end_bleeding = int(12*60/ts)
index_end_transfusion = int(56*60/ts)


# if bleeding is not stopped, BIS, MAP and CO should decrease, and TOL should increase
assert George.dataframe['BIS'][0] > George.dataframe['BIS'][index_end_bleeding]
assert George.dataframe['MAP'][0] > George.dataframe['MAP'][index_end_bleeding]
assert George.dataframe['CO'][0] > George.dataframe['CO'][index_end_bleeding]
assert George.dataframe['TOL'][0] < George.dataframe['TOL'][index_end_bleeding]

# if transfusion is not stopped, BIS, MAP and CO should increase, and TOL should decrease
assert George.dataframe['BIS'][index_end_bleeding] < George.dataframe['BIS'][index_end_transfusion]
assert George.dataframe['MAP'][index_end_bleeding] < George.dataframe['MAP'][index_end_transfusion]
assert George.dataframe['CO'][index_end_bleeding] < George.dataframe['CO'][index_end_transfusion]
assert George.dataframe['TOL'][index_end_bleeding] > George.dataframe['TOL'][index_end_transfusion]


print('test ok')
