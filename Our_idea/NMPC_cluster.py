#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:46:11 2022

@author: aubouinb
"""

from scipy.linalg import block_diag
from filterpy.common import Q_continuous_white_noise
from functools import partial
import pandas as pd
import numpy as np
import multiprocessing
import time
from Estimators import EKF
from Controller import NMPC, MPC
from src.PAS import Patient, disturbances, metrics
import sys
import os
path = os.getcwd()
path_root = path[:-9]
sys.path.append(str(path_root))


def simu(Patient_info: list, style: str, MPC_param: list, random_PK: bool = False, random_PD: bool = False):
    ''' This function perform a closed-loop Propofol-Remifentanil anesthesia simulation with a PID controller.

    Inputs: - Patient_info: list of patient informations, Patient_info = [Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax]
            - style: either 'induction' or 'maintenance' to describe the phase to simulate
            - MPC_param: parameter of the PID controller P = [N, Q, R]
            - random: bool to add uncertainty to simulate intra-patient variability in the patient model

    Outputs:- IAE: Integrated Absolute Error, performance index of the function
            - data: list of the signals during the simulation data = [BIS, MAP, CO, up, ur]
    '''

    age = Patient_info[0]
    height = Patient_info[1]
    weight = Patient_info[2]
    gender = Patient_info[3]
    Ce50p = Patient_info[4]
    Ce50r = Patient_info[5]
    gamma = Patient_info[6]
    beta = Patient_info[7]
    E0 = Patient_info[8]
    Emax = Patient_info[9]

    ts = 5

    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = Patient.Patient(age, height, weight, gender, BIS_param=BIS_param,
                             Random_PK=random_PK, Random_PD=random_PD, Te=ts)  # , model_propo = 'Eleveld', model_remi = 'Eleveld')

    # Nominal parameters
    George_nominal = Patient.Patient(
        age, height, weight, gender, BIS_param=[None] * 6, Te=ts)
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    BIS_param_nominal[4] = George.BisPD.BIS_param[4]

    Ap = George_nominal.PropoPK.A
    Ar = George_nominal.RemiPK.A
    Bp = George_nominal.PropoPK.B
    Br = George_nominal.RemiPK.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # init state estimator
    Q = Q_continuous_white_noise(4, spectral_density=0, block_size=2)
    P0 = np.diag([10**0] * 8)
    estimator = EKF(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts,
                    P0=P0, R=10, Q=Q)

    # init controller
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    R_mpc = MPC_param[2]
    ki_mpc = MPC_param[3]
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    dup_max = 0.2 * ts * 100
    dur_max = 0.4 * ts * 100

    MPC_controller = NMPC(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts, N=N_mpc, Nu=Nu_mpc,
                          R=R_mpc, umax=[up_max, ur_max], dumax=[dup_max, dur_max],
                          dumin=[-dup_max, - dur_max], dymin=0, ki=0)

    # MPC_controller = MPC(A_nom, B_nom, BIS_param = BIS_param_nominal, ts = ts, N = N_mpc, Nu = Nu_mpc,
    #                       Q = Q_mpc, R = R_mpc, umax = [up_max, ur_max], dumax = [dup_max, dur_max],
    #                       dumin = [-dup_max, - dur_max], ymin = 0, ki = 0)

    if style == 'induction':
        N_simu = int(10 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_cible_MPC = np.zeros(N_simu)
        BIS_EKF = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Xp_EKF = np.zeros((4, N_simu))
        Xr_EKF = np.zeros((4, N_simu))
        uP = 1e-3
        uR = 1e-3
        for i in range(N_simu):

            Dist = disturbances.compute_disturbances(i * ts, 'null')
            Bis, Co, Map = George.one_step(uP, uR, Dist=Dist, noise=False)
            Xp[:, i] = George.PropoPK.x.T[0]
            Xr[:, i] = George.RemiPK.x.T[0]

            BIS[i] = min(100, Bis)
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            Up[i] = uP
            Ur[i] = uR
            # estimation
            X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
            Xp_EKF[:, i] = X[:4]
            Xr_EKF[:, i] = X[4:]
            # X_MPC = np.concatenate((Xp[:,i],Xr[:,i]),axis = 0)
            if i == 20:  # or (BIS_EKF[i]<50 and MPC_controller.ki==0):
                MPC_controller.ki = ki_mpc
                BIS_cible = 50
            X = np.clip(X, a_min=0, a_max=1e10)
            uP, uR, bis = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
            BIS_cible_MPC[i] = MPC_controller.internal_target

    elif style == 'total':
        N_simu = int(60 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_EKF = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Xp_EKF = np.zeros((4, N_simu))
        Xr_EKF = np.zeros((4, N_simu))
        L = np.zeros(N_simu)
        uP = 1
        uR = 1
        for i in range(N_simu):
            # if i == 100:
            #     print("break")

            Dist = disturbances.compute_disturbances(i * ts, 'realistic')
            Bis, Co, Map = George.one_step(uP, uR, Dist=Dist, noise=True)
            Xp[:, i] = George.PropoPK.x.T[0]
            Xr[:, i] = George.RemiPK.x.T[0]

            BIS[i] = min(100, Bis)
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            Up[i] = uP
            Ur[i] = uR
            # estimation
            X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
            Xp_EKF[:, i] = X[:4]
            Xr_EKF[:, i] = X[4:]
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])

    error = BIS - BIS_cible
    for i in range(len(error)):
        if error[i] < 0:
            error[i] = error[i] * 2
    IAE = np.sum(np.abs(error))

    return (IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF],
            George.BisPD.BIS_param)


# %% Inter patient variability
# Simulation parameter
phase = 'induction'
Number_of_patient = 128
MPC_param = [20, 20, 10**(1.5) * np.diag([3, 1]), 1e-2]


def one_simu(i):
    '''cost of one simulation, i is the patient index'''
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    iae, data, BIS_param = simu(Patient_info, 'induction', MPC_param,
                                random_PK=True, random_PD=True,)
    return ([iae, data, BIS_param, i])


# param_opti = pd.read_csv('optimal_parameters_MPC.csv')
# x = param_opti.to_numpy()
# x = x[0, 1:]
# x[3] = 2e-2
# x = [30,  30,  -6,  0]
pool_obj = multiprocessing.Pool(8)
# func = partial(one_simu, x)
result = pool_obj.map(one_simu, range(0, Number_of_patient))
pool_obj.close()
pool_obj.join()

# print([r[0] for r in result])
df = pd.DataFrame()

for i in range(Number_of_patient):
    print(i)

    data = result[i][1]
    name = ['BIS', 'MAP', 'CO', 'Up', 'Ur']
    dico = {str(i) + '_' + name[j]: data[j] for j in range(5)}
    df = pd.concat([df, pd.DataFrame(dico)], axis=1)

df.to_csv("result_NMPC_n=" + str(Number_of_patient) + '.csv')
