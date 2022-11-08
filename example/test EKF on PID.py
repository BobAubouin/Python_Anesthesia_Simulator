#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:05:00 2022

@author: aubouinb
"""

import sys
import os 
path = os.getcwd()
path_root = path[:-7]
sys.path.append(str(path_root))
from src.PAS import Patient, disturbances, metrics
from Controller import PID
from Estimators import EKF


import numpy as np
import pandas as pd
import casadi as cas
from filterpy.common import Q_continuous_white_noise
from scipy.linalg import block_diag
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.io import output_notebook
import matplotlib.pyplot as plt



age = 24
height = 170
weight = 55
gender = 1
Ce50p = 6.5
Ce50r = 12.2
gamma = 3
beta = 0.5
E0 = 98
Emax = 95


BIS_param = [None]*6 #[Ce50p, Ce50r, gamma, beta, E0, Emax] #
George = Patient.Patient(age, height, weight, gender, BIS_param = BIS_param, Random_PK = True, Random_PD = True)
# George.BisPD.plot_surface()
param_opti = pd.read_csv('optimal_parameters.csv')
BIS_cible = 70
up_max = 6.67
ur_max = 16.67
ratio = 2
Kp = float(param_opti.loc[param_opti['ratio']==ratio, 'Kp'])
Ti = float(param_opti.loc[param_opti['ratio']==ratio, 'Ti'])
Td = float(param_opti.loc[param_opti['ratio']==ratio, 'Td'])
PID_controller = PID(Kp = Kp, Ti = Ti, Td = Td,
                  N = 5, Te = 1, umax = max(up_max, ur_max/ratio), umin = 0)

#init state estimator
George_nominal = Patient.Patient(age, height, weight, gender)
BIS_param_nominal =  George_nominal.BisPD.BIS_param
Ap = George_nominal.PropoPK.A
Ar = George_nominal.RemiPK.A
Bp = George_nominal.PropoPK.B
Br = George_nominal.RemiPK.B
A = block_diag(Ap,Ar)
B = block_diag(Bp,Br)

Q = Q_continuous_white_noise(4, spectral_density = 1, block_size = 2)
P0 = np.diag([100,100,100,1000]*2)
estimator = EKF(A, B, BIS_param = BIS_param_nominal, ts = 1,
                P0 = P0, R = 1e3, Q = Q, x0 = np.ones(8)*0)


N_simu = 30*60
BIS = np.zeros(N_simu)
BIS_EKF = np.zeros(N_simu)
MAP = np.zeros(N_simu)
CO = np.zeros(N_simu)
Up = np.zeros(N_simu)
Ur = np.zeros(N_simu)
Xp = np.ones((4, N_simu))
Xr = np.ones((4, N_simu))
Xp_EKF = np.zeros((4, N_simu))*1
Xr_EKF = np.zeros((4, N_simu))*1
uP = 0
for i in range(N_simu):
    # if i == 100:
    #     print("break")

    uR = min(ur_max,max(0,uP*ratio)) #+ 0.5*max(0,np.sin(i/15)) + 0.2*int(i>200)
    uP = min(up_max,max(0,uP)) #+ 0.5*max(0,np.cos(i/8)) + 0.1*int(i>400)
    Dist = disturbances.compute_disturbances(i, 'step')
    Bis, Co, Map = George.one_step(uP, uR, Dist = Dist, noise = True)
    Xp[:,i] =  George.PropoPK.x.T[0]       
    Xr[:,i] =  George.RemiPK.x.T[0]          
    
    BIS[i] = min(100,Bis)
    MAP[i] = Map[0,0]
    CO[i] = Co[0,0]
    Up[i] = uP
    Ur[i] = uR
    #estimation
    X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
    Xp_EKF[:,i] =  X[:4]       
    Xr_EKF[:,i] =  X[4:]        
    uP = PID_controller.one_step(BIS_EKF[i], BIS_cible) 

    
fig, axs = plt.subplots(8,figsize=(14,16))
for i in range(4):
    axs[i].plot(Xp[i,:], '-')
    axs[i].plot(Xp_EKF[i,:], '-')
    axs[i].set(xlabel='t', ylabel='$xp_' + str(i+1) + '$')
    plt.grid()
    axs[i+4].plot(Xr[i,:], '-')
    axs[i+4].plot(Xr_EKF[i,:], '-')
    axs[i+4].set(xlabel='t', ylabel='$xr_' + str(i+1) + '$')
plt.show()


plt.plot(Up, label = 'Propofol')
plt.plot(Ur, label = 'Remifentanil')
plt.grid()
plt.legend()
plt.show()


plt.plot(BIS, label = 'Measure')
plt.plot(BIS_EKF, label = 'Estimation')
plt.grid()
plt.legend()
plt.show()

