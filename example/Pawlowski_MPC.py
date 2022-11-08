#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:44:46 2022

@author: aubouinb
"""


import sys
import os 
path = os.getcwd()
path_root = path[:-7]
sys.path.append(str(path_root))
from src.PAS import Patient, disturbances, metrics
from Controller import GPC
from Estimators import EKF


import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
import matplotlib.pyplot as plt

def simu(Patient_info: list,style: str, MPC_param: list, random_PK: bool = False, random_PD: bool = False):
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
    
    ts = 1
    
    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = Patient.Patient(age, height, weight, gender, BIS_param = BIS_param,
                             Random_PK = random_PK, Random_PD = random_PD, Te = ts)#, model_propo = 'Eleveld', model_remi = 'Eleveld')

    #Nominal parameters
    George_nominal = Patient.Patient(age, height, weight, gender, BIS_param = BIS_param, Te = ts)
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    
    Ap = George_nominal.PropoPK.A_d
    Ar = George_nominal.RemiPK.A_d
    Bp = George_nominal.PropoPK.B_d
    Br = George_nominal.RemiPK.B_d

    #init controller
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    lambda_param = MPC_param[2]

    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    K = 2
    MPC_controller = GPC(Ap, Bp, Ar, Br, BIS_param = BIS_param_nominal, ts = ts, N = N_mpc, Nu = Nu_mpc,
                          lambda_param = lambda_param, umax = min(up_max, ur_max/K))




    if style=='induction':
        N_simu = int(30/ts)*60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Yp_cible = np.zeros(N_simu)
        Yr_estimate = np.zeros(N_simu)
        Yp_estimate = np.zeros(N_simu)
        uP = 0
        uR = 0
        for i in range(N_simu):

            Dist = disturbances.compute_disturbances(i*ts, 'null')
            Bis, Co, Map = George.one_step(uP, uR, Dist = Dist, noise = False)
            Xp[:,i] =  George.PropoPK.x.T[0]       
            Xr[:,i] =  George.RemiPK.x.T[0]          

            BIS[i] = min(100,Bis)
            MAP[i] = Map[0,0]
            CO[i] = Co[0,0]
            Up[i] = uP
            Ur[i] = uR
            #estimation
            uP, uR = MPC_controller.one_step(BIS[i], BIS_cible)         
            Yp_cible[i] = MPC_controller.Yp_target
            Yr_estimate[i] = MPC_controller.Yr_list[0]
            Yp_estimate[i] = MPC_controller.Yp_list_vague[0]
    elif style=='total':
        N_simu = int(60/ts)*60
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
        uP = 1
        uR = 1
        for i in range(N_simu):
            # if i == 100:
            #     print("break")

            Dist = disturbances.compute_disturbances(i*ts, 'realistic')
            Bis, Co, Map = George.one_step(uP, uR, Dist = Dist, noise = True)
            Xp[:,i] =  George.PropoPK.x.T[0]       
            Xr[:,i] =  George.RemiPK.x.T[0]          

            BIS[i] = min(100,Bis)
            MAP[i] = Map[0,0]
            CO[i] = Co[0,0]
            Up[i] = uP
            Ur[i] = uR
            #estimation
            uP, uR = MPC_controller.one_step(BIS[i], BIS_cible) 
      
    plt.plot(Yp_cible)
    plt.plot(Yr_estimate)
    plt.plot(Yp_estimate)
    plt.show()
    IAE = np.sum(np.abs(BIS - BIS_cible))
    return IAE, [BIS, MAP, CO, Up, Ur]

#%% Patient table

#index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 3.8, 18.3, 1.00, 0, 91.1, 94.3],
                 [2,  36, 163, 50, 0, 3.2, 28.6,  1.3, 0, 92.6, 100],
                 [3,  28, 164, 52, 0, 3.1,  4.9,    1, 0,   97, 94],
                 [4,  50, 163, 83, 0, 5.1, 20.5,  0.7, 0, 97.9, 98.3],
                 [5,  28, 164, 60, 1, 6.5, 30.1,  1.6, 0,  100,  86.7],
                 [6,  43, 163, 59, 0, 6.2, 23.1,  0.6, 0,   80, 88.6],
                 [7,  37, 187, 75, 1, 5. , 16.7,  1.8, 0,   97, 94.8],
                 [8,  38, 174, 80, 0, 2.7, 15.3,  1.6, 0, 97.2, 92.8],
                 [9,  41, 170, 70, 0, 3.5, 15.5,  0.6, 0,  100,  95.3],
                 [10, 37, 167, 58, 0, 4.5, 14.8,  1.5, 0,  100,  89. ],
                 [11, 42, 179, 78, 1, 6.3,   22,  1.7, 0, 97.6, 85.9],
                 [12, 34, 172, 58, 0, 6.2, 18.5,  1.7, 0, 90.4, 100],
                 [13, 38, 169, 65, 0, 2.8, 14.6,  1.6, 0, 82.9, 98.7]]

phase = 'induction'
MPC_param = [36, 34, 10]


IAE_list = []
TT_list = []
p1 = figure(plot_width=900, plot_height=300)
p2 = figure(plot_width=900, plot_height=300)
p3 = figure(plot_width=900, plot_height=300)
for i in range(1,14):
    Patient_info = Patient_table[i-1][1:]
    IAE, data = simu(Patient_info, phase, MPC_param)
    p1.line(np.arange(0,len(data[0]))/60, data[0])
    p2.line(np.arange(0,len(data[0]))/60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0,len(data[0]))/60, data[2]*10, legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0,len(data[3]))/60, data[3], line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0,len(data[4]))/60, data[4], line_color="#f46d43", legend_label='remifentanil (ng/min)')
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(data[0], Te = 1, phase = phase)
    TT_list.append(TT)
    IAE_list.append(IAE)
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p1.xaxis.axis_label = 'Time (min)'
p2.xaxis.axis_label = 'Time (min)'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3,p1,p2))

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean TT : " + str(np.mean(TT_list)))
print("Min TT : " + str(np.min(TT_list)))
print("Max TT : " + str(np.max(TT_list)))

#%% Inter-Variability

#Simulation parameter
phase = 'induction'
Number_of_patient = 20
MPC_param = [36, 34, 10]

IAE_list = []
TT_list = []
p1 = figure(plot_width=900, plot_height=300)
p2 = figure(plot_width=900, plot_height=300)
p3 = figure(plot_width=900, plot_height=300)

for i in range(Number_of_patient):
    #Generate random patient information with uniform distribution
    age = 30 #np.random.randint(low=18,high=70)
    height = 175#np.random.randint(low=150,high=190)
    weight = 75#np.random.randint(low=50,high=100)
    gender = 1#np.random.randint(low=0,high=1)

    Patient_info = [age, height, weight, gender] + [None]*6
    IAE, data = simu(Patient_info, phase, MPC_param, random_PD = True)
    p1.line(np.arange(0,len(data[0]))*5/60, data[0])
    # p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
    p2.line(np.arange(0,len(data[0]))*5/60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0,len(data[0]))*5/60, data[2]*10, legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0,len(data[3]))*5/60, data[3], line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0,len(data[4]))*5/60, data[4], line_color="#f46d43", legend_label='remifentanil (ng/min)')
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(data[0], Te = 5, phase = phase)
    TT_list.append(TT)
    IAE_list.append(IAE)
    
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3,p1,p2))

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean TT : " + str(np.mean(TT_list)))
print("Min TT : " + str(np.min(TT_list)))
print("Max TT : " + str(np.max(TT_list)))

