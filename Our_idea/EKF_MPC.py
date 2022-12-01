#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:44:46 2022

@author: aubouinb
"""

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
from Controller import NMPC, MPC
from Estimators import EKF

import time
import numpy as np
import pandas as pd
from filterpy.common import Q_continuous_white_noise
from scipy.linalg import block_diag
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import HoverTool
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
    
    ts = 5
    
    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = Patient.Patient(age, height, weight, gender, BIS_param = BIS_param,
                             Random_PK = random_PK, Random_PD = random_PD, Te = ts)#, model_propo = 'Eleveld', model_remi = 'Eleveld')

    #Nominal parameters
    George_nominal = Patient.Patient(age, height, weight, gender, BIS_param = [None]*6, Te = ts)
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    BIS_param_nominal[4] = George.BisPD.BIS_param[4]
    
    Ap = George_nominal.PropoPK.A
    Ar = George_nominal.RemiPK.A
    Bp = George_nominal.PropoPK.B
    Br = George_nominal.RemiPK.B
    A_nom = block_diag(Ap,Ar)
    B_nom = block_diag(Bp,Br)
    
    #init state estimator
    Q = Q_continuous_white_noise(4, spectral_density = 1, block_size = 2)
    P0 = np.diag([10,10,10,10]*2)
    estimator = EKF(A_nom, B_nom, BIS_param = BIS_param_nominal, ts = ts,
                    P0 = P0, R = 10, Q = Q)

    #init controller
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    Q_mpc = MPC_param[2]
    R_mpc = MPC_param[3]
    ki_mpc = MPC_param[4]
    BIS_cible = 50
    up_max = 6.67*ts
    ur_max = 16.67*ts
    dup_max = 0.2*ts
    dur_max = 0.4*ts
    
    # MPC_controller = NMPC(A_nom, B_nom, BIS_param = BIS_param_nominal, ts = ts, N = N_mpc, Nu = Nu_mpc,
    #                       Q = Q_mpc, R = R_mpc, umax = [up_max, ur_max], dumax = [dup_max, dur_max], 
    #                       dumin = [-dup_max, - dur_max], dymin = 0, ki = 0)

    MPC_controller = MPC(A_nom, B_nom, BIS_param = BIS_param_nominal, ts = ts, N = N_mpc, Nu = Nu_mpc,
                          Q = Q_mpc, R = R_mpc, umax = [up_max, ur_max], dumax = [dup_max, dur_max], 
                          dumin = [-dup_max, - dur_max], ymin = 0, ki = 0)


    if style=='induction':
        N_simu = int(10/ts)*60
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
            X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
            Xp_EKF[:,i] =  X[:4]       
            Xr_EKF[:,i] =  X[4:]
            # X_MPC = np.concatenate((Xp[:,i],Xr[:,i]),axis = 0)
            if  i==20 : #or (BIS_EKF[i]<50 and MPC_controller.ki==0):
                MPC_controller.ki = ki_mpc
            X = np.clip(X, a_min=0, a_max = 1e10)
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
            BIS_cible_MPC[i] = MPC_controller.internal_target
            
            
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
        L = np.zeros(N_simu)
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
            X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
            Xp_EKF[:,i] =  X[:4]       
            Xr_EKF[:,i] =  X[4:]        
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
            
    IAE = np.sum(np.abs(BIS - BIS_cible))
    return IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF], George.BisPD.BIS_param

#%% Table simultation
#Patient table:
#index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 3.8, 18.3, 1.00, 0, 91.1, 94.3],
                 [2,  36, 163, 50, 0, 3.2, 28.6,  1.3, 0, 92.6, 100],
                 [3,  28, 164, 52, 0, 3.1,  4.9,  1, 0,   97, 94],
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

#Simulation parameters
phase = 'induction'

IAE_list = []
TT_list = []
ST10_list = []
p1 = figure(plot_width=900, plot_height=300)
p2 = figure(plot_width=900, plot_height=300)
p3 = figure(plot_width=900, plot_height=300)
p4 = figure(plot_width=500, plot_height=500)

param_opti = [30, 30, 0.05, 2e-2]
MPC_param = [param_opti[0], param_opti[1], 1000, 10**param_opti[2]*np.diag([1,1]), param_opti[3]]
t0 = time.time()
for i in range(1,14):
    Patient_info = Patient_table[i-1][1:]
    IAE, data, BIS_param = simu(Patient_info, phase, MPC_param)
    source = pd.DataFrame(data = data[0], columns = ['BIS'])
    source.insert(len(source.columns),"time", np.arange(0,len(data[0]))*5/60)
    source.insert(len(source.columns),"Ce50_P", BIS_param[0])
    source.insert(len(source.columns),"Ce50_R", BIS_param[1])
    source.insert(len(source.columns),"gamma", BIS_param[2])
    source.insert(len(source.columns),"beta", BIS_param[3])
    source.insert(len(source.columns),"E0", BIS_param[4])
    source.insert(len(source.columns),"Emax", BIS_param[5])
    
    plot = p1.line(x = 'time', y = 'BIS', source = source)
    tooltips = [('Ce50_P',"@Ce50_P"), ('Ce50_R',"@Ce50_R"),
                ('gamma',"@gamma"), ('beta',"@beta"),
                ('E0',"@E0"), ('Emax',"@Emax")]
    p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
    p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
    p2.line(np.arange(0,len(data[0]))*5/60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0,len(data[0]))*5/60, data[2]*10, legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0,len(data[3]))*5/60, data[3], line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0,len(data[4]))*5/60, data[4], line_color="#f46d43", legend_label='remifentanil (ng/min)')
    p4.line(data[6][3], data[7][3])
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(data[0], Te = 5, phase = phase)
    TT_list.append(TT)
    ST10_list.append(ST10)
    IAE_list.append(IAE)
    
t1 = time.time()
print('one_step time: ' + str((t1-t0)/(13*60))) 
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p1.xaxis.axis_label = 'Time (min)'
p4.xaxis.axis_label = 'Ce_propo (µg/ml)'
p4.yaxis.axis_label = 'Ce_remi (ng/ml)'
p2.xaxis.axis_label = 'Time (min)'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3,p1,p2), p4)

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean ST10 : " + str(np.round(np.nanmean(ST10_list),2)))
print("Min ST10 : " + str(np.round(np.nanmin(ST10_list),2)))
print("Max ST10 : " + str(np.round(np.nanmax(ST10_list),2)))

#%% Inter patient variability

#Simulation parameter
phase = 'induction'
Number_of_patient = 10
param_opti = [30, 30, 2.6, 2e-2]
MPC_param = [param_opti[0], param_opti[1], 1, 10**param_opti[2]*np.diag([10,1]), param_opti[3]]
IAE_list = []
TT_list = []
p1 = figure(plot_width=900, plot_height=300)
p2 = figure(plot_width=900, plot_height=300)
p3 = figure(plot_width=900, plot_height=300)

for i in range(Number_of_patient):
    #Generate random patient information with uniform distribution
    age = np.random.randint(low=18,high=70)
    height = np.random.randint(low=150,high=190)
    weight = np.random.randint(low=50,high=100)
    gender = np.random.randint(low=0,high=1)

    Patient_info = [age, height, weight, gender] + [None]*6
    IAE, data, BIS_param = simu(Patient_info, phase, MPC_param, random_PD = True)
    source = pd.DataFrame(data = data[0], columns = ['BIS'])
    source.insert(len(source.columns),"time", np.arange(0,len(data[0]))*5/60)
    source.insert(len(source.columns),"Ce50_P", BIS_param[0])
    source.insert(len(source.columns),"Ce50_R", BIS_param[1])
    source.insert(len(source.columns),"gamma", BIS_param[2])
    source.insert(len(source.columns),"beta", BIS_param[3])
    source.insert(len(source.columns),"E0", BIS_param[4])
    source.insert(len(source.columns),"Emax", BIS_param[5])
    
    plot = p1.line(x = 'time', y = 'BIS', source = source)
    tooltips = [('Ce50_P',"@Ce50_P"), ('Ce50_R',"@Ce50_R"),
                ('gamma',"@gamma"), ('beta',"@beta"),
                ('E0',"@E0"), ('Emax',"@Emax")]
    p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
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

