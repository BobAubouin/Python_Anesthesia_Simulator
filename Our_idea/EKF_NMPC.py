#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:51:29 2022

@author: aubouinb
"""

from bokeh.io import export_svg
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from scipy.linalg import block_diag
from filterpy.common import Q_continuous_white_noise
from functools import partial
import pandas as pd
import numpy as np
import multiprocessing
import time
from Estimators import linear_Kalman, EKF_extended, EKF
from Controller import NMPC, MPC, MPC_lin
import Patient
import disturbances
import metrics


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
        age, height, weight, gender, BIS_param=[None]*6, Te=ts)  #
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    BIS_param_nominal[4] = George.BisPD.BIS_param[4]

    Ap = George_nominal.PropoPK.A
    Ar = George_nominal.RemiPK.A
    Bp = George_nominal.PropoPK.B
    Br = George_nominal.RemiPK.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # init state estimator
    Q = Q_continuous_white_noise(4, spectral_density=1, block_size=2)
    P0 = np.diag([10, 10, 10, 10]*2)
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
    dup_max = 0.2*ts*100
    dur_max = 0.4*ts*100

    MPC_controller = NMPC(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts, N=N_mpc, Nu=Nu_mpc,
                          R=R_mpc, umax=[up_max, ur_max], dumax=[dup_max, dur_max],
                          dumin=[-dup_max, - dur_max], dymin=0, ki=0)

    # MPC_controller = MPC(A_nom, B_nom, BIS_param = BIS_param_nominal, ts = ts, N = N_mpc, Nu = Nu_mpc,
    #                       Q = Q_mpc, R = R_mpc, umax = [up_max, ur_max], dumax = [dup_max, dur_max],
    #                       dumin = [-dup_max, - dur_max], ymin = 0, ki = 0)

    if style == 'induction':
        N_simu = int(5/ts)*60
        BIS = np.zeros(N_simu)
        BIS_Pred = np.zeros((N_simu, N_mpc))
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
            X_real = np.concatenate((Xp[:, i], Xr[:, i]), axis=0)
            # X = X_real
            uP, uR, bis = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
            BIS_Pred[i, :] = bis
            BIS_cible_MPC[i] = MPC_controller.internal_target

    elif style == 'total':
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
            X_real = np.concatenate((Xp[:, i], Xr[:, i]), axis=0)
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])

    error = BIS - BIS_cible
    for i in range(len(error)):
        if error[i] < 0:
            error[i] = error[i]*2
    IAE = np.sum(np.abs(error))
    if False:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.arange(0, len(BIS))*5/60, BIS)
        for i in range(N_simu):
            ax[0].plot(np.arange(0, len(BIS_Pred[i, :]))
                       * 5/60 + (i+1)*5/60, BIS_Pred[i, :])
        ax[0].grid()
        ax[0].set_xlim(0, 12)

        ax[1].plot(np.arange(0, len(BIS))*5/60, Up)
        ax[1].plot(np.arange(0, len(BIS))*5/60, Ur)
        ax[1].grid()
        ax[1].set_xlim(0, 12)

    return IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF, BIS_EKF], George.BisPD.BIS_param


# %% Table simultation
# Patient table:
#index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 4.73, 24.97,  1.08,  0.3, 97.86, 89.62],
                 [2,  36, 163, 50, 0, 4.43, 19.33,  1.16,  0.29, 89.1, 98.86],
                 [3,  28, 164, 52, 0, 4.81, 16.89,  1.54,  0.14, 93.66, 94.],
                 [4,  50, 163, 83, 0, 3.86, 20.97,  1.37,  0.12, 94.6, 93.2],
                 [5,  28, 164, 60, 1, 5.22, 18.95,  1.21,  0.68, 97.43, 96.21],
                 [6,  43, 163, 59, 0, 3.41, 23.26,  1.34,  0.58, 85.33, 97.07],
                 [7,  37, 187, 75, 1, 4.83, 15.21,  1.84,  0.13, 91.87, 90.84],
                 [8,  38, 174, 80, 0, 4.36, 13.86,  2.23,  1.05, 97.45, 96.36],
                 [9,  41, 170, 70, 0, 4.57, 16.20,  1.69,  0.16, 85.83, 94.6],
                 [10, 37, 167, 58, 0, 6.02, 23.47,  1.27,  0.77, 95.18, 88.17],
                 [11, 42, 179, 78, 1, 3.79, 22.25,  2.35,  1.12, 98.02, 96.95],
                 [12, 34, 172, 58, 0, 5.70, 18.64,  2.02,  0.4, 99.57, 96.94],
                 [13, 38, 169, 65, 0, 4.64, 19.50,  1.43,  0.48, 93.82, 94.40]]

# df_patiens = pd.DataFrame(Patient_table)
# print(df_patiens.to_latex(index=False, header = ['index', 'Age (y)', 'Height (cm)', 'Weight (kg)',
#                                                   'Gender', '$C_{50p}$', '$C_{50r}$', '$\gamma$', '$\beta$', '$E_0$', '$E_{max}$']))


# Simulation parameters
phase = 'induction'

IAE_list = []
TT_list = []
ST10_list = []
p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)
p4 = figure(width=500, height=500)

# param_opti = pd.read_csv('optimal_parameters_MPC.csv')
# param_opti = [int(param_opti['N']), int(param_opti['Nu']),
#               float(param_opti['R']), float(param_opti['ki'])]
# # param_opti = [20, 10,  2.6,  0.01]
# MPC_param = [param_opti[0], param_opti[1], 10 **
#              param_opti[2]*np.diag([10, 1]), param_opti[3]]
MPC_param = [20, 20, 10**(1.5)*np.diag([3, 1]), 1e-2]
t0 = time.time()
for i in range(1, 14):
    Patient_info = Patient_table[i-1][1:]
    IAE, data, BIS_param = simu(Patient_info, phase, MPC_param)
    source = pd.DataFrame(data=data[0], columns=['BIS'])
    source.insert(len(source.columns), "time", np.arange(0, len(data[0]))*5/60)
    source.insert(len(source.columns), "Ce50_P", BIS_param[0])
    source.insert(len(source.columns), "Ce50_R", BIS_param[1])
    source.insert(len(source.columns), "gamma", BIS_param[2])
    source.insert(len(source.columns), "beta", BIS_param[3])
    source.insert(len(source.columns), "E0", BIS_param[4])
    source.insert(len(source.columns), "Emax", BIS_param[5])

    plot = p1.line(x='time', y='BIS', source=source)
    tooltips = [('Ce50_P', "@Ce50_P"), ('Ce50_R', "@Ce50_R"),
                ('gamma', "@gamma"), ('beta', "@beta"),
                ('E0', "@E0"), ('Emax', "@Emax")]
    p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
    p1.line(np.arange(0, len(data[0]))*5/60, data[5],
            legend_label='internal target', line_color="#f46d43")
    # p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
    p2.line(np.arange(0, len(data[0]))*5/60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0, len(data[0]))*5/60, data[2]*10,
            legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0, len(data[3]))*5/60, data[3],
            line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0, len(data[4]))*5/60, data[4],
            line_color="#f46d43", legend_label='remifentanil (ng/min)')
    p4.line(data[6][3], data[7][3])
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
        data[0], Te=5, phase=phase)
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
grid = row(column(p3, p1, p2), p4)

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean ST10 : " + str(np.round(np.nanmean(ST10_list), 2)))
print("Min ST10 : " + str(np.round(np.nanmin(ST10_list), 2)))
print("Max ST10 : " + str(np.round(np.nanmax(ST10_list), 2)))

# %% Inter patient variability
# Simulation parameter
phase = 'induction'
Number_of_patient = 32
MPC_param = [35, 5, 1e-6, 1e6*np.diag([10, 1]), 0.5]

IAE_list = []
TT_list = []
ST10_list = []
ST20_list = []
US_list = []
BIS_NADIR_list = []

p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)


def one_simu(x, i):
    '''cost of one simulation, i is the patient index'''
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None]*6
    iae, data, BIS_param = simu(Patient_info, 'induction', [int(x[0]), int(x[1]), 10**(x[2])*np.diag([10, 1]), x[3]],
                                random_PK=False, random_PD=True,)
    return [iae, data, BIS_param, i]


param_opti = pd.read_csv('optimal_parameters_MPC.csv')
x = param_opti.to_numpy()
x = x[0, 1:]
x[0] = 30
x[1] = 30
x[2] = 1
x[3] = 1e-2
# x = [30,  30,  -6,  0]
pool_obj = multiprocessing.Pool(8)
func = partial(one_simu, x)
result = pool_obj.map(func, range(0, Number_of_patient))
pool_obj.close()
pool_obj.join()

# print([r[0] for r in result])

for i in range(Number_of_patient):
    print(i)
    IAE = result[i][0]
    data = result[i][1]
    BIS_param = result[i][2]

    source = pd.DataFrame(data=data[0], columns=['BIS'])
    source.insert(len(source.columns), "time", np.arange(0, len(data[0]))*5/60)
    source.insert(len(source.columns), "Ce50_P", BIS_param[0])
    source.insert(len(source.columns), "Ce50_R", BIS_param[1])
    source.insert(len(source.columns), "gamma", BIS_param[2])
    source.insert(len(source.columns), "beta", BIS_param[3])
    source.insert(len(source.columns), "E0", BIS_param[4])
    source.insert(len(source.columns), "Emax", BIS_param[5])

    plot = p1.line(x='time', y='BIS', source=source)
    tooltips = [('Ce50_P', "@Ce50_P"), ('Ce50_R', "@Ce50_R"),
                ('gamma', "@gamma"), ('beta', "@beta"),
                ('E0', "@E0"), ('Emax', "@Emax")]
    p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
    p1.line(np.arange(0, len(data[0]))*5/60, data[8],
            legend_label='internal target', line_color="#f46d43")
    # p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
    p2.line(np.arange(0, len(data[0]))*5/60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0, len(data[0]))*5/60, data[2]*10,
            legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0, len(data[3]))*5/60, data[3],
            line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0, len(data[4]))*5/60, data[4],
            line_color="#f46d43", legend_label='remifentanil (ng/min)')
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
        data[0], Te=5, phase=phase)
    TT_list.append(TT)
    BIS_NADIR_list.append(BIS_NADIR)
    ST10_list.append(ST10)
    ST20_list.append(ST20)
    US_list.append(US)
    IAE_list.append(IAE)

p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3, p1, p2))

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean ST10 : " + str(np.round(np.nanmean(ST10_list), 2)))
print("Min ST10 : " + str(np.round(np.nanmin(ST10_list), 2)))
print("Max ST10 : " + str(np.round(np.nanmax(ST10_list), 2)))

result_table = pd.DataFrame()
result_table.insert(len(result_table.columns), "", ['mean', 'std', 'min', 'max'])
result_table.insert(len(result_table.columns), "TT (min)", [np.round(np.nanmean(TT_list), 2),
                                                            np.round(
                                                                np.nanstd(TT_list), 2),
                                                            np.round(
                                                                np.nanmin(TT_list), 2),
                                                            np.round(np.nanmax(TT_list), 2)])
result_table.insert(len(result_table.columns), "BIS_NADIR", [np.round(np.nanmean(BIS_NADIR_list), 2),
                                                             np.round(
                                                                 np.nanstd(BIS_NADIR_list), 2),
                                                             np.round(
                                                                 np.nanmin(BIS_NADIR_list), 2),
                                                             np.round(np.nanmax(BIS_NADIR_list), 2)])
result_table.insert(len(result_table.columns), "ST10 (min)", [np.round(np.nanmean(ST10_list), 2),
                                                              np.round(
                                                                  np.nanstd(ST10_list), 2),
                                                              np.round(
                                                                  np.nanmin(ST10_list), 2),
                                                              np.round(np.nanmax(ST10_list), 2)])
result_table.insert(len(result_table.columns), "ST20 (min)", [np.round(np.nanmean(ST20_list), 2),
                                                              np.round(
                                                                  np.nanstd(ST20_list), 2),
                                                              np.round(
                                                                  np.nanmin(ST20_list), 2),
                                                              np.round(np.nanmax(ST20_list), 2)])
result_table.insert(len(result_table.columns), "US", [np.round(np.nanmean(US_list), 2),
                                                      np.round(np.nanstd(US_list), 2),
                                                      np.round(np.nanmin(US_list), 2),
                                                      np.round(np.nanmax(US_list), 2)])

print(result_table.to_latex(index=False))

p1.output_backend = "svg"
export_svg(p1, filename="BIS_NMPC.svg")
p3.output_backend = "svg"
export_svg(p3, filename="input_NMPC.svg")
