#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:32:55 2022

@author: aubouinb
"""

"""MISO PID control of anesthesia from "Individualized PID tuning for maintenance of general anesthesia with
propofol and remifentanil coadministration" Michele Schiavo, Fabrizio Padula, Nicola Latronico, 
Massimiliano Paltenghi,Antonio Visioli 2022"""

# Since the file is on a subfolder of the git we need to add the main folder to the path
from bokeh.io import export_svg
from bokeh.models import HoverTool
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
import casadi as cas
from pyswarm import pso
import pandas as pd
import numpy as np
from src.Control.Controller import PID
from src.PAS import Patient, disturbances, metrics
from pathlib import Path
import sys
import time
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


# Patient table:
#index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1, 40, 163, 54, 0, 4.73, 24.97, 1.08, 0.30, 97.86, 89.62],
                 [2, 36, 163, 50, 0, 4.43, 19.33, 1.16, 0.29, 89.10, 98.86],
                 [3, 28, 164, 52, 0, 4.81, 16.89, 1.54, 0.14, 93.66, 94.],
                 [4, 50, 163, 83, 0, 3.86, 20.97, 1.37, 0.12, 94.60, 93.2],
                 [5, 28, 164, 60, 1, 5.22, 18.95, 1.21, 0.68, 97.43, 96.21],
                 [6, 43, 163, 59, 0, 3.41, 23.26, 1.34, 0.58, 85.33, 97.07],
                 [7, 37, 187, 75, 1, 4.83, 15.21, 1.84, 0.13, 91.87, 90.84],
                 [8, 38, 174, 80, 0, 4.36, 13.86, 2.23, 1.05, 97.45, 96.36],
                 [9, 41, 170, 70, 0, 4.57, 16.20, 1.69, 0.16, 85.83, 94.6],
                 [10, 37, 167, 58, 0, 6.02, 23.47, 1.27, 0.77, 95.18, 88.17],
                 [11, 42, 179, 78, 1, 3.79, 22.25, 2.35, 1.12, 98.02, 96.95],
                 [12, 34, 172, 58, 0, 5.70, 18.64, 2.02, 0.40, 99.57, 96.94],
                 [13, 38, 169, 65, 0, 4.64, 19.50, 1.43, 0.48, 93.82, 94.40]]


def simu(Patient_info: list, style: str, PID_param: list, random_PK: bool = False, random_PD: bool = False):
    ''' This function perform a closed-loop Propofol-Remifentanil anesthesia simulation with a PID controller.

    Inputs: - Patient_info: list of patient informations, Patient_info = [Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax]
            - style: either 'induction' or 'maintenance' to describe the phase to simulate
            - PID_param: parameter of the PID controller P = [Kp, Ti, Td, ratio]
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

    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = Patient.Patient(age, height, weight, gender,
                             BIS_param=BIS_param, Random_PK=random_PK, Random_PD=random_PD)

    ts = 1
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    ratio = PID_param[3]
    PID_controller = PID(Kp=PID_param[0], Ti=PID_param[1], Td=PID_param[2],
                         N=5, Te=1, umax=max(up_max, ur_max / ratio), umin=0)

    if style == 'induction':
        N_simu = 10 * 60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        for i in range(N_simu):
            # if i == 100:
            #     print("break")

            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Bis, Co, Map = George.one_step(uP, uR, noise=False)
            BIS[i] = Bis
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(Bis, BIS_cible)
    elif style == 'total':
        N_simu = int(60 / ts) * 60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        for i in range(N_simu):
            # if i == 100:
            #     print("break")
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Dist = disturbances.compute_disturbances(i * ts, 'realistic')
            Bis, Co, Map = George.one_step(uP, uR, Dist=Dist, noise=False)

            BIS[i] = min(100, Bis)
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(Bis, BIS_cible)

    elif style == 'maintenance':
        N_simu = 25 * 60  # 25 minutes
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)

        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        Ap = np.array(George.PropoPK.A)
        Bp = np.array(George.PropoPK.B)
        Ar = np.array(George.RemiPK.A)
        Br = np.array(George.RemiPK.B)

        x0p = np.linalg.solve(Ap, Bp * up_max / 20)
        x0r = np.linalg.solve(Ar, Br * up_max / 10)
        xp = cas.MX.sym('xp', 4)
        w0 += x0p[:, 0].tolist()
        xr = cas.MX.sym('xr', 4)
        w0 += x0r[:, 0].tolist()
        UP = cas.MX.sym('up', 1)
        w = [xp, xr, UP]
        w0 += [up_max / 2]
        lbw = [1e-6] * 9
        ubw = [1e4] * 9

        up = xp[3] / Ce50p
        ur = xr[3] / Ce50r
        Phi = up / (up + ur)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur) / U_50
        J = (50 - (E0 - Emax * i ** gamma / (1 + i ** gamma)))**2

        g = [Ap @ xp + Bp * UP, Ar @ xr + Br * (ratio * UP)]
        lbg = [-1e-8] * 8
        ubg = [1e-8] * 8

        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, {
                            'ipopt.max_iter': 300, 'verbose': False},)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # set the patien at the equilibrium point
        George.PropoPK.x = np.expand_dims(np.array(w_opt[:4]), axis=1)
        George.RemiPK.x = np.expand_dims(np.array(w_opt[4:8]), axis=1)
        George.Hemo.CeP = w_opt[3]
        George.Hemo.CeR = w_opt[7]
        uP = w_opt[-1]
        # initialize the PID at the equilibriium point
        PID_controller.integral_part = uP / PID_controller.Kp
        PID_controller.last_BIS = 50
        for i in range(N_simu):
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Bis, Co, Map = George.one_step(uP, uR, noise=False)
            dist_bis, dist_map, dist_co = disturbances.compute_disturbances(i, 'step')
            BIS[i] = min(100, Bis) + dist_bis
            MAP[i] = Map[0, 0] + dist_map
            CO[i] = Co[0, 0] + dist_co
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(BIS[i], BIS_cible)

    IAE = np.sum(np.abs(BIS - BIS_cible))
    return IAE, [BIS, MAP, CO, Up, Ur]


# %% PSO

def cost(x, ratio):
    '''cost of the optimization, x is the vector of the PID controller
    x = [Kp, Ti, Td]
    IAE is the maximum integrated absolut error over the patient population'''
    IAE = []
    for i in range(13):
        Patient_info = Patient_table[i - 1][1:]
        iae, data = simu(Patient_info, 'induction', [x[0], x[1], x[2], ratio])
        IAE.append(iae)
    return max(IAE)


try:
    param_opti = pd.read_csv('optimal_parameters_PID.csv')
except:
    param_opti = pd.DataFrame(columns=['ratio', 'Kp', 'Ti', 'Td'])
    for ratio in range(2, 3):
        def local_cost(x): return cost(x, ratio)
        lb = [1e-6, 100, 10]
        ub = [1, 300, 40]
        # Default: 100 particles as in the article
        xopt, fopt = pso(local_cost, lb, ub, debug=True, minfunc=1e-2)
        param_opti = pd.concat((param_opti, pd.DataFrame(
            {'ratio': ratio, 'Kp': xopt[0], 'Ti': xopt[1], 'Td': xopt[2]}, index=[0])), ignore_index=True)
        print(ratio)

    param_opti.to_csv('optimal_parameters_PID.csv')

# %%test on patient table


# phase = 'induction'


# IAE_list = []
# TT_list = []
# p1 = figure(width=900, height=300)
# p2 = figure(width=900, height=300)
# p3 = figure(width=900, height=300)

# for ratio in range(2, 3):
#     print('ratio = ' + str(ratio))
#     Kp = float(param_opti.loc[param_opti['ratio'] == ratio, 'Kp'])
#     Ti = float(param_opti.loc[param_opti['ratio'] == ratio, 'Ti'])
#     Td = float(param_opti.loc[param_opti['ratio'] == ratio, 'Td'])
#     PID_param = [Kp, Ti, Td, ratio]
#     for i in range(1, 14):
#         Patient_info = Patient_table[i-1][1:]
#         IAE, data = simu(Patient_info, phase, PID_param)
#         p1.line(np.arange(0, len(data[0]))/60, data[0])
#         p2.line(np.arange(0, len(data[0]))/60, data[1], legend_label='MAP (mmgh)')
#         p2.line(np.arange(0, len(data[0]))/60, data[2]*10,
#                 legend_label='CO (cL/min)', line_color="#f46d43")
#         p3.line(np.arange(0, len(data[3]))/60, data[3],
#                 line_color="#006d43", legend_label='propofol (mg/min)')
#         p3.line(np.arange(0, len(data[4]))/60, data[4],
#                 line_color="#f46d43", legend_label='remifentanil (ng/min)')
#         TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
#             data[0], Te=1, phase=phase)
#         TT_list.append(TT)
#         IAE_list.append(IAE)
# p1.title.text = 'BIS'
# p3.title.text = 'Infusion rates'
# p3.xaxis.axis_label = 'Time (min)'
# grid = row(column(p3, p1, p2))

# show(grid)

# print("Mean IAE : " + str(np.mean(IAE_list)))
# print("Mean TT : " + str(np.mean(TT_list)))
# print("Min TT : " + str(np.min(TT_list)))
# print("Max TT : " + str(np.max(TT_list)))

# %% inter-patient variability test
# Simulation parameter
# phase = 'induction'
# ratio = 2
# Number_of_patient = 500

# # Controller parameters
# Kp = float(param_opti.loc[param_opti['ratio']==ratio, 'Kp'])
# Ti = float(param_opti.loc[param_opti['ratio']==ratio, 'Ti'])
# Td = float(param_opti.loc[param_opti['ratio']==ratio, 'Td'])
# PID_param = [Kp, Ti, Td, ratio]


# IAE_list = []
# TT_list = []
# p1 = figure(plot_width=900, plot_height=300)
# p2 = figure(plot_width=900, plot_height=300)
# p3 = figure(plot_width=900, plot_height=300)

# for i in range(Number_of_patient):
#     if i%20==0:
#         print(i)
#     #Generate random patient information with uniform distribution
#     age = np.random.randint(low=18,high=70)
#     height = np.random.randint(low=150,high=190)
#     weight = np.random.randint(low=50,high=100)
#     gender = np.random.randint(low=0,high=1)
#     #generate PD model with normal distribution from Bouillon et al.
#     Ce50p = np.random.normal(loc = 4.47, scale = 4.92*0.066)
#     Ce50r = np.random.normal(loc = 19.3, scale = 8*0.394)
#     gamma = np.random.normal(loc = 1.43, scale = 1.43*0.142)
#     beta = 0
#     E0 = min(100, np.random.normal(loc = 97.4, scale = 97.4*0.005)) #standard deviation not gived in the article, arbitrary fixed to 5%
#     Emax = E0

#     Patient_info = [age, height, weight, gender, Ce50p, Ce50r, gamma, beta, E0, Emax]
#     IAE, data = simu(Patient_info, phase, PID_param)
#     p1.line(np.arange(0,len(data[0]))/60, data[0])
#     p2.line(np.arange(0,len(data[0]))/60, data[1], legend_label='MAP (mmgh)')
#     p2.line(np.arange(0,len(data[0]))/60, data[2]*10, legend_label='CO (cL/min)', line_color="#f46d43")
#     p3.line(np.arange(0,len(data[3]))/60, data[3], line_color="#006d43", legend_label='propofol (mg/min)')
#     p3.line(np.arange(0,len(data[4]))/60, data[4], line_color="#f46d43", legend_label='remifentanil (ng/min)')
#     TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(data[0], Te = 1, phase = phase)
#     TT_list.append(TT)
#     IAE_list.append(IAE)

# p1.title.text = 'BIS'
# p3.title.text = 'Infusion rates'
# p3.xaxis.axis_label = 'Time (min)'
# grid = row(column(p3,p1,p2))

# show(grid)

# print("Mean IAE : " + str(np.mean(IAE_list)))
# print("Mean TT : " + str(np.mean(TT_list)))
# print("Min TT : " + str(np.min(TT_list)))
# print("Max TT : " + str(np.max(TT_list)))


# %% Intra patient variability


# Simulation parameter
phase = 'induction'
ratio = 2
Number_of_patient = 128

# Controller parameters
Kp = float(param_opti.loc[param_opti['ratio'] == ratio, 'Kp'])
Ti = float(param_opti.loc[param_opti['ratio'] == ratio, 'Ti'])
Td = float(param_opti.loc[param_opti['ratio'] == ratio, 'Td'])
PID_param = [Kp, Ti, Td, ratio]


IAE_list = []
TT_list = []
ST10_list = []
ST20_list = []
US_list = []
BIS_NADIR_list = []
p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)

for i in range(Number_of_patient):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    IAE, data = simu(Patient_info, phase, PID_param, random_PD=True, random_PK=True)
    p1.line(np.arange(0, len(data[0])) / 60, data[0])
    p2.line(np.arange(0, len(data[0])) / 60, data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0, len(data[0])) / 60, data[2] * 10,
            legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0, len(data[3])) / 60, data[3],
            line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0, len(data[4])) / 60, data[4],
            line_color="#f46d43", legend_label='remifentanil (ng/min)')
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
        data[0], Te=1, phase=phase)
    TT_list.append(TT)
    BIS_NADIR_list.append(BIS_NADIR)
    ST10_list.append(ST10)
    ST20_list.append(ST20)
    US_list.append(US)
    IAE_list.append(IAE)

p1.title.text = 'BIS'
p1.xaxis.axis_label = 'Time (min)'
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
export_svg(p1, filename="BIS_PID.svg")
p3.output_backend = "svg"
export_svg(p3, filename="input_PID.svg")
