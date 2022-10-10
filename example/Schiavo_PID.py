#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:32:55 2022

@author: aubouinb
"""

""" MISO PID control of anesthesia from "Individualized PID tuning for maintenance of general anesthesia with
propofol and remifentanil coadministration" Michele Schiavo, Fabrizio Padula, Nicola Latronico, 
Massimiliano Paltenghi,Antonio Visioli 2022"""


import numpy as np

from Patient import Patient
from PID import PID
from pyswarm import pso
import casadi as cas
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot, row, column
from bokeh.models import Range1d
#Patient table:
#index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 6.33, 12.5, 2.24, 2.00, 98.8, 94.1],
                 [2,  36, 163, 50, 0, 6.76, 12.7, 4.29, 1.50, 98.6, 86],
                 [3,  28, 164, 52, 0, 8.44, 7.1,  4.10, 1.00, 91.2, 80.7],
                 [4,  50, 163, 83, 0, 6.44, 11.1, 2.18, 1.30, 95.9, 102],
                 [5,  28, 164, 60, 1, 4.93, 12.5, 2.46, 1.20, 94.7, 85.3],
                 [6,  43, 163, 59, 0, 12.0, 12.7, 2.42, 1.30, 90.2, 147],
                 [7,  37, 187, 75, 1, 8.02, 10.5, 2.10, 0.80, 92.0, 104],
                 [8,  38, 174, 80, 0, 6.56, 9.9,  4.12, 1.00, 95.5, 76.4],
                 [9,  41, 170, 70, 0, 6.15, 11.6, 6.89, 1.70, 89.2, 63.8],
                 [10, 37, 167, 58, 0, 13.7, 16.7, 3.65, 1.90, 83.1, 151],
                 [11, 42, 179, 78, 1, 4.82, 14.0, 1.85, 1.20, 91.8, 77.9],
                 [12, 34, 172, 58, 0, 4.95, 8.8,  1.84, 0.90, 96.2, 90.8],
                 [13, 38, 169, 65, 0, 7.42, 10.5, 3.00, 1.00, 93.1, 96.58]]

Disturb_point_1 = np.array([[0,     0,  0, 0  ], #time, BIS signal, MAP, CO signals
                            [9.9,   0,  0, 0  ],
                            [10,   20, 10, 0.6],
                            [12,   20, 10, 0.6],
                            [13,    0,  0, 0  ],
                            [19.9,  0,  0, 0  ],
                            [20.2, 20, 10, 0.5],
                            [21,   20, 10, 0.5],
                            [21.5,  0,  0, 0  ],
                            [26,  -20,-10,-0.8],
                            [27,   20, 10, 0.9],
                            [28,   10,  7, 0.2],
                            [36,   10,  7, 0.2],
                            [37,   30, 15, 0.8],
                            [37.5, 30, 15, 0.8],
                            [38,   10,  5, 0.2],
                            [41,   10,  5, 0.2],
                            [41.5, 30, 10, 0.5],
                            [42,   30, 10, 0.5],
                            [43,   10,  5, 0.2],
                            [47,   10,  5, 0.2],
                            [47.5, 30, 10, 0.9],
                            [50,   30,  8, 0.9],
                            [51,   10,  5, 0.2],
                            [56,   10,  5, 0.2],
                            [56.5,  0,  0, 0  ]])


def simu(style, P, Patient_index, Patient_table):
    age = Patient_table[Patient_index-1][1]
    height = Patient_table[Patient_index-1][2]
    weight = Patient_table[Patient_index-1][3]
    gender = Patient_table[Patient_index-1][4]
    Ce50p = Patient_table[Patient_index-1][5]
    Ce50r = Patient_table[Patient_index-1][6]
    gamma = Patient_table[Patient_index-1][7]
    beta = Patient_table[Patient_index-1][8]
    E0 = Patient_table[Patient_index-1][9]
    Emax = Patient_table[Patient_index-1][10]
    
    George = Patient(age, height, weight, gender, C50p_BIS = Ce50p, C50r_BIS = Ce50r, gamma_BIS = gamma, beta_BIS = beta, E0_BIS = E0, Emax_BIS = Emax)
    
    
    
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67 
    ratio = P[3]
    PID_controller = PID(Kp = P[0], Ti = P[1], Td = P[2],
                      N = 5, Te = 1, umax = max(up_max, ur_max/ratio), umin = 0)
    
    if style=='induction':
        N_simu = 5*60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        for i in range(N_simu):
            # if i == 100:
            #     print("break")
            
            uR = min(ur_max,max(0,uP*ratio))
            uP = min(up_max,max(0,uP))
            Bis,Co,Map = George.one_step(uP, uR, noise = False)
            BIS[i] = Bis
            MAP[i] = Map[0,0]
            CO[i] = Co[0,0]
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(Bis,BIS_cible)
    if style=='stabilization':
        N_simu = 70*60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        
        
        w=[]
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g=[]
        lbg = []
        ubg = []
        
        Ap = np.array(George.PropoPK.A)
        Bp = np.array(George.PropoPK.B)
        Ar = np.array(George.RemiPK.A)
        Br = np.array(George.RemiPK.B)
        
        x0p = np.linalg.solve(Ap,Bp*up_max/2)
        x0r = np.linalg.solve(Ar,Br*up_max)
        xp = cas.MX.sym('xp',4)
        w0 += x0p.tolist()
        xr = cas.MX.sym('xr',4)
        w0 += x0r.tolist()
        UP = cas.MX.sym('up',1)
        w = [xp,xr,UP]
        w0 += [up_max/2]
        lbw = [0]*8 + [1]
        ubw = [1e4]*9
        
        up = xp[3] / Ce50p
        ur = xr[3] / Ce50r
        Phi = up/(up + ur)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        J = (50 - (E0 + Emax * i ** gamma / (1 + i ** gamma)))**2
        
        

        
        g = [Ap @ xp + Bp * UP, Ar @ xr + Br * (ratio*UP)]
        lbg = [-1e-5]*8
        ubg = [1e-5]*8
        
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, {'ipopt.max_iter': 300})
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        
        George.PropoPK.x = np.expand_dims(np.array(w_opt[:4]),axis=1)
        George.RemiPK.x = np.expand_dims(np.array(w_opt[4:8]),axis=1)
        uP = w_opt[-1]
        for i in range(N_simu):
            # if i == 100:
            #     print("break")
            
            uR = min(ur_max,max(0,uP*ratio))
            uP = min(up_max,max(0,uP))
            Bis,Co,Map = George.one_step(uP, uR, noise = True)
            BIS[i] = min(100,Bis + np.interp(i/60, Disturb_point_1[:,0], Disturb_point_1[:,1]))
            MAP[i] = Map[0,0] + np.interp(i/60, Disturb_point_1[:,0], Disturb_point_1[:,2]) 
            CO[i] = Co[0,0] + np.interp(i/60, Disturb_point_1[:,0], Disturb_point_1[:,3]) 
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(BIS[i] ,BIS_cible)
            if i==55*60:
                BIS_cible=100
        
    IAE = np.sum(np.abs(BIS - BIS_cible))
    return IAE, [BIS, MAP, CO, Up, Ur]
        
        
        



#%% PSO
# ratio = 2   
# def banana(x):
#     IAE = []
#     ratio = 2
#     for i in range(13):
#         iae, data = simu('induction', [x[0], x[1], x[2], ratio], i+1, Patient_table)
#         IAE.append(iae)
#     return max(IAE)

# lb = [1e-6, 100, 10]
# ub =[1,300,40]
# xopt, fopt = pso(banana, lb, ub, debug = True, minfunc = 1e-3) #Default: 100 particles as in the article

# print(xopt)
# P = [xopt[0], xopt[1], xopt[2], ratio]
#%%test

#xoptimal
# P = [0.43886005037982306, 173.35958590681594, 20.0, 2]
# P = [0.48272254, 196.10932941,  16.35625313, 2] 
P = [0.007945134003877127, 195.54111899704745, 17.166289073381094, 2]
ratio_list = [0.5, 2.5, 15]
IAE = []
TT = [0]*13
p1 = figure(plot_width=900, plot_height=300)
p2 = figure(plot_width=900, plot_height=300)
p3 = figure(plot_width=900, plot_height=300)
for ratio in range(4,5): #
    ratio = ratio/2
    Kp = (0.053*ratio**(-0.35) - 0.013)
    Ti = 206.98
    Td = 29.83 
    P[3] = ratio
    # Kp = -0.0598
    # Ti = 28.476
    # Td = 2.368
    # P = [Kp, Ti, Td, ratio]
    for i in range(1,2):
        iae, data = simu('stabilization', P, i+2, Patient_table)
        p1.line(np.arange(0,len(data[0]))/60, data[0])
        p2.line(np.arange(0,len(data[0]))/60, data[1], legend_label='MAP (mmgh)')
        p2.line(np.arange(0,len(data[0]))/60, data[2]*10, legend_label='CO (cL/min)', line_color="#f46d43")
        p3.line(np.arange(0,len(data[3]))/60, data[3], line_color="#006d43", legend_label='propofol (mg/min)')
        #p3.line(np.arange(0,len(data[4]))/60, data[4], line_color="#f46d43", legend_label='remifentanil (ng/min)')
        for j in range(len(data[0])):
            if data[0][j]<55:
                TT[i] = j/60
                break
        IAE.append(iae)
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3,p1,p2),p3)

show(grid)


print(np.mean(IAE))
        
print(np.mean(TT))
print(np.min(TT))
print(np.max(TT))