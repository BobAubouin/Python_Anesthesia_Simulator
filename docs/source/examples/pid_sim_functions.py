#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:09 2022

@author: aubouinb
"""
# Standard import
import time
import multiprocessing
from functools import partial

# Third party imports
import numpy as np
import pandas as pd
import casadi as cas
from pyswarm import pso

# Local imports
import python_anesthesia_simulator as pas

# Patient table:
# index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 4.73, 24.97,  2.97,  0.3, 97.86, 89.62],
                 [2,  36, 163, 50, 0, 4.43, 19.33,  2.04,  0.29, 89.1, 98.86],
                 [3,  28, 164, 52, 0, 4.81, 16.89,  1.18,  0.14, 93.66, 94.],
                 [4,  50, 163, 83, 0, 3.86, 20.97,  1.08,  0.12, 94.6, 93.2],
                 [5,  28, 164, 60, 1, 5.22, 18.95,  2.43,  0.68, 97.43, 96.21],
                 [6,  43, 163, 59, 0, 3.41, 23.26,  3.16,  0.58, 85.33, 97.07],
                 [7,  37, 187, 75, 1, 4.83, 15.21,  2.94,  0.13, 91.87, 90.84],
                 [8,  38, 174, 80, 0, 4.36, 13.86,  3.37,  1.05, 97.45, 96.36],
                 [9,  41, 170, 70, 0, 4.57, 16.2,  1.55,  0.16, 85.83, 94.6],
                 [10, 37, 167, 58, 0, 6.02, 23.47,  2.83,  0.77, 95.18, 88.17],
                 [11, 42, 179, 78, 1, 3.79, 22.25,  2.35,  1.12, 98.02, 96.95],
                 [12, 34, 172, 58, 0, 5.7, 18.64,  2.67,  0.4, 99.57, 96.94],
                 [13, 38, 169, 65, 0, 4.64, 19.50,  2.38,  0.48, 93.82, 94.40]]


def simu(Patient_info: list, style: str, PID_param: list,
         random_PK: bool = False, random_PD: bool = False) -> tuple[float, list, list]:
    """
    Perform a closed-loop Propofol-Remifentanil anesthesia simulation with a PID controller.

    Parameters
    ----------
    Patient_info : list
        list of patient informations, Patient_info = [Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax].
    style : str
        Either 'induction' or 'total' to describe the phase to simulate.
    PID_param : list
        Parameters of the NMPC controller, MPC_param = [Kp, Ti, Td, ratio].
    random_PK : bool, optional
        Add uncertaintie to the PK model. The default is False.
    random_PD : bool, optional
        Add uncertainties to the PD model. The default is False.

    Returns
    -------
    IAE : float
        Integrated Absolute Error, performance index of the function
    data : list
        list of the signals during the simulation, data = [BIS, MAP, CO, up, ur]
    BIS_param: list
        BIS parameters of the simulated patient.

    """

    hill_param = Patient_info[4:]
    if hill_param[0] is None:
        hill_param = None
    George = pas.Patient(Patient_info[:4], hill_param=hill_param,
                         random_PK=random_PK, random_PD=random_PD)
    hill_param = George.bis_pd.hill_param
    Ce50p = hill_param[0]
    Ce50r = hill_param[1]
    gamma = hill_param[2]
    beta = hill_param[3]
    E0 = hill_param[4]
    Emax = hill_param[5]

    ts = 1
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    ratio = PID_param[3]
    PID_controller = PID(Kp=PID_param[0], Ti=PID_param[1], Td=PID_param[2],
                         N=5, Ts=1, umax=max(up_max, ur_max / ratio), umin=0)

    if style == 'induction':
        N_simu = 10 * 60
        Bis = George.bis
        for i in range(N_simu):
            uP = PID_controller.one_step(Bis, BIS_cible)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Bis, _, _, _ = George.one_step(uP, uR, noise=False)

    elif style == 'total':
        N_simu = int(60 / ts) * 60
        Bis = George.bis
        for i in range(N_simu):
            uP = PID_controller.one_step(Bis, BIS_cible)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Dist = pas.compute_disturbances(i * ts, 'realistic')
            Bis, _, _, _ = George.one_step(uP, uR, dist=Dist, noise=False)

    elif style == 'maintenance':
        N_simu = 25 * 60  # 25 minutes

        # find equilibrium input
        uP, uR = George.find_bis_equilibrium_with_ratio(BIS_cible, ratio)
        # initialize the simulator with the equilibrium input
        George.initialized_at_given_input(u_propo=uP, u_remi=uR)
        Bis = George.bis
        # initialize the PID at the equilibriium point
        PID_controller.integral_part = uP / PID_controller.Kp
        PID_controller.last_BIS = 50
        for i in range(N_simu):
            uP = PID_controller.one_step(Bis, BIS_cible)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Dist = pas.compute_disturbances(i * ts, 'step')
            Bis, _, _, _ = George.one_step(uP, uR, dist=Dist, noise=False)

    IAE = np.sum(np.abs(George.dataframe['BIS'] - BIS_cible))
    return IAE, George.dataframe, hill_param


def one_simu(x, ratio, phase, i):
    """Cost of one simulation, i is the patient index."""
    Patient_info = Patient_table[i-1][1:]
    iae, _, _ = simu(Patient_info, phase, [x[0], x[1], x[2], ratio])
    return iae


def cost(x, ratio, phase):
    """Cost of the optimization, x is the vector of the PID controller.

    x = [Kp, Ti, Td]
    IAE is the maximum integrated absolut error over the patient population.
    """
    pool_obj = multiprocessing.Pool()
    func = partial(one_simu, x, ratio, phase)
    IAE = pool_obj.map(func, range(0, 13))
    pool_obj.close()
    pool_obj.join()

    return max(IAE)


class PID():
    """PID class to implement PID on python."""

    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 5,
                 Ts: float = 1, umax: float = 1e10, umin: float = -1e10):
        """ Implementation of a working PID with anti-windup:

            PID = Kp ( 1 + Te / (Ti - Ti z^-1) + Td (1-z^-1) / (Td/N (1-z^-1) + Te) )
        Inputs : - Kp: Gain
                 - Ti: Integrator time constant
                 - Td: Derivative time constant
                 - N: interger to filter the derivative part
                 - Ts: sampling time
                 - umax: upper saturation of the control input
                 - umin: lower saturation of the control input."""
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.N = N
        self.Ts = Ts
        self.umax = umax
        self.umin = umin

        self.integral_part = 0
        self.derivative_part = 0
        self.last_BIS = 100

    def one_step(self, BIS: float, Bis_target: float):
        """Compute the next command for the PID controller."""
        error = -(Bis_target - BIS)
        self.integral_part += self.Ts / self.Ti * error

        self.derivative_part = (self.derivative_part * self.Td / self.N +
                                self.Td * (BIS - self.last_BIS)) / (self.Ts +
                                                                    self.Td / self.N)
        self.last_BIS = BIS

        control = self.Kp * (error + self.integral_part + self.derivative_part)

        # Anti windup Conditional Integration from
        # Visioli, A. (2006). Anti-windup strategies. Practical PID control, 35-60.
        if (control >= self.umax) and control * error <= 0:
            self.integral_part = self.umax / self.Kp - error - self.derivative_part
            control = self.umax

        elif (control <= self.umin) and control * error <= 0:
            self.integral_part = self.umin / self.Kp - error - self.derivative_part
            control = self.umin

        return control


# ------------------- MAIN ------------------- #
if __name__ == "__main__":
    phase = 'maintenance'

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
    if phase == 'maintenance':
        param_opti.to_csv('./optimal_parameters_PID_reject.csv')
    else:
        param_opti.to_csv('./optimal_parameters_PID.csv')
