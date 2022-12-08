#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:42:11 2022

@author: aubouinb
"""
import numpy as np
import casadi as cas
from scipy.linalg import expm
from scipy.linalg import block_diag


def RK4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

#k10, k12, k13, k21, k31, ke0, k1e, gamma, C50 = patient(index = 2)


def dxdt(A, B, x, u):
    return A @ np.expand_dims(x, axis=-1).T + B @ u


def xdot_fun(x, t, u):  # Function for integration
    return dxdt(x, u)


def xdot_linfun(x, t, u, A, B):  # Function for integration
    y = A@x + B@u
    return y


def xplus(x, u, ts):  # Euler approximation
    return x + ts*dxdt(x, u)


def discretize(A, B, ts):
    (n, m) = B.shape
    Mt = np.zeros((n+m, n+m))
    Mt[0:n, 0:n] = A
    Mt[0:n, n:n+m] = B
    Mtd = expm(Mt*ts/60)
    Ad = Mtd[0:n, 0:n]
    Bd = Mtd[0:n, n:n+m]
    return Ad, Bd


def BIS(xep, xer, Bis_param):
    C50p = Bis_param[0]
    C50r = Bis_param[1]
    gamma = Bis_param[2]
    beta = Bis_param[3]
    E0 = Bis_param[4]
    Emax = Bis_param[5]
    up = max(0, xep / C50p)
    ur = max(0, xer / C50r)
    Phi = up/(up + ur + 1e-6)
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    BIS = E0 - Emax * i ** gamma / (1 + i ** gamma)
    return BIS


class EKF:
    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), P0: list = np.eye(8)):

        Ad, Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.R = R
        self.Q = Q
        self.P = P0

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 8, 8)   # P matrix
        Pup = cas.MX.sym('P', 8, 8)   # P matrix

        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        Ppred = cas.MX(Ad) @ P @ cas.MX(Ad.T) + cas.MX(self.Q)
        self.Pred = cas.Function('Pred', [x, u, P], [xpred, Ppred], [
                                 'x', 'u', 'P'], ['xpred', 'Ppred'])

        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        h_fun = E0 - Emax * i ** gamma / (1 + i ** gamma)
        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        error = y - h_fun
        xup = x + K @ error
        Pup = (cas.MX(np.identity(8)) - K@H)@P
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup, error], [
                                   'x', 'y', 'P'], ['xup', 'Pup', 'error'])

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]
        self.error = 0

    def estimate(self, u, bis):

        self.Predk = self.Pred(x=self.x, u=u, P=self.P)
        self.xpr = self.Predk['xpred'].full().flatten()
        self.Ppr = self.Predk['Ppred'].full()

        self.Updatek = self.Update(x=self.xpr, y=bis, P=self.Ppr)
        self.x = self.Updatek['xup'].full().flatten()
        self.P = self.Updatek['Pup']
        self.error = float(self.Updatek['error'])

        self.x[3] = max(1e-3, self.x[3])
        self.x[7] = max(1e-3, self.x[7])

        self.Bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.Bis


class linear_Kalman:
    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), P0: list = np.eye(8)):

        self.Ad, self.Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.R = R
        self.Q = Q
        self.P = P0

        # output function
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        up = Cep / C50p
        ur = Cer / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)
        H = cas.gradient(bis, cas.vertcat(*[Cep, Cer]))
        self.Output = cas.Function('Output', [Cep, Cer], [bis], ['Cep', 'Cer'], ['bis'])
        self.Output_grad = cas.Function('Output', [Cep, Cer], [H], ['Cep', 'Cer'], ['H'])

        # init state and output
        self.x = x0

    def init_first_step(self, Bis_target):
        """Compute the optimal effect site concentration target to reach Bis_target and respect cost on the Input"""

        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')

        w = [Cep, Cer]
        w0 = [self.BIS_param[0], self.BIS_param[1]]
        ubw = [10, 20]
        lbw = [0, 0]

        Hk = self.Output(Cep=Cep, Cer=Cer/2)
        bis = Hk['bis']
        # + self.R[0,0] * (self.C50p - Cep)**2 + self.R[1,1] * (self.C50r - Cer)**2
        J = (bis - Bis_target)**2
        g = []
        ubg = []
        lbg = []

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        w_opt = sol['x'].full().flatten()
        self.Cep_target = w_opt[0]
        self.Cer_target = w_opt[1]
        Hk = self.Output(Cep=self.Cep_target, Cer=self.Cer_target)
        bis = Hk['bis']
        self.Bis_target_Ce = bis

        Grad = self.Output_grad(Cep=self.Cep_target, Cer=self.Cer_target)
        self.a = float(Grad['H'][0])
        self.b = float(Grad['H'][1])
        self.c = self.Bis_target_Ce - (self.a*self.Cep_target + self.b*self.Cer_target)
        self.C = np.array([[0, 0, 0, self.a, 0, 0, 0, self.b]])

    def estimate(self, u, bis):

        # predict
        self.x = self.Ad @ self.x + self.Bd @ np.expand_dims(u, axis=1)
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

        # correct
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self. R)
        self.x = self.x + K @ (bis - self.C @ self.x - self.c)
        self.P = self.P - K @ self.C @ self.P

        self.x = np.clip(self.x, a_min=1e-3, a_max=1e10)
        self.x[3] = max(1e-3, self.x[3])
        self.x[7] = max(1e-3, self.x[7])

        self.Bis = self.C @ self.x + self.c

        return self.x, self.Bis

    def update_output(self, sol):
        self.a = sol[0]
        self.b = sol[1]
        self.c = sol[2]
        self.C = np.array([[0, 0, 0, self.a, 0, 0, 0, self.b]])


class EKF_extended:
    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((11, 1)),
                 Q: list = np.eye(11), R: list = np.array([1]), P0: list = np.eye(11)):

        Ad, Bd = discretize(A, B, ts)
        Ad = block_diag(Ad, np.eye(3))
        Bd = np.concatenate((Bd, np.zeros((3, 2))), axis=0)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.R = R
        self.Q = Q
        self.P = P0
        self.P0 = P0

        # declare CASADI variables
        x = cas.MX.sym('x', 11)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 11, 11)   # P matrix
        Pup = cas.MX.sym('P', 11, 11)   # P matrix

        # declare CASADI functions
        # linear part
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        Ppred = cas.MX(Ad) @ P @ cas.MX(Ad.T) + cas.MX(self.Q)
        self.Pred = cas.Function('Pred', [x, u, P], [xpred, Ppred], [
                                 'x', 'u', 'P'], ['xpred', 'Ppred'])

        # output function
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        up = Cep / C50p
        ur = Cer / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)
        H = cas.gradient(bis, cas.vertcat(*[Cep, Cer]))
        self.Output = cas.Function('Output', [Cep, Cer], [bis], ['Cep', 'Cer'], ['bis'])
        self.Output_grad = cas.Function('Output', [Cep, Cer], [H], ['Cep', 'Cer'], ['H'])

        h_fun = x[8] * x[3] + x[9] * x[4] + x[10]
        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        xup = x + K @ (y - h_fun)
        Pup = (cas.MX(np.identity(11)) - K@H)@P
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup], [
                                   'x', 'y', 'P'], ['xup', 'Pup'])

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]

    def init_first_step(self, Bis_target):
        """Compute the optimal effect site concentration target to reach Bis_target and respect cost on the Input"""

        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')

        w = [Cep, Cer]
        w0 = [self.BIS_param[0], self.BIS_param[1]]
        ubw = [10, 20]
        lbw = [0, 0]

        Hk = self.Output(Cep=Cep, Cer=Cer/2)
        bis = Hk['bis']
        # + self.R[0,0] * (self.C50p - Cep)**2 + self.R[1,1] * (self.C50r - Cer)**2
        J = (bis - Bis_target)**2
        g = []
        ubg = []
        lbg = []

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        w_opt = sol['x'].full().flatten()
        self.Cep_target = w_opt[0]
        self.Cer_target = w_opt[1]
        Hk = self.Output(Cep=self.Cep_target, Cer=self.Cer_target)
        bis = Hk['bis']
        self.Bis_target_Ce = bis

        Grad = self.Output_grad(Cep=self.Cep_target, Cer=self.Cer_target)
        self.a = Grad['H'][0]
        self.b = Grad['H'][1]
        self.c = self.Bis_target_Ce - (self.a*self.Cep_target + self.b*self.Cer_target)

        self.x[8] = self.a
        self.x[9] = self.b
        self.x[10] = self.c

        self.P[8:, 8:] = self.P0[8:, 8:]

    def estimate(self, u, bis):

        self.Predk = self.Pred(x=self.x, u=u, P=self.P)
        self.xpr = self.Predk['xpred'].full().flatten()
        self.Ppr = self.Predk['Ppred'].full()

        self.Updatek = self.Update(x=self.xpr, y=bis, P=self.Ppr)
        self.x = self.Updatek['xup'].full().flatten()
        self.P = self.Updatek['Pup']

        self.x[3] = max(1e-3, self.x[3])
        self.x[7] = max(1e-3, self.x[7])

        self.Bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.Bis


class MHE:
    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), P0: list = np.eye(8)):

        Ad, Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.R = R
        self.Q = Q
        self.P = P0

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 8, 8)   # P matrix
        Pup = cas.MX.sym('P', 8, 8)   # P matrix

        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        Ppred = cas.MX(Ad) @ P @ cas.MX(Ad.T) + cas.MX(self.Q)
        self.Pred = cas.Function('Pred', [x, u, P], [xpred, Ppred], [
                                 'x', 'u', 'P'], ['xpred', 'Ppred'])

        xep = cas.MX(1)
        xer = cas.MX(1)
        up = xep / C50p
        ur = xer / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        h_fun = E0 - Emax * i ** gamma / (1 + i ** gamma)
        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        xup = x + K @ (y - h_fun)
        Pup = (cas.MX(np.identity(8)) - K@H)@P
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup], [
                                   'x', 'y', 'P'], ['xup', 'Pup'])

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]

    def estimate(self, u, bis):

        self.Predk = self.Pred(x=self.x, u=u, P=self.P)
        self.xpr = self.Predk['xpred'].full().flatten()
        self.Ppr = self.Predk['Ppred'].full()

        self.Updatek = self.Update(x=self.xpr, y=bis, P=self.Ppr)
        self.x = self.Updatek['xup'].full().flatten()
        self.P = self.Updatek['Pup']

        self.Bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.Bis
