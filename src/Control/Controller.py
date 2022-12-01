#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:09 2022

@author: aubouinb
"""

import numpy as np
import casadi as cas
import control
from scipy.linalg import expm
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def discretize(A,B,ts):
    (n, m) = B.shape 
    Mt = np.zeros((n+m,n+m)) 
    Mt[0:n,0:n] = A
    Mt[0:n,n:n+m] = B
    Mtd = expm(Mt*ts/60)
    Ad = Mtd[0:n,0:n]
    Bd = Mtd[0:n,n:n+m]
    return Ad, Bd


class PID():
    """PID class to implement PID on python"""
    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 5,
                 Te: float = 1, umax: float = 1e10, umin: float = -1e10):
        """ Implementation of a working PID with anti-windup:
            PID = Kp ( 1 + Te / (Ti - Ti z^-1) + Td (1-z^-1) / (Td/N (1-z^-1) + Te) )
        Inputs : - Kp: Gain
                 - Ti: Integrator time constant
                 - Td: Derivative time constant
                 - N: interger to filter the derivative part
                 - Te: sampling time
                 - umax: upper saturation of the control input
                 - umin: lower saturation of the control input"""
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.N = N
        self.Te = Te
        self.umax = umax
        self.umin = umin
        
        self.integral_part = 0
        self.derivative_part = 0
        self.last_BIS = 100
        
    def one_step(self, BIS : float, Bis_cible : float):
        error = -(Bis_cible - BIS)
        self.integral_part += self.Te / self.Ti * error
        
        self.derivative_part = (self.derivative_part * self.Td / self.N + self.Td * (BIS - self.last_BIS)) / (self.Te + self.Td / self.N)
        self.last_BIS = BIS
        
        control = self.Kp * ( error + self.integral_part + self.derivative_part)
        
        # Anti windup Conditional Integration from Visioli, A. (2006). Anti-windup strategies. Practical PID control, 35-60.
        if (control>=self.umax ) and control * error <=0 :
            self.integral_part  = self.umax / self.Kp - error - self.derivative_part
            control = self.umax 
        
        elif (control<=self.umin ) and control * error <=0 :
            self.integral_part  = self.umin / self.Kp - error - self.derivative_part
            control = self.umin 
            
            
        return control





class NMPC:
    def __init__(self, A: list, B:list, BIS_param: list, ts: float = 1,
                 N: int = 10, Nu: int = 10, R:list = np.diag([2,1]), 
                 umax: list = [1e10]*2, umin: list = [0]*2,
                 dumax: list = [0.2, 0.4], dumin: list = [-0.2, -0.4],
                 ymax: float = 100, dymin: float = 0, ki: float = 0):
         
        Ad, Bd = discretize(A,B,ts)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]
        
        self.umax = umax
        self.umin = umin
        self.ymax = ymax
        self.dymin = dymin
        self.R = R #control cost
        self.N = N #horizon
        self.Nu = Nu
        
        # declare CASADI variables
        x = cas.MX.sym('x',8) # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')   #Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   #Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)


        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred],['x', 'u'], ['x+'])
        
        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        
        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)

        self.Output = cas.Function('Output', [x], [bis],['x'], ['bis'])
        
        self.U_prec = [0]*N*2
        # integrator
        self.ki = ki
        self.internal_target = None
        
        # Optimization problem definition
        w=[]
        self.lbw = []
        self.ubw = []
        J = 0
        gu = []
        gbis =[]
        self.lbg_u = []
        self.ubg_u = []
        self.lbg_bis = []
        self.ubg_bis = []
        
        X0 = cas.MX.sym('X0',8)
        Bis_target = cas.MX.sym('Bis_target')
        U_prec_true = cas.MX.sym('U_prec',2)
        Xk = X0
        for k in range(self.N):
            if k<=self.Nu-1:
                U = cas.MX.sym('U',2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k==0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            elif k>self.Nu-1:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = U
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]
                
            Xk = Pred['x+']
            Hk = self.Output(x=Xk)
            bis = Hk['bis']
            
            # J+= ((bis - Bis_target)**2/100 + ((bis - Bis_target - 25)/30)**8) + (U-U_prec).T @ self.R @ (U-U_prec) #+ 
            Ju = ((U-U_prec).T @ self.R @ (U-U_prec)/100 + (((U-U_prec).T @ self.R @ (U-U_prec)+10)/10)**4)
            
            J+= ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) + Ju
        
            #J+= self.Q * (bis - Bis_target)**2 + (U).T @ self.R @ (U)
            if k==self.N-1:
                J+= ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) * 1e3
                
            gu += [U-U_prec]
            gbis += [bis]
            self.lbg_u += dumin
            self.ubg_u += dumax
            self.ubg_bis += [self.ymax]
            
        opts = {'ipopt.print_level':1, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, U_prec_true]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*(gu))} #+gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        
        
    def one_step(self, x : list, Bis_target : float, Bis_measure: float):
        
        #the init point of the optimization proble is the previous optimal solution
        w0 = []
        for k in range(self.Nu):
            if k < self.Nu-1:
                w0  += self.U_prec[2*(k+1):2*(k+1)+2]
            else:
                w0  += self.U_prec[2*k:2*k+2]
        
        #Init internal target
        if self.internal_target is None:
            self.internal_target = Bis_target
            
        self.lbg_bis = [min(90,self.internal_target - self.dymin)]*self.N
        sol = self.solver(x0=w0, p = list(x) + [self.internal_target] + list(self.U_prec[0:2]), lbx=self.lbw, ubx=self.ubw, lbg=self.lbg_u, ubg=self.ubg_u) # + self.lbg_bis + self.ubg_bis
          
        w_opt = sol['x'].full().flatten()
        
        Up = w_opt[::2]
        Ur = w_opt[1::2]
        self.U_prec = list(w_opt)
        
        Hk = self.Output(x=x)
        bis = float(Hk['bis'])
        
        #integrator
        self.internal_target = self.internal_target + self.ki * (Bis_target - Bis_measure)

        
        # print for debug
        if True:
            bis = np.zeros(self.N)
            Xk = x
            for k in range(self.N):
                if k < self.Nu-1:
                    u = w_opt[2*k:2*k+2]
                else:
                    u = w_opt[-2:]
                Pred = self.Pred(x=Xk, u=u)
                Xk = Pred['x+']
                Hk = self.Output(x=Xk)
                bis[k] = float(Hk['bis'])
        if False:
            fig, axs = plt.subplots(2,figsize=(6,8))
            axs[0].plot(Up, label = 'propofol')
            axs[0].plot(Ur, label = 'remifentanil')
            axs[0].grid()
            axs[0].legend()
            axs[1].plot(bis, label = 'bis')
            axs[1].plot([self.internal_target]*len(bis), label = 'bis target')
            axs[1].grid()
            axs[1].legend()
            plt.show()
            
        return Up[0], Ur[0], bis

class MPC:
    
    
    
    def __init__(self, A: list, B:list, BIS_param: list, ts: float = 1,
                 N: int = 10, Nu: float = 10, Q:list = 10, R:list = np.diag([2,1]), 
                 umax: list = [1e10]*2, umin: list = [0]*2,
                 dumax: list = [0.2, 0.4], dumin: list = [-0.2, -0.4],
                 ymax: float = 100, ymin: float = 30, ki: float = 0):
         
        Ad, Bd = discretize(A,B,ts)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]
        
        self.umax = umax
        self.umin = umin
        self.ymax = ymax
        self.ymin = ymin
        self.R = R #control cost
        self.Q = Q #objective cost
        self.N = N #horizon
        self.Nu = Nu
        self.ki = ki
        # declare CASADI variables
        x = cas.MX.sym('x',8) # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)


        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred],['x', 'u'], ['x+'])
        
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        up = Cep / C50p
        ur = Cer / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        
        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)

        self.Output = cas.Function('Output', [Cep, Cer], [bis],['Cep', 'Cer'], ['bis'])
        
        self.U_prec = [10]*N*2
        self.Bis_target_Ce = None
        
        # Start with an empty NLP
        w=[]
        self.lbw = []
        self.ubw = []
        J = 0
        g_u=[]
        g_p=[]
        g_r=[]
        self.lbg_u = []
        self.ubg_u = []
        Cep_target = cas.MX.sym('Cep_target',1)
        Cer_target = cas.MX.sym('Cer_target',1)
        X0 = cas.MX.sym('Xk',8)
        Xk = X0
        U_prec_true = cas.MX.sym('U_prec_true',2)
        for k in range(self.N):
            U = cas.MX.sym('U',2)
            w   += [U]
            self.lbw += self.umin
            self.ubw += self.umax
            
            if k==0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]
            Xk = Pred['x+']
            
            J+= self.Q * ((Xk[3] - Cep_target)**2 + (Xk[7] - Cer_target)**2) + (U-U_prec).T @ self.R @ (U-U_prec)
            
            g_u += [U-U_prec]
            self.lbg_u += dumin
            self.ubg_u += dumax
            
            g_p += [Xk[3]]            
            g_r += [Xk[4]]


            
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p':cas.vertcat(*[Cep_target, Cer_target, U_prec_true, X0]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*(g_u + g_p + g_r))}
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        
        #integrator
        self.internal_target = None        
        
        
    def find_Ce_target(self, x: list, Bis_target : float):
        """Compute the optimal effect site concentration target to reach Bis_target and respect cost on the Input"""
        
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        
        w = [Cep, Cer]
        w0 =[self.BIS_param[0], self.BIS_param[1]]
        ubw = [20, 20]
        lbw = [0, 0]
        
        Hk = self.Output(Cep=Cep, Cer = Cer)
        bis = Hk['bis']
        J = (bis - Bis_target)**2 + 2* (x[3] - Cep)**2 + 0.1* (x[7] - Cer)**2
        g = []
        ubg = []
        lbg = []
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            
        w_opt = sol['x'].full().flatten()
        self.Cep_target = w_opt[0]
        self.Cer_target = w_opt[1]
        self.Bis_target_Ce = Bis_target
        
    def one_step(self, x : list, Bis_target : float, Bis_measure: float):
        """Compute the optimal commande to reach Bis_target value using propofol and remifentanil inputs
        Inputs: -x state of the Propofol-Remifentanil PK model
                - Bis_target: in %
        Outputs: - Up: optimal Propofol rate for the next time stamp
                 - Ur: optimal Remifentanil rate for the next time stamp"""
        if self.internal_target is None:
            self.internal_target = Bis_target
                    
        if self.internal_target != self.Bis_target_Ce:
            self.find_Ce_target(x, self.internal_target)
        
        w0 = []
        for k in range(self.N):
            if k < self.Nu-1:
                w0  += self.U_prec[2*(k+1):2*(k+1)+2]
            else:
                w0  += self.U_prec[2*k:2*k+2]
       
        lbg_p = [0]*self.N
        ubg_p = [self.Cep_target]*self.N
        lbg_r = [0]*self.N
        ubg_r = [self.Cer_target]*self.N
        
        sol = self.solver(x0=w0, p=[self.Cep_target, self.Cer_target] + self.U_prec[0:2] + list(x), lbx=self.lbw, ubx=self.ubw, lbg=(self.lbg_u + lbg_p + lbg_r), ubg=(self.ubg_u + ubg_p + ubg_r))
        w_opt = sol['x'].full().flatten()
        
        #integrator
        self.internal_target = self.internal_target + self.ki * (Bis_target - Bis_measure)
        
        Up = w_opt[::2]
        Ur = w_opt[1::2]
        self.U_prec = list(w_opt)
        # print for debug
        if False:
            bis = np.zeros(self.N)
            Xk = x
            for k in range(self.N):
                Pred = self.Pred(x=Xk, u=w_opt[2*k:2*k+2])
                Xk = Pred['x+']
                Hk = self.Output(x=Xk)
                bis[k] = float(Hk['bis'])
            
            fig, axs = plt.subplots(2,figsize=(14,16))
            axs[0].plot(Up, label = 'propofol')
            axs[0].plot(Ur, label = 'remifentanil')
            axs[1].plot(bis, label = 'bis')
            plt.grid()
            plt.legend()
            plt.show()
            
        return Up[0], Ur[0]


class MPC_lin:
    
    def __init__(self, A: list, B:list, BIS_param: list, ts: float = 1,
                 N: int = 10, Nu: float = 10, R:list = np.diag([2,1]), 
                 umax: list = [1e10]*2, umin: list = [0]*2,
                 dumax: list = [0.2, 0.4], dumin: list = [-0.2, -0.4],
                 dymin: float = 30, ki: float = 0):
         
        Ad, Bd = discretize(A,B,ts)
        self.BIS_param = BIS_param
        self.C50p = BIS_param[0]
        self.C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]
        
        self.umax = umax
        self.umin = umin
        self.dumax = dumax
        self.dumin = dumin
        self.dymin = dymin
        self.R = R #control cost
        self.N = N #horizon
        self.Nu = Nu
        self.ki = ki
        # declare CASADI variables
        x = cas.MX.sym('x',8) # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)


        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred],['x', 'u'], ['x+'])
        
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        up = Cep / self.C50p
        ur = Cer / self.C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        
        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)
        H = cas.gradient(bis, cas.vertcat(*[Cep, Cer]))
        self.Output = cas.Function('Output', [Cep, Cer], [bis],['Cep', 'Cer'], ['bis'])
        self.Output_grad = cas.Function('Output', [Cep, Cer], [H],['Cep', 'Cer'], ['H'])
        
        self.U_prec = [0]*Nu*2
        self.Bis_target_Ce = None
        
        #integrator
        self.internal_target = None
        
    def linearized_problem(self, x: list, Bis_target : float):
        """Find the parameters of the function BIS = a * Cep + b * Cer + c to approximate the 3D-hill function."""
        self.find_Ce_target(x, Bis_target)
        Grad = self.Output_grad(Cep = self.Cep_target, Cer = self.Cer_target)
        self.a = Grad['H'][0]
        self.b = Grad['H'][1]
        self.c = self.Bis_target_Ce - (self.a*self.Cep_target + self.b*self.Cer_target)
        
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        self.Output_lin = cas.Function('Output', [Cep, Cer], [self.a*Cep + self.b*Cer + self.c],['Cep', 'Cer'], ['bis'])
        # Optimization problem definition
        w=[]
        self.lbw = []
        self.ubw = []
        J = 0
        gu = []
        gbis =[]
        self.lbg_u = []
        self.ubg_u = []
        self.lbg_bis = []
        self.ubg_bis = []
        
        X0 = cas.MX.sym('X0',8)
        Bis_target = cas.MX.sym('Bis_target')
        U_prec_true = cas.MX.sym('U_prec',2)
        Xk = X0
        for k in range(self.N):
            if k<=self.Nu-1:
                U = cas.MX.sym('U',2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k==0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            elif k>self.Nu-1:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = U
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]
                
            Xk = Pred['x+']
            Hk = self.Output_lin(Cep = Xk[3], Cer = Xk[7])
            bis = Hk['bis']
            
            J+= (bis - Bis_target)**2 + (U-U_prec).T @ self.R @ (U-U_prec)
            #J+= self.Q * (bis - Bis_target)**2 + (U).T @ self.R @ (U)
            
            gu += [U-U_prec]
            gbis += [bis]
            self.lbg_u += self.dumin
            self.ubg_u += self.dumax
            self.ubg_bis += [100]
            
        opts = {'print_time':0, 'verbose':False}#, 'qpsol.printLevel': 'PL_NONE', 'qpoases.verbose':1, 'ipopt.max_iter': 300} 
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, U_prec_true]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*(gu))} #+gbis
        self.solver = cas.qpsol('solver', 'qpoases', prob, opts)
        
        
         
    def find_Ce_target(self, x: list, Bis_target : float):
        """Compute the optimal effect site concentration target to reach Bis_target and respect cost on the Input"""
        
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        
        w = [Cep, Cer]
        w0 =[self.BIS_param[0], self.BIS_param[1]]
        ubw = [20, 20]
        lbw = [0, 0]
        
        Hk = self.Output(Cep=Cep, Cer = Cer)
        bis = Hk['bis']
        J = (bis - Bis_target)**2 #+ self.R[0,0] * (self.C50p - Cep)**2 + self.R[1,1] * (self.C50r - Cer)**2
        g = []
        ubg = []
        lbg = []
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            
        w_opt = sol['x'].full().flatten()
        self.Cep_target = w_opt[0]
        self.Cer_target = w_opt[1]
        Hk = self.Output(Cep=self.Cep_target, Cer = self.Cer_target)
        bis = Hk['bis']
        self.Bis_target_Ce = bis
        
    def one_step(self, x : list, Bis_target : float, Bis_measure: float):
        """Compute the optimal commande to reach Bis_target value using propofol and remifentanil inputs
        Inputs: -x state of the Propofol-Remifentanil PK model
                - Bis_target: in %
        Outputs: - Up: optimal Propofol rate for the next time stamp
                 - Ur: optimal Remifentanil rate for the next time stamp"""
        if self.internal_target is None:
            self.internal_target = Bis_target
            self.linearized_problem(x, Bis_target)
        
        w0 = []
        for k in range(self.Nu-1):
            w0  += self.U_prec[2*(k+1):2*(k+1)+2]
        w0+=self.U_prec[2*(self.Nu-1):2*(self.Nu-1)+2]
        
        self.lbg_bis = [self.internal_target - self.dymin]*self.N
        w0 = [0]*self.Nu*2
        sol = self.solver(x0=w0, p=list(x) + [self.internal_target] + list(self.U_prec[0:2]), lbx=self.lbw, ubx=self.ubw, lbg=self.lbg_u , ubg=self.ubg_u ) #+ self.ubg_bis + self.lbg_bis
        w_opt = sol['x'].full().flatten()
        
        #integrator
        self.internal_target = self.internal_target + self.ki * (Bis_target - Bis_measure)
        
        Up = w_opt[::2]
        Ur = w_opt[1::2]
        self.U_prec = list(w_opt)
        # print for debug
        if False:
            bis = np.zeros(self.N)
            Xk = x
            for k in range(self.N):
                Pred = self.Pred(x=Xk, u=w_opt[2*k:2*k+2])
                Xk = Pred['x+']
                Hk = self.Output_lin(Cep=Xk[3], Cer=Xk[7])
                bis[k] = float(Hk['bis'])
            
            fig, axs = plt.subplots(2,figsize=(14,16))
            axs[0].plot(Up, label = 'propofol')
            axs[0].plot(Ur, label = 'remifentanil')
            axs[1].plot(bis, label = 'bis')
            plt.grid()
            plt.legend()
            plt.show()
            
        return Up[0], Ur[0]
    def update_output(self, new_param):
        Cep = cas.MX.sym('Cep')
        Cer = cas.MX.sym('Cer')
        self.Output_lin = cas.Function('Output', [Cep, Cer], [new_param[0]*Cep + new_param[1]*Cer + new_param[2]],['Cep', 'Cer'], ['bis'])
        # Optimization problem definition
        w=[]
        self.lbw = []
        self.ubw = []
        J = 0
        gu = []
        gbis =[]
        self.lbg_u = []
        self.ubg_u = []
        self.lbg_bis = []
        self.ubg_bis = []
        
        X0 = cas.MX.sym('X0',8)
        Bis_target = cas.MX.sym('Bis_target')
        U_prec_true = cas.MX.sym('U_prec',2)
        Xk = X0
        for k in range(self.N):
            if k<=self.Nu-1:
                U = cas.MX.sym('U',2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k==0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            elif k>self.Nu-1:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = U
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]
                
            Xk = Pred['x+']
            Hk = self.Output_lin(Cep = Xk[3], Cer = Xk[7])
            bis = Hk['bis']
            
            J+= (bis - Bis_target)**2 + (U-U_prec).T @ self.R @ (U-U_prec)
            #J+= self.Q * (bis - Bis_target)**2 + (U).T @ self.R @ (U)
            
            gu += [U-U_prec]
            gbis += [bis]
            self.lbg_u += self.dumin
            self.ubg_u += self.dumax
            self.ubg_bis += [100]
            
        opts = {'print_time':0, 'verbose':False}#, 'qpsol.printLevel': 'PL_NONE', 'qpoases.verbose':1, 'ipopt.max_iter': 300} 
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, U_prec_true]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*(gu))} #+gbis
        self.solver = cas.qpsol('solver', 'qpoases', prob, opts)
    
    def ls_identification(self, Cep: list, Cer: list, BIS: list):
        A = np.array([Cep, Cer, np.ones(len(Cer))]).T
        b = BIS
        fmin = lambda x,A,b: (A @ x - b) @ (A @ x - b).T
        
        bnds = ((None,0),(None,0),(0,None))
        args = (A,b)
        sol = optimize.minimize(fmin, [-5, -3, 50],args, method='SLSQP',
                                tol=1e-10,options={'disp': False})
        sol = sol.x
        sol[0] = max(-10, min(-1, sol[0]))
        sol[1] = max(-10, min(-1, sol[1]))
        sol[2] = max(20, min(100, sol[2]))
        self.update_output(sol)
        return sol
    
class GPC:
    def __init__(self, Ap: list, Bp:list, Ar: list, Br:list, BIS_param: list,
                 Td: float = 96, ts: float = 1, K: float = 2, N: int = 10,
                 Nu: int = 10, lambda_param : float = 1,
                 umax: float = 1e10, umin: float = 0,
                 dumax: float = 0.2, dumin: float = -0.2):
        
        self.BIS_param = BIS_param
        self.umax = umax
        self.umin = umin
        self.dumax = dumax
        self.dumin = dumin
        self.lambda_param = lambda_param
        self.N = N #horizon
        self.Nu = Nu
        self.K = K #ratio between drugs
        
        C50p = self.BIS_param[0]
        C50r = self.BIS_param[1]
        
        # filter model
        
        self.Fd = control.tf([1], [Td, 1])
        self.Fd = control.sample_system(self.Fd, ts)
        self.Fd = control.tfdata(self.Fd)
        self.Af = self.Fd[1][0][0]
        self.Bf = np.flip(self.Fd[0][0][0])
        
        #Propofol system model
        Cp = np.zeros((1,4))
        Cp[0,3] = 1/C50p
        D = np.zeros(1)
        self.syst_p = control.ss(Ap, Bp, Cp, D, True)
        self.tf_p = control.tfdata(self.syst_p)
        self.Az_p = self.tf_p[1][0][0]
        self.Bz_p = np.concatenate(([0], self.tf_p[0][0][0]))
        
        impulse = control.impulse_response(self.syst_p, T = np.arange(0,self.N +2))
        self.hp = impulse[1]
        
        #Remifentanil system model
        Cr = np.zeros((1,4))
        Cr[0,3] = 1/C50r
        self.syst_r = control.ss(Ar, Br, Cr, D, True)
        self.tf_r = control.tfdata(self.syst_r)
        self.Az_r = self.tf_r[1][0][0]
        self.Bz_r = np.concatenate(([0], self.tf_r[0][0][0]))
        
        temp = np.poly1d(np.flip(self.Az_p)) * np.poly1d(np.flip(self.Bz_r)) * self.K
        self.Bz_r_from_p = np.flip(np.array(temp))
        temp = np.poly1d(np.flip(self.Bz_p)) * np.poly1d(np.flip(self.Az_r))
        self.Az_r_from_p = np.flip(np.array(temp))
        
        # Prediction function
        yk = cas.MX.sym('yk', len(self.Az_p)-1)
        uk = cas.MX.sym('uk', len(self.Bz_p))
        
        yk_plus = - cas.MX(self.Az_p[1:]).T @ yk + cas.MX(self.Bz_p).T @ uk
        
        self.Pred_propo = cas.Function('Pred', [yk, uk], [yk_plus],['yk', 'uk'], ['yk_plus'])

        yk = cas.MX.sym('yk', len(self.Az_r)-1)
        uk = cas.MX.sym('uk', len(self.Bz_r))
        
        yk_plus = - cas.MX(self.Az_r[1:]).T @ yk + cas.MX(self.Bz_r).T @ uk
        
        self.Pred_remi = cas.Function('Pred', [yk, uk], [yk_plus],['yk', 'uk'], ['yk_plus'])
        
        # Optimization problem definition
        w=[]
        self.lbw = []
        self.ubw = []
        J = 0
        g=[]
        self.lbg = []
        self.ubg = []
        
        Yp_measure = cas.MX.sym('Yp_measure', len(self.Az_p)-1)
        U_past = cas.MX.sym('U_past', len(self.Bz_p))
        Yp_target = cas.MX.sym('Yp_target')
        
        for k in range(self.N):
            if k<=self.Nu-1:
                U = cas.MX.sym('U',1)
                w   += [U]
                self.lbw += [self.umin]
                self.ubw += [self.umax]

            if k==0:
                yk = Yp_measure
                uk = cas.vertcat(U, U_past[:-1])
            else:
                yk = cas.vertcat(yk_plus, yk[:-1])
                uk = cas.vertcat(U, uk[:-1])
                
            g += [U - uk[1]]
            self.lbg += [self.dumin]
            self.ubg += [self.dumax]    
            
            Pred = self.Pred_propo(yk=yk, uk=uk)
            yk_plus = Pred['yk_plus']
            
            J+= (yk_plus - Yp_target)**2 + self.lambda_param * (uk[0] - uk[1])**2
            
            
            
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[Yp_target, Yp_measure, U_past]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        
        #init the system
        
        self.Yp_list_vague = np.zeros(len(self.Az_p)-1)
        self.Yp_list_hat_filt = np.zeros(len(self.Bz_r_from_p))
        self.Yr_list = np.zeros(len(self.Az_r_from_p)-1)
        self.control_list = np.zeros(len(self.Bz_p))
        self.control_optim = np.zeros(Nu)
        
    def Inverse_Hill(self, estimate_Yr, Bis):
        
        gamma = self.BIS_param[2]
        beta = self.BIS_param[3]
        E0 = self.BIS_param[4]
        Emax = self.BIS_param[5]
        
        #coeff such that Yp**3 + b Yp**2 + c Yp + d =0
        
        temp = (max(0,E0-Bis)/(Emax-E0+Bis))**(1/gamma)
        
        b = 3*estimate_Yr - temp
        c = 3*estimate_Yr**2 - (2 - beta) * estimate_Yr * temp
        d = estimate_Yr**3 - estimate_Yr**2*temp
        
        p = np.poly1d([1, b, c, d])
        
        real_root = 0
        try:
            for el in np.roots(p):
                if np.real(el)==el and np.real(el)>0:
                    real_root = np.real(el)
                    break
        except:
            print('bug')
        estimate_Yp = real_root
        return estimate_Yp
    
    def find_yp_yr(self):
        pred_p = self.Pred_propo(yk = self.Yp_list_vague, uk = self.control_list)
        pred_r = self.Pred_remi(yk = self.Yr_list[:len(self.Az_r)-1], uk = self.K * self.control_list)
        
        new_yp_vague = pred_p['yk_plus']
        new_yr = pred_r['yk_plus']
        
        if float(new_yr)<0:
            print('pause')
        self.Yp_list_vague = np.concatenate((np.array([float(new_yp_vague)]), self.Yp_list_vague[:-1]), axis = 0)
        self.Yr_list = np.concatenate((np.array([float(new_yr)]), self.Yr_list[:-1]), axis = 0)
        
    
    def find_target(self, Yp_list, Yr_list, Bis_target):
        
        Yp_target = cas.MX.sym('target')
        Yp_list = cas.vertcat(*[Yp_target, Yp_list[:-1]])
        Yr_target = self.Yr_list[0] #(- cas.MX(self.Az_r_from_p[1:]).T @ cas.MX(Yr_list) + cas.MX(self.Bz_r_from_p).T @ Yp_list)/cas.MX(self.Az_r_from_p[0])
        
        gamma = self.BIS_param[2]
        beta = self.BIS_param[3]
        E0 = self.BIS_param[4]
        Emax = self.BIS_param[5]
        
        #coeff such that Yp**3 + b Yp**2 + c Yp + d =0
        
        temp = ((E0-Bis_target)/(Emax-E0+Bis_target))**(1/gamma)
        
        b = 3*Yr_target - temp
        c = 3*Yr_target**2 - (2 - beta) * Yr_target * temp
        d = Yr_target**3 - Yr_target**2*temp
        
        J = Yp_target**3 + b * Yp_target**2 + c * Yp_target + d
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': Yp_target}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=1, lbx=0, ubx=4)
          
        w_opt = sol['x'].full().flatten()
        Yp_target = w_opt[0]
        return Yp_target
    
    
    def one_step(self, Bis_measure, Bis_target):
        
        self.find_yp_yr()
        
        self.new_yp_hat = self.Inverse_Hill(self.Yr_list[0], Bis_measure)
        
        self.new_yp_hat_filt = self.Yp_list_vague[0]  + (-self.Af[1]*(self.Yp_list_hat_filt[0] - self.Yp_list_vague[1]) + self.Bf[0]*(self.new_yp_hat - self.Yp_list_vague[0]))
        
        self.Yp_list_hat_filt = np.concatenate(([self.new_yp_hat_filt], self.Yp_list_hat_filt[:-1]), axis = 0)
        
        self.Yp_target = self.Inverse_Hill(self.Yr_list[0], Bis_target) #self.find_target(self.Yp_list_hat_filt, self.Yr_list, Bis_target)
        
        
        w0 = list(np.concatenate((self.control_optim[1:], [self.control_optim[-1]])))
            
            
        sol = self.solver(x0=w0, p = [self.Yp_target] + list(self.Yp_list_hat_filt[:len(self.Az_p)-1]) + list(self.control_list), lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
          
        w_opt = sol['x'].full().flatten()
        
        Up = w_opt[0]
        self.control_list = np.concatenate(([Up], self.control_list[:-1]), axis = 0)
        
        return Up, self.K*Up
        
        