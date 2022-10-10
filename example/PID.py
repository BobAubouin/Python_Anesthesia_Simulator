#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:09 2022

@author: aubouinb
"""

"PID class to implement PID on python"

class PID():
    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 10, Te: float = 1, umax = 1e10, umin = -1e10):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.N = N
        self.Te = Te
        self.umax = umax
        self.umin = umin
        
        self.integral_part = 0
        self.derivative_part = 0
        self.last_BIS = 0
        
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
