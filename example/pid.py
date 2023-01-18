#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:09 2022

@author: aubouinb
"""



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
