"""
Created on Wed Jan  4 14:30:58 2023

@author: aubouinb
"""

import numpy as np


class Baye_MMPC_best():
    """Implementation of Multiples models Predictive Control with Baye formula for model choice."""

    def __init__(self, estimator_list: list, controller_list: list, K: list = None,
                 Pinit: list = None, hysteresis: float = 0.05, BIS_target: float = 50):
        """Init the class Baye MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - controller_list: list of controllers, each one with its own BIS_model parameters,
                                   corresponding index with the previous list
                - K: gain matrix for baye formula
                - Pinit: prior probability of each model
                - hysteresis: value of the hysteris (as a probability) to switch between models
        """
        self.estimator_list = estimator_list
        self.controller_list = controller_list
        self.N_model = len(controller_list)  # total number of model
        if K is not None:
            self.K = K
        else:
            self.K = 1
        if Pinit is not None:
            self.Pinit = Pinit
        else:
            self.Pinit = np.ones(self.N_model)/self.N_model

        self.P = self.Pinit
        self.hysteresis = hysteresis
        self.BIS_target = BIS_target
        # init best model
        self.idx_best = 13

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimator are updated and the model probabilty are updated using the baye formula.
        Then the best model is choosed as the one with the higher probability with a small hysteresis to
        avoid instability. This model is then used in a Non linear MPC to obtain the next control input.
        """
        # update the estimators
        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)
        # update the probaility
        Ptot = np.sum([np.exp(-self.estimator_list[idx].error * self.K * self.estimator_list[idx].error)*self.P[idx]
                       for idx in range(self.N_model)])
        for idx in range(self.N_model):
            self.P[idx] = np.exp(-self.estimator_list[idx].error * self.K *
                                 self.estimator_list[idx].error)*self.P[idx]/Ptot

        proba_max = max(self.P)
        idx_best_list = [i for i, j in enumerate(self.P) if j == proba_max]
        idx_best_new = idx_best_list[0]
        if abs(self.P[idx_best_new] - self.P[self.idx_best]) > self.hysteresis:
            self.idx_best = idx_best_new

        self.controller_list[self.idx_best].U_prec[0:2] = U_prec
        uP, uR, b = self.controller_list[self.idx_best].one_step(
            self.estimator_list[self.idx_best].x, self.BIS_target, self.estimator_list[self.idx_best].Bis)

        return [uP, uR], self.idx_best


class Baye_MMPC_mean():
    """Implementation of Multiples models Predictive Control with Baye formula for model choice."""

    def __init__(self, estimator_list: list, controller: list, Bis_param_list: list,
                 K: list = None, Pinit: list = None, BIS_target: float = 50):
        """Init the class Baye MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - Bis_param_list: list of the BIS_parameters associated with the BIS_model,
                                   corresponding index with the previous list
                - controller: NMPC controller able to update his model
                - K: gain matrix for baye formula
                - Pinit: prior probability of each model
                - hysteresis: value of the hysteris (as a probability) to switch between models
        """
        self.estimator_list = estimator_list
        self.controller = controller
        self.Bis_param_list = Bis_param_list
        self.N_model = len(estimator_list)  # total number of model
        if K is not None:
            self.K = K
        else:
            self.K = 1
        if Pinit is not None:
            self.Pinit = Pinit
        else:
            self.Pinit = np.ones(self.N_model)/self.N_model

        self.P = self.Pinit
        self.BIS_target = BIS_target
        # init best model
        self.idx_best = 0
        # init weights
        self.W = np.ones(self.N_model)/self.N_model

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimator are updated and the model probabilty are updated using the baye formula.
        Then the best model is choosed as the one with the higher probability with a small hysteresis to
        avoid instability. This model is then used in a Non linear MPC to obtain the next control input.
        """
        # update the estimators
        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)
        # update the probaility
        Ptot = np.sum([np.exp(-self.estimator_list[idx].error * self.K * self.estimator_list[idx].error)*self.P[idx]
                       for idx in range(self.N_model)])
        for idx in range(self.N_model):
            self.P[idx] = np.exp(-self.estimator_list[idx].error * self.K *
                                 self.estimator_list[idx].error)*self.P[idx]/Ptot

        for idx in range(self.N_model):
            self.W[idx] = self.P[idx] / np.sum(self.P)

        self.X = np.sum([self.W[idx] * self.estimator_list[self.idx_best].x for idx in range(self.N_model)], axis=0)
        self.BIS = np.sum([self.W[idx] * self.estimator_list[self.idx_best].Bis for idx in range(self.N_model)], axis=0)
        self.Bis_param = np.sum([self.W[idx] * np.array(self.Bis_param_list[idx])
                                for idx in range(self.N_model)], axis=0)

        self.controller.update_model(self.Bis_param)
        uP, uR, b = self.controller.one_step(self.X, self.BIS_target, self.BIS)
        return [uP, uR], self.W[27]
