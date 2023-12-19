# Standard import
from typing import Optional
# Third party imports
import numpy as np
import control
import pandas as pd
import casadi as cas
# Local imports
from .pk_models import CompartmentModel
from .pd_models import BIS_model, TOL_model, Hemo_PD_model


class Patient:
    r"""Define a Patient class able to simulate Anesthesia process.

    Parameters
    ----------
    Patient_characteristic: list
        Patient_characteristic = [age (yr), height(cm), weight(kg), gender(0: female, 1: male)]
    co_base : float, optional
        Initial cardiac output. The default is 6.5L/min.
    map_base : float, optional
        Initial Mean Arterial Pressure. The default is 90mmHg.
    model_propo : str, optional
        Name of the Propofol PK Model. The default is 'Schnider'.
    model_remi : str, optional
        Name of the Remifentanil PK Model. The default is 'Minto'.
    ts : float, optional
        Samplling time (s). The default is 1.
    BIS_param : list, optional
        Parameter of the BIS model (Propo Remi interaction)
        list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS].
        The default is None.
    random_PK : bool, optional
        Add uncertainties in the Propodfol and Remifentanil PK models. The default is False.
    random_PD : bool, optional
        Add uncertainties in the BIS PD model. The default is False.
    co_update : bool, optional
        Turn on the option to update PK parameters thanks to the CO value. The default is False.
    save_data_bool : bool, optional
        Save all interns variable at each sampling time in a data frame. The default is True.

    Attributes
    ----------
    age : float
        Age of the patient (yr).
    height : float
        Height of the patient (cm).
    weight : float
        Weight of the patient (kg).
    gender : bool
        0 for female, 1 for male.
    co_base : float
        Initial cardiac output (L/min).
    map_base : float
        Initial mean arterial pressure (mmHg).
    ts : float
        Sampling time (s).
    model_propo : str
        Name of the propofol PK model.
    model_remi : str
        Name of the remifentanil PK model.
    model_bis : str
        Name of the BIS PD model.
    hill_param : list
        Parameter of the BIS model (Propo Remi interaction)
        list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS].
    random_PK : bool
        Add uncertainties in the Propodfol and Remifentanil PK models.
    random_PD : bool
        Add uncertainties in the BIS PD model.
    co_update : bool
        Turn on the option to update PK parameters thanks to the CO value.
    save_data_bool : bool
        Save all interns variable at each sampling time in a data frame.
    lbm : float
        Lean body mass (kg).
    propo_pk : CompartmentModel
        6-comparments model for Propofol.
    remi_pk : CompartmentModel
        5-comparments model for Remifentanil.
    nore_pk : CompartmentModel
        1-comparments model for Norepinephrine.
    bis_pd : BIS_model
        Surface-response model for bis computation.
    tol_pd : TOL_model
        Hierarchical model for TOL computation.
    hemo_pd : Hemo_PD_model
        Hemodynamic model for CO and MAP computation.
    data : pd.DataFrame
        Dataframe containing all the intern variables at each sampling time.
    bis : float
        Bispectral index (%).
    tol : float
        Tolerance of laryngospie probability (0-1).
    co : float
        Cardiac output (L/min).
    map : float
        Mean arterial pressure (mmHg).
    blood_volume : float
        Blood volume (L).
    bis_noise_std : float
        Standard deviation of the BIS noise.
    co_noise_std : float
        Standard deviation of the CO noise.
    map_noise_std : float
        Standard deviation of the MAP noise.


    """

    def __init__(self,
                 patient_characteristic: list,
                 co_base: float = 6.5,
                 map_base: float = 90,
                 model_propo: str = 'Schnider',
                 model_remi: str = 'Minto',
                 model_bis: str = 'Bouillon',
                 ts: float = 1,
                 hill_param: list = None,
                 random_PK: bool = False,
                 random_PD: bool = False,
                 co_update: bool = False,
                 save_data_bool: bool = True):
        """
        Initialise a patient class for anesthesia simulation.

        Returns
        -------
        None.

        """
        self.age = patient_characteristic[0]
        self.height = patient_characteristic[1]
        self.weight = patient_characteristic[2]
        self.gender = patient_characteristic[3]
        self.co_base = co_base
        self.map_base = map_base
        self.ts = ts
        self.model_propo = model_propo
        self.model_remi = model_remi
        self.hill_param = hill_param
        self.random_PK = random_PK
        self.random_PD = random_PD
        self.co_update = co_update
        self.save_data_bool = save_data_bool

        # LBM computation
        if self.gender == 1:  # homme
            self.lbm = 1.1 * self.weight - 128 * (self.weight / self.height) ** 2
        elif self.gender == 0:  # femme
            self.lbm = 1.07 * self.weight - 148 * (self.weight / self.height) ** 2

        # Init PK models for all drugs
        self.propo_pk = CompartmentModel(patient_characteristic, self.lbm, drug="Propofol",
                                         ts=self.ts, model=model_propo, random=random_PK)

        self.remi_pk = CompartmentModel(patient_characteristic, self.lbm, drug="Remifentanil",
                                        ts=self.ts, model=model_remi, random=random_PK)

        self.nore_pk = CompartmentModel(patient_characteristic, self.lbm, drug="Norepinephrine",
                                        ts=self.ts, model=model_remi, random=random_PK)

        # Init PD model for BIS
        self.bis_pd = BIS_model(hill_model=model_bis, hill_param=hill_param, random=random_PD)
        self.hill_param = self.bis_pd.hill_param

        # Init PD model for TOL
        self.tol_pd = TOL_model(model='Bouillon', random=random_PD)

        # Init PD model for Hemodynamic
        self.hemo_pd = Hemo_PD_model(random=random_PD, co_base=co_base, map_base=map_base)

        # init blood loss volume
        self.blood_volume = self.propo_pk.v1
        self.blood_volume_init = self.propo_pk.v1

        # init noise model
        self.bis_noise_std = 3
        self.co_noise_std = 0.1
        self.map_noise_std = 5
        xi = 0.2
        target_peak_fr = 0.03*2*np.pi
        omega = target_peak_fr/np.sqrt(1-2*xi**2)
        noise_filter = control.tf([0.1, 1], [1/omega**2, 2*xi/omega, 1])
        self.noise_filter_d = control.sample_system(noise_filter, self.ts)
        white_noise = np.random.normal(0, self.bis_noise_std, 1000)
        _, self.bis_noise = control.forced_response(self.noise_filter_d, U=white_noise, squeeze=True)
        self.noise_index = 0

        # Init all the output variable
        self.bis = self.bis_pd.compute_bis(0, 0)
        self.tol = self.tol_pd.compute_tol(0, 0)
        self.map = map_base
        self.co = co_base

        # Save data
        if self.save_data_bool:
            self.init_dataframe()
            self.save_data()

    def one_step(self, u_propo: float = 0, u_remi: float = 0, u_nore: float = 0,
                 blood_rate: float = 0, dist: list = [0]*3, noise: bool = True) -> tuple[float, float, float, float]:
        r"""
        Simulate one step time of the patient.

        Parameters
        ----------
        u_propo : float, optional
            Propofol infusion rate (mg/s). The default is 0.
        u_remi : float, optional
            Remifentanil infusion rate (µg/s). The default is 0.
        u_nore : float, optional
            Norepinephrine infusion rate (µg/s). The default is 0.
        blood_rate : float, optional
            Fluid rates from blood volume (mL/min), negative is bleeding while positive is a transfusion.
            The default is 0.
        dist : list, optional
            Disturbance vector on [BIS (%), MAP (mmHg), CO (L/min)]. The default is [0]*3.
        noise : bool, optional
            bool to add measurement noise on the outputs. The default is True.

        Returns
        -------
        bis : float
            Bispectral index(%).
        co : float
            Cardiac output (L/min).
        map : float
            Mean arterial pressure (mmHg).
        tol : float
            Tolerance of Laringoscopy index (0-1).

        """
        # compute PK model
        self.c_es_propo = self.propo_pk.one_step(u_propo)
        self.c_es_remi = self.remi_pk.one_step(u_remi)
        self.c_blood_nore = self.nore_pk.one_step(u_nore)
        # BIS
        self.bis = self.bis_pd.compute_bis(self.c_es_propo, self.c_es_remi)
        # TOL
        self.tol = self.tol_pd.compute_tol(self.c_es_propo, self.c_es_remi)
        # Hemodynamic
        self.map, self.co = self.hemo_pd.compute_hemo(self.propo_pk.x[4:], self.remi_pk.x[4], self.c_blood_nore)
        # disturbances
        self.bis += dist[0]
        self.map += dist[1]
        self.co += dist[2]

        # blood loss effect
        if blood_rate != 0 or self.blood_volume != self.blood_volume_init:
            self.blood_loss(blood_rate)
            self.map *= self.blood_volume/self.blood_volume_init
            self.co *= self.blood_volume/self.blood_volume_init

        # update PK model with CO
        if self.co_update:
            self.propo_pk.update_param_CO(self.co/self.co_base)
            self.remi_pk.update_param_CO(self.co/self.co_base)
            self.nore_pk.update_param_CO(self.co/self.co_base)

        # add noise
        if noise:
            self.add_noise()

        # Save data
        if self.save_data_bool:
            index = int(self.Time/self.ts)
            self.dataframe.loc[index, 'u_propo'] = u_propo
            self.dataframe.loc[index, 'u_remi'] = u_remi
            self.dataframe.loc[index, 'u_nore'] = u_nore
            # compute time
            self.Time += self.ts
            self.save_data()

        return (self.bis, self.co, self.map, self.tol)

    def add_noise(self):
        r"""
        Add noise on the outputs of the model (except TOL).

        The MAP and CO noises are considered white noise while the BIS noise is filtered.
        The filter of the BIS noise is a second order low pass filter with a cut-off frequency of 0.03 Hz.

        """
        # compute filter noise for BIS
        # white noise
        self.noise_index += 1
        if self.noise_index >= len(self.bis_noise):
            self.noise_index = 0
            # new list noise
            white_noise = np.random.normal(0, self.bis_noise_std, 1000)
            _, self.bis_noise = control.forced_response(self.noise_filter_d, U=white_noise, squeeze=True)
        self.bis += self.bis_noise[self.noise_index]
        self.bis = np.clip(self.bis, 0, 100)
        # random noise for MAP and CO
        self.map += np.random.normal(scale=self.map_noise_std)
        self.co += np.random.normal(scale=self.co_noise_std)

    def find_equilibrium(self, bis_target: float, tol_target: float,
                         map_target: float) -> tuple[float, float, float]:
        r"""
        Find the input to meet the targeted outputs at the equilibrium.

        Solve the optimization problem to find the equilibrium input for BIS - TOL:

        .. math::  min_{C_{p,es}, C_{r,es}} \frac{||BIS_{target} - BIS||^2}{100^2} + ||TOL_{target} - TOL||^2

        Then compute the concentration of Noradrenaline to meet the MAP target.

        Finally, compute the input of Propofol, Remifentanil and Noradrenaline to meet the targeted concentration.

        Parameters
        ----------
        bis_target : float
            BIS target (%).
        tol_target : float
            TOL target ([0, 1]).
        map_target:float
            MAP target (mmHg).

        Returns
        -------
        u_propo : float:
            Propofol infusion rate (mg/s).
        u_remi : float:
            Remifentanil infusion rate (µg/s).
        u_nore : float:
            Norepinephrine infusion rate (µg/s).

        """
        # find Remifentanil and Propofol Concentration from BIS and TOL
        cep = cas.MX.sym('cep')  # effect site concentration of propofol in the optimization problem
        cer = cas.MX.sym('cer')  # effect site concentration of remifentanil in the optimization problem

        bis = self.bis_pd.compute_bis(cep, cer)
        tol = self.tol_pd.compute_tol(cep, cer)

        J = (bis - bis_target)**2/100**2 + (tol - tol_target)**2
        w = [cep, cer]
        w0 = [self.bis_pd.c50p, self.bis_pd.c50r/2.5]
        lbw = [0, 0]
        ubw = [50, 50]

        opts = {'ipopt.print_level': 0, 'print_time': 0}
        prob = {'f': J, 'x': cas.vertcat(*w)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw)
        w_opt = sol['x'].full().flatten()
        self.c_blood_propo_eq = w_opt[0]
        self.c_blood_remi_eq = w_opt[1]

        # get Norepinephrine rate from MAP target
        # first compute the effect of propofol and remifentanil on MAP
        map_without_nore, co_without_nore = self.hemo_pd.compute_hemo([self.c_blood_propo_eq, self.c_blood_propo_eq],
                                                                      self.c_blood_remi_eq, 0)
        # Then compute the right nore concentration to meet the MAP target
        wanted_map_effect = map_target - map_without_nore
        self.c_blood_nore_eq = self.hemo_pd.c50_nore_map * (wanted_map_effect /
                                                            (self.hemo_pd.emax_nore_map-wanted_map_effect)
                                                            )**(1/self.hemo_pd.gamma_nore_map)
        _, self.co_eq = self.hemo_pd.compute_hemo([self.c_blood_propo_eq, self.c_blood_propo_eq],
                                                  self.c_blood_remi_eq, self.c_blood_nore_eq)
        # update pharmacokinetics model from co value
        if self.co_update:
            self.propo_pk.update_param_CO(self.co_eq/self.co_base)
            self.remi_pk.update_param_CO(self.co_eq/self.co_base)
            self.nore_pk.update_param_CO(self.co_eq/self.co_base)
        # get rate input
        self.u_propo_eq = self.c_blood_propo_eq / control.dcgain(self.propo_pk.continuous_sys)
        self.u_remi_eq = self.c_blood_remi_eq / control.dcgain(self.remi_pk.continuous_sys)
        self.u_nore_eq = self.c_blood_nore_eq / control.dcgain(self.nore_pk.continuous_sys)

        return self.u_propo_eq, self.u_remi_eq, self.u_nore_eq

    def find_bis_equilibrium_with_ratio(self, bis_target: float,
                                        rp_ratio: float = 2) -> tuple[float, float]:
        r"""
        Find the input of Propofol and Remifentanil to meet the BIS target at the
        equilibrium with a fixed ratio between drugs rates.

        Solve the optimization problem:

        .. math:: J = (bis - bis_{target})^2
        Where :math:`bis` is the BIS computed from the pharmacodynamic model.
        And with the constraints:

        .. math:: u_{propo} = u_{remi} * rp_{ratio}
        .. math:: A_{propo} x_{propo} + B_{propo} u_{propo} = 0
        .. math:: A_{remi} x_{remi} + B_{remi} u_{remi} = 0

        Parameters
        ----------
        bis_target : float
            BIS target (%).
        rp_ratio : float
            remifentanil over propofol rates ratio. The default is 2.

        Returns
        -------
        u_propo : float:
            Propofol infusion rate (mg/s).
        u_remi : float:
            Remifentanil infusion rate (µg/s).

        """
        # solve the optimization problem
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        Ap = self.propo_pk.discretize_sys.A
        Bp = self.propo_pk.discretize_sys.B
        Ar = self.remi_pk.discretize_sys.A
        Br = self.remi_pk.discretize_sys.B

        x0p = np.linalg.solve(Ap-np.eye(6), - Bp * 7 / 20)
        x0r = np.linalg.solve(Ar-np.eye(5), - Br * 7 / 10)
        w0 += x0p[:, 0].tolist()
        w0 += x0r[:, 0].tolist()

        xp = cas.MX.sym('xp', 6, 1)
        xr = cas.MX.sym('xr', 5, 1)
        UP = cas.MX.sym('up', 1)
        w = [xp, xr, UP]
        w0 += [7 / 2]
        lbw = [1e-3] * 12
        ubw = [1e4] * 12

        bis = self.bis_pd.compute_bis(xp[3], xr[3])
        J = (bis_target - bis)**2

        g = [(Ap-np.eye(6)) @ xp + Bp * UP, (Ar-np.eye(5)) @ xr + Br * (rp_ratio * UP)]
        lbg = [-1e-8] * 11
        ubg = [1e-8] * 11
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        self.u_propo_eq = w_opt[-1]
        self.u_remi_eq = rp_ratio * self.u_propo_eq
        return self.u_propo_eq, self.u_remi_eq

    def initialized_at_given_input(self, u_propo: float = 0, u_remi: float = 0, u_nore: float = 0):
        r"""
        Initialize the patient Simulator at the given input as an equilibrium point.

        For each drug, the equilibrium state is computed from the input.
        Then this state is used to intitialze each drug pharmacokinetic model.

        Parameters
        ----------
        u_propo : float, optional
            Propofol infusion rate (mg/s). The default is 0.
        u_remi : float, optional
            Remifentanil infusion rate (µg/s). The default is 0.
        u_nore : float, optional
            Norepinephrine infusion rate (µg/s). The default is 0.

        Returns
        -------
        None.

        """
        self.u_propo_eq = u_propo
        self.u_remi_eq = u_remi
        self.u_nore_eq = u_nore

        self.c_blood_propo_eq = u_propo * control.dcgain(self.propo_pk.continuous_sys)
        self.c_blood_remi_eq = u_remi * control.dcgain(self.remi_pk.continuous_sys)
        self.c_blood_remi_eq = u_nore * control.dcgain(self.nore_pk.continuous_sys)

        # PK models
        x_init_propo = np.linalg.solve(-self.propo_pk.continuous_sys.A, self.propo_pk.continuous_sys.B * u_propo)
        self.propo_pk.x = x_init_propo

        x_init_remi = np.linalg.solve(-self.remi_pk.continuous_sys.A, self.remi_pk.continuous_sys.B * u_remi)
        self.remi_pk.x = x_init_remi

        x_init_nore = np.linalg.solve(-self.nore_pk.continuous_sys.A, self.nore_pk.continuous_sys.B * u_nore)
        self.nore_pk.x = x_init_nore
        if self.save_data_bool:
            self.init_dataframe()
            # recompute output variable
            # BIS
            self.bis = self.bis_pd.compute_bis(self.propo_pk.x[3], self.remi_pk.x[3])
            # TOL
            self.tol = self.tol_pd.compute_tol(self.propo_pk.x[3], self.remi_pk.x[3])
            # Hemodynamic
            self.map, self.co = self.hemo_pd.compute_hemo(self.propo_pk.x[4:], self.remi_pk.x[4], self.nore_pk.x[0])
            self.save_data()

    def initialized_at_maintenance(self, bis_target: float, tol_target: float,
                                   map_target: float) -> tuple[float, float, float]:
        r"""Initialize the patient model at the equilibrium point for the given output value.

        Parameters
        ----------
        bis_target : float
            BIS target (%).
        rass_target : float
            RASS target ([0, -5]).
        map_target:float
            MAP target (mmHg).

        Returns
        -------
        u_propo : float:
            Propofol infusion rate (mg/s).
        u_remi : float:
            Remifentanil infusion rate (µg/s).
        u_nore : float:
            Norepinephrine infusion rate (µg/s).
        """
        # Find equilibrium point

        self.find_equilibrium(bis_target, tol_target, map_target)

        # set them as starting point in the simulator

        self.initialized_at_given_input(u_propo=self.u_propo_eq,
                                        u_remi=self.u_remi_eq,
                                        u_nore=self.u_nore_eq)
        return self.u_propo_eq, self.u_remi_eq, self.u_nore_eq

    def blood_loss(self, fluid_rate: float = 0):
        """Actualize the patient parameters to mimic blood loss.

        Parameters
        ----------
        fluid_rate : float, optional
            Fluid rates from blood volume (mL/min), negative is bleeding while positive is a transfusion.
            The default is 0.

        Returns
        -------
        None.

        """
        fluid_rate = fluid_rate/1000 / 60  # in L/s
        # compute the blood volume
        self.blood_volume += fluid_rate*self.ts

        # Update the models
        self.propo_pk.update_param_blood_loss(self.blood_volume/self.blood_volume_init)
        self.remi_pk.update_param_blood_loss(self.blood_volume/self.blood_volume_init)
        self.nore_pk.update_param_blood_loss(self.blood_volume/self.blood_volume_init)
        self.bis_pd.update_param_blood_loss(self.blood_volume/self.blood_volume_init)

    def init_dataframe(self):
        r"""Initilize the dataframe variable."""
        self.Time = 0
        column_names = ['Time',  # time
                        'BIS', 'TOL', 'MAP', 'CO',  # outputs
                        'u_propo', 'u_remi', 'u_nore',  # inputs
                        'x_propo_1', 'x_propo_2', 'x_propo_3', 'x_propo_4', 'x_propo_5', 'x_propo_6',  # x_PK_propo
                        'x_remi_1', 'x_remi_2', 'x_remi_3', 'x_remi_4', 'x_remi_5',  # x_PK_remi
                        'x_nore', 'blood_volume']  # nore concentration and blood volume

        self.dataframe = pd.DataFrame(columns=column_names)

    def save_data(self, inputs: list = [0, 0, 0]):
        r"""Save all current intern variable as a new line in self.dataframe."""
        # store data

        new_line = {'Time': self.Time,
                    'BIS': self.bis, 'TOL': self.tol, 'MAP': self.map, 'CO': self.co,  # outputs
                    'u_propo': inputs[0], 'u_remi': inputs[1], 'u_nore': inputs[2],  # inputs
                    'x_nore': self.nore_pk.x[0],  # concentration
                    'blood_volume': self.blood_volume}  # blood volume

        line_x_propo = {'x_propo_' + str(i+1): self.propo_pk.x[i] for i in range(6)}
        line_x_remi = {'x_remi_' + str(i+1): self.remi_pk.x[i] for i in range(5)}
        new_line.update(line_x_propo)
        new_line.update(line_x_remi)

        self.dataframe = pd.concat((self.dataframe, pd.DataFrame(new_line, index=[1])), ignore_index=True)

    def full_sim(self, u_propo: Optional[np.array] = None, u_remi: Optional[np.array] = None, u_nore: Optional[np.array] = None,
                 x0_propo: Optional[np.array] = None, x0_remi: Optional[np.array] = None, x0_nore: Optional[np.array] = None) -> pd.DataFrame:
        r"""Simulate the patient model with the given inputs.

        Parameters
        ----------
        u_propo : numpy array, optional
            Propofol infusion rate (mg/s).
        u_remi : numpy array, optional
            Remifentanil infusion rate (µg/s).
        u_nore : numpy array, optional
            Norepinephrine infusion rate (µg/s).
        x0_propo : numpy array, optional
            Initial state of the propofol PK model. The default is zeros.
        x0_remi : numpy array, optional
            Initial state of the remifentanil PK model. The default is zeros.
        x0_nore : numpy array, optional
            Initial state of the norepinephrine PK model. The default is zeros.

        Returns
        -------
        pandas.Dataframes
            Dataframe with all the data.

        """
        if u_propo is None and u_remi is None and u_nore is None:
            raise ValueError('No input given')
        if u_propo is None:
            if u_remi is None:
                u_propo = [0]*len(u_nore)
            else:
                u_propo = [0]*len(u_remi)
        if u_remi is None:
            if u_propo is None:
                u_remi = [0]*len(u_nore)
            else:
                u_remi = [0]*len(u_propo)
        if u_nore is None:
            if u_propo is None:
                u_nore = [0]*len(u_remi)
            else:
                u_nore = [0]*len(u_propo)
        if not len(u_propo) == len(u_remi) == len(u_nore):
            raise ValueError('Inputs must have the same length')

        # init the dataframe
        self.init_dataframe()

        # simulate
        x_propo = self.propo_pk.full_sim(u_propo, x0_propo)
        x_remi = self.remi_pk.full_sim(u_remi, x0_remi)
        x_nore = self.nore_pk.full_sim(u_nore, x0_nore)

        # compute outputs
        bis = self.bis_pd.compute_bis(x_propo[3, :], x_remi[3, :])
        tol = self.tol_pd.compute_tol(x_propo[3, :], x_remi[3, :])
        map, co = self.hemo_pd.compute_hemo(x_propo[4:, :], x_remi[4, :], x_nore[0, :])

        # save data
        df = pd.DataFrame({'Time': np.arange(0, len(u_propo)*self.ts, self.ts),
                           'BIS': bis, 'TOL': tol, 'MAP': map, 'CO': co,
                           'u_propo': u_propo, 'u_remi': u_remi, 'u_nore': u_nore})

        for i in range(6):
            df['x_propo_' + str(i+1)] = x_propo[i, :]
        for i in range(5):
            df['x_remi_' + str(i+1)] = x_remi[i, :]
        df['x_nore'] = x_nore[0, :]

        return df
