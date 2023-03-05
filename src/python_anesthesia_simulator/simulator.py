"""Python Anesthesia Simulator.

@author: Aubouin--Pairault Bob 2023, bob.aubouin@tutanota.com
"""
# Standard import

# Third party imports
import numpy as np
import control
import pandas as pd
import casadi as cas
# Local imports
from .pk_models import CompartmentModel
from .pd_models import BIS_model, TOL_model, fsig


class Patient:
    """Define a Patient class able to simulate Anesthesia process."""

    def __init__(self,
                 patient_characteristic: list,
                 co_base: float = 6.5,
                 map_base: float = 90,
                 model_propo: str = 'Schnider',
                 model_remi: str = 'Minto',
                 ts: float = 1,
                 hill_param: list = None,
                 random_PK: bool = False,
                 random_PD: bool = False,
                 co_update: bool = False,
                 save_data: bool = True):
        """
        Initialise a patient class for anesthesia simulation.

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
        self.save_data = save_data

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
        self.bis_pd = BIS_model(hill_model='Bouillon', hill_param=hill_param, random=random_PD)
        self.hill_param = self.bis_pd.hill_param

        # Init PD model for TOL
        self.tol_pd = TOL_model(model='Bouillon', random=random_PD)

        # init blood loss constant
        self.blood_loss_tf = control.tf([1], [60, 1])
        self.blood_loss_tfd = control.sample_system(self.blood_loss_tf, self.ts)
        self.blood_loss_sysd = control.tf2ss(self.blood_loss_tfd)
        self.x_v_loss = 0
        self.v_loss = 0
        self.blood_trans_tf = control.tf([1], [600, 1])
        self.blood_trans_tfd = control.sample_system(self.blood_trans_tf, self.ts)
        self.blood_trans_sysd = control.tf2ss(self.blood_trans_tfd)
        self.x_v_trans = 0
        self.v_trans = 0

        # Init all the output variable
        self.bis = self.bis_pd.compute_bis(0, 0)
        self.tol = self.tol_pd.compute_tol(0, 0)
        self.map = map_base
        self.co = co_base

        # Save data ?
        if self.save_data:
            # Time variable which will be stored
            self.Time = 0
            column_names = ['Time',  # time
                            'BIS', 'TOL', 'MAP', 'CO', 'NMB',  # outputs
                            'u_propo', 'u_remi', 'u_dopamine', 'u_snp', 'u_atracium',  # inputs
                            'x_propo_1', 'x_propo_2', 'x_propo_3', 'x_propo_4',  # x_PK_propo
                            'x_remi_1', 'x_remi_2', 'x_remi_3', 'x_remi_4',  # x_PK_remi
                            'c_blood_nore', 'x_rass', 'x_hemo_c_propo', 'x_hemo_c_remi']  # x nmb, rass and hemo

            self.dataframe = pd.DataFrame(columns=column_names)

    def one_step(self, u_propo: float = 0, u_remi: float = 0, u_nore: float = 0, uS: float = 0, uD: float = 0,
                 Dist: list = [0]*3, noise: bool = True) -> list:
        """
        Simulate one step time of the patient.

        Parameters
        ----------
        u_propo : float, optional
            Propofol infusion rate (mg/s). The default is 0.
        u_remi : float, optional
            Remifentanil infusion rate (µg/s). The default is 0.
        u_nore : float, optional
            Norepinephrine infusion rate (µg/s). The default is 0.
        uS : float, optional
            Sodium Nitroprucide rate (mg/s). The default is 0.
        uD : float, optional
            Dopamine rate (µg/s). The defauls is 0.
        Dist : list, optional
            Disturbance vector on [BIS (%), MAP (mmHg), CO (L/min)]. The default is [0]*3.
        noise : bool, optional
            bool to add measurement n noise on the outputs. The default is True.

        Returns
        -------
        output : list
            [BIS, MAP, CO] : current BIS (%), MAP (mmHg) ,and CO (L/min)

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

        # disturbances
        self.bis += Dist[0]
        self.map += Dist[1]
        self.co += Dist[2]

        # update PK model with CO
        if self.co_update:
            self.propo_pk.update_param_CO(self.CO/self.CO_init)
            self.remi_pk.update_param_CO(self.CO/self.CO_init)
            self.nore_pk.update_param_CO(self.CO/self.CO_init)

        # add noise
        if noise:
            self.bis += np.random.normal(scale=3)
            self.map += np.random.normal(scale=0.5)
            self.co += np.random.normal(scale=0.1)

        # Save data ?
        if self.save_data:
            # compute time
            self.Time += self.ts
            # store data

            new_line = {'Time': self.Time,
                        'BIS': self.bis, 'TOL': self.tol, 'MAP': self.map, 'CO': self.co,  # outputs
                        'u_propo': u_propo, 'u_remi': u_remi, 'u_nore': u_nore, 'u_snp': uS,  # inputs
                        'c_blood_nore': self.c_blood_nore}  # concentration

            line_x_propo = {'x_propo_' + str(i+1): self.propo_pk.x[i] for i in range(4)}
            line_x_remi = {'x_remi_' + str(i+1): self.remi_pk.x[i] for i in range(4)}
            # line_x_g11 = {'x_hemo_g11_' + str(i+1): self.Hemo.x11[i, 0] for i in range(len(self.Hemo.x11))}
            # line_x_g12 = {'x_hemo_g12_' + str(i+1): self.Hemo.x12[i, 0] for i in range(len(self.Hemo.x12))}
            # line_x_g21 = {'x_hemo_g21_' + str(i+1): self.Hemo.x21[i, 0] for i in range(len(self.Hemo.x21))}
            # line_x_g22 = {'x_hemo_g22_' + str(i+1): self.Hemo.x22[i, 0] for i in range(len(self.Hemo.x22))}
            new_line.update(line_x_propo)
            new_line.update(line_x_remi)

            self.dataframe = pd.concat((self.dataframe, pd.DataFrame(new_line)), ignore_index=True)

        return([self.bis, self.co, self.map, self.tol])

    def find_equilibrium(self, bis_target: float, tol_target: float, map_target: float,
                         co_target: float) -> list:
        """
        Find the input to meet the targeted output at the equilibrium.

        Parameters
        ----------
        bis_target : float
            BIS target (%).
        tol_target : float
            TOL target ([0, 1]).
        map_target:float
            MAP target (mmHg).
        co_target : float
            CO target (L/min).
        Returns
        -------
        list
            list of input [up, ur, ud, us, ua] with the respective units [mg/s, µg/s, µg/s, µg/s, µg/s].

        """
        # find Remifentanil and Propofol Concentration from BIS and TOL
        cep = cas.MX('cep')  # effect site concentration of propofol in the optimization problem
        cer = cas.MX('cer')  # effect site concentration of remifentanil in the optimization problem
        # temp = (max(0, self.bis_pd.E0-bis_target)/(self.bis_pd.Emax-self.bis_pd.E0+bis_target))**(1/self.bis_pd.gamma)

        # Yp = cep / self.bis_pd.c50p
        # Yr = cer / self.bis_pd.c50r
        # b = 3*Yr - temp
        # c = 3*Yr**2 - (2 - self.BisPD.beta) * Yr * temp
        # d = Yr**3 - Yr**2*temp

        # post_opioid = self.tol_pd.pre_intensity * (1 - fsig(cer, self.tol_pd.c50r*self.tol_pd.pre_intensity,
        #                                                     self.tol_pd.gamma_r))
        # tol = 1 - fsig(cep, self.tol_pd.c50p*post_opioid, self.tol_pd.gamma_p)

        bis = self.bis_pd.hill_curve(cep, cer)
        tol = self.tol_pd.one_step(cep, cer)

        J = (bis - bis_target)**2/100**2 + (tol - tol_target)**2
        w = [cep, cer]
        w0 = [self.bis_pd.c50p, self.bis_pd.c50r]
        lbw = [0, 0]
        ubw = [50, 50]

        opts = {'ipopt.print_level': 0, 'print_time': 0}
        prob = {'f': J, 'x': cas.vertcat(*w)}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw)
        w_opt = sol['x'].full().flatten()
        self.c_blood_propo_eq = w_opt[0]
        self.c_blood_remi_eq = w_opt[1]

        # get Dopamine and Sodium nitroprusside from CO and MAP target

        # update pharmacokinetics model from co value
        if self.co_update:
            self.propo_pk.update_param_CO(self.co_eq/self.co_base)
            self.remi_pk.update_param_CO(self.co_eq/self.co_base)
        # get rate input
        self.u_propo_eq = self.c_blood_propo_eq / control.dcgain(self.propo_pk.continuous_sys)
        self.u_remi_eq = self.c_blood_remi_eq / control.dcgain(self.remi_pk.continuous_sys)

        return self.u_propo_eq, self.u_remi_eq

    def initialized_at_given_input(self, u_propo: float = 0, u_remi: float = 0, u_nore: float = 0,
                                   us: float = 0, ua: float = 0):
        """
        Initialize the patient Simulator at the given input as an equilibrium point.

        Parameters
        ----------
        u_propo : float, optional
            Propofol infusion rate (mg/s). The default is 0.
        u_remi : float, optional
            Remifentanil infusion rate (µg/s). The default is 0.
        u_nore : float, optional
            Norepinephrine infusion rate (µg/s). The default is 0.
        uS : float, optional
            Sodium Nitroprucide rate (mg/s). The default is 0.
        us : float, optional
            Sodium Nitroprucide rate (µg/s). The default is 0.
        ua : float, optional
            Atracium infusion rate (µg/s). The default is 0.

        Returns
        -------
        None.

        """
        self.u_propo_eq = u_propo
        self.u_remi_eq = u_remi
        self.u_nore_eq = u_nore
        self.uS_eq = us
        self.uA_eq = ua

        self.c_blood_propo_eq = u_propo * control.dcgain(self.propo_pk.continuous_sys)
        self.c_blood_remi_eq = u_remi * control.dcgain(self.remi_pk.continuous_sys)
        self.c_blood_remi_eq = u_nore * control.dcgain(self.nore_pk.continuous_sys)

        # PK models
        x_init_propo = np.linalg.solve(-self.propo_pk.continuous_sys.A, self.propo_pk.continuous_sys.B * u_propo)
        self.propo_pk.x = x_init_propo

        x_init_remi = np.linalg.solve(-self.remi_pk.continuous_sys.A, self.remi_pk.continuous_sys.B * u_propo)
        self.remi_pk.x = x_init_remi

        x_init_nore = np.linalg.solve(-self.nore_pk.continuous_sys.A, self.nore_pk.continuous_sys.B * u_nore)
        self.u_nore.x = x_init_nore

        # Effect site models
        self.bis_pd.c_es_propo = self.c_blood_propo_eq
        self.bis_pd.c_es_remi = self.c_blood_remi_eq

        # Hemo dynamics
        self.Hemo.CeP = self.Cp_ES_eq
        self.Hemo.CeR = self.Cr_ES_eq

        x11_init = np.linalg.solve(-(self.Hemo.A11 - np.eye(len(self.Hemo.A11))),
                                   self.Hemo.B11 * self.uD_eq)
        self.Hemo.x11 = x11_init

        x12_init = np.linalg.solve(-(self.Hemo.A12 - np.eye(len(self.Hemo.A12))),
                                   self.Hemo.B12 * self.uD_eq)
        self.Hemo.x12 = x12_init

        x21_init = np.linalg.solve(-(self.Hemo.A21 - np.eye(len(self.Hemo.A21))),
                                   self.Hemo.B21 * self.uD_eq)
        self.Hemo.x21 = x21_init

        x22_init = np.linalg.solve(-(self.Hemo.A22 - np.eye(len(self.Hemo.A22))),
                                   self.Hemo.B22 * self.uD_eq)
        self.Hemo.x22 = x22_init

    def initialized_at_maintenance(self, bis_target: float, rass_target: float, map_target: float,
                                   co_target: float, nmb_target: float = None):
        """Initialize the patient model at the equilibrium point for the given output value.

        Parameters
        ----------
        bis_target : float
            BIS target (%).
        rass_target : float
            RASS target ([0, -5]).
        map_target:float
            MAP target (mmHg).
        co_target : float
            CO target (L/min).
        nmb_target : float, optional
            NMB target (%). The default is None.

        Returns
        -------
        None.

        """
        # Find equilibrium point

        self.find_equilibrium(bis_target, rass_target, map_target, co_target)

        # set them as starting point in the simulator

        self.initialized_at_given_input(u_propo=self.u_propo_eq,
                                        u_remi=self.u_remi_eq,
                                        u_nore=self.u_nore_eq,
                                        us=self.uS_eq,
                                        ua=self.uA_eq)

    def blood_loss(self, blood_loss_target: float = 0, transfusion_target: float = 0, mode: int = 0):
        """Actualize the patient parameters to mimic blood loss.

        Parameters
        ----------
        blood_loss_target : float, optional
            DESCRIPTION. The default is 0.
        transfusion_target : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        # compute the blood loss ans transfusion dynamic
        self.x_v_loss = self.blood_loss_sysd.A * self.x_v_loss + self.blood_loss_sysd.B * blood_loss_target
        self.x_v_trans = self.blood_trans_sysd.A * self.x_v_trans + self.blood_trans_sysd.B * transfusion_target

        self.v_loss = self.blood_loss_sysd.C * self.x_v_loss + self.blood_loss_sysd.D * blood_loss_target
        self.v_trans = self.blood_trans_sysd.C * self.x_v_trans + self.blood_trans_sysd.D * transfusion_target

        # Update the models
        self.PropoPK.update_coeff_blood_loss(v_loss=self.v_loss - self.v_trans, mode=mode)
        self.RemiPK.update_coeff_blood_loss(v_loss=self.v_loss - self.v_trans, mode=mode)
