"""Anesthesia Simulator: translation of the Open-Source Matlab simulator form Ionescu et al."""
# Standard import

# Third party imports
import numpy as np

# Local imports
from .models import PKmodel, Bismodel, Hemodynamics, NMB_PKPD, Rassmodel


class Patient:
    """Define a Patient class able to simulate Anestjesia process."""

    def __init__(self,
                 age: int,
                 height: int,
                 weight: int,
                 gender: bool,
                 CO_base: float = 6.5,
                 MAP_base: float = 90,
                 model_propo: str = 'Schnider',
                 model_remi: str = 'Minto',
                 Ts: float = 1,
                 BIS_param: list = [None]*6,
                 Random_PK: bool = False,
                 Random_PD: bool = False,
                 CO_update: bool = False):
        """
        Initialise a patient class for anesthesia simulation.

        Parameters
        ----------
        age : float
            in year.
        height : float
            in cm.
        weight : float
            in kg.
        gender : bool
            1=male, 0= female.
        CO_base : float, optional
            Initial cardiac output. The default is 6.5L/min.
        MAP_base : float, optional
            Initial Mean Arterial Pressure. The default is 90mmHg.
        model_propo : str, optional
            Name of the Propofol PK Model. The default is 'Schnider'.
        model_remi : str, optional
            Name of the Remifentanil PK Model. The default is 'Minto'.
        Ts : float, optional
            Samplling time (s). The default is 1.
        BIS_param : list, optional
            Parameter of the BIS model (Propo Remi interaction)
            list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS].
            The default is [None]*6.
        Random_PK : bool, optional
            Add uncertainties in the Propodfol and Remifentanil PK models. The default is False.
        Random_PD : bool, optional
            Add uncertainties in the BIS PD model. The default is False.
        CO_update : bool, optional
            Turn on the option to update PK parameters thanks to the CO value. The default is False.

        Returns
        -------
        None.

        """
        self.age = age
        self.height = height
        self.weight = weight
        self.gender = gender
        self.CO_init = CO_base
        self.MAP_init = MAP_base
        self.Ts = Ts
        self.model_propo = model_propo
        self.model_remi = model_remi
        self.BIS_param = BIS_param
        self.Random_PD = Random_PD
        self.Random_PK = Random_PK
        self.CO_update = CO_update

        # LBM computation
        if self.gender == 1:  # homme
            self.lbm = 1.1 * weight - 128 * (weight / height) ** 2
        elif self.gender == 0:  # femme
            self.lbm = 1.07 * weight - 148 * (weight / height) ** 2

        # Init PK models for propofol and remifentanil
        self.PropoPK = PKmodel(age, height, weight, gender, self.lbm,
                               drug="Propofol", Ts=self.Ts, model=model_propo, random=Random_PK)
        self.RemiPK = PKmodel(age, height, weight, gender, self.lbm,
                              drug="Remifentanil", Ts=self.Ts, model=model_remi, random=Random_PK)

        # Init PD model for BIS
        self.BisPD = Bismodel(
            model='Bouillon', BIS_param=BIS_param, random=Random_PD)
        self.BIS_param = self.BisPD.BIS_param

        # Ini Hemodynamic model
        self.Hemo = Hemodynamics(CO_init=CO_base,
                                 MAP_init=MAP_base,
                                 ke=[self.PropoPK.A[3][0]/2, self.RemiPK.A[3][0]/2],
                                 Ts=self.Ts)

        # Init NMB model
        self.nmb_pkpd = NMB_PKPD(Ts=self.Ts)
        # Init RASS model
        self.rass_model = Rassmodel(Ts=self.Ts)

    def one_step(self, uP: float = 0, uR: float = 0, uA: float = 0, uS: float = 0, uD: float = 0,
                 Dist: list = [0]*3, noise: bool = True) -> list:
        """
        Simulate one step time of the patient.

        Parameters
        ----------
        uP : float, optional
            Propofol infusion rate (mg/s). The default is 0.
        uR : float, optional
            Remifentanil infusion rate (µg/s). The default is 0.
        uA : float, optional
            Atracium infusion rate (µg/s). The default is 0.
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
        [self.Cp_blood, self.Cp_ES] = self.PropoPK.one_step(uP)
        [self.Cr_blood, self.Cr_ES] = self.RemiPK.one_step(uR)
        self.NMB = self.nmb_pkpd.one_step(ua=uA, cer=self.Cr_ES)

        # BIS
        self.BIS = self.BisPD.compute_bis(self.Cp_ES, self.Cr_ES)
        # RASS
        self.RASS = self.rass_model.one_step(cer=self.Cr_ES)
        # Hemodynamic
        [self.CO, self.MAP] = self.Hemo.one_step(CP_blood=self.PropoPK.x[0], CR_blood=self.RemiPK.x[0],
                                                 uD=uD, uS=uS/self.weight)

        # disturbances
        self.BIS += Dist[0]
        self.MAP += Dist[1]
        self.CO += Dist[2]

        # update PK model with CO
        if self.CO_update:
            self.PropoPK.update_coeff_CO(self.CO, self.CO_init)
            self.RemiPK.update_coeff_CO(self.CO, self.CO_init)

        # add noise
        if noise:
            self.BIS += np.random.normal(scale=3)
            self.MAP += np.random.normal(scale=0.5)
            self.CO += np.random.normal(scale=0.1)

        return([self.BIS, self.CO, self.MAP, self.RASS, self.NMB])
