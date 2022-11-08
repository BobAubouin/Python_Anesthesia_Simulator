from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from Models import PKmodel, Bismodel, Hemodynamics 
import numpy as np


class Patient:
    def __init__(self, 
                 age : int, 
                 height: int, 
                 weight : int, 
                 gender : bool, 
                 CO_base: float = 6.5, 
                 MAP_base: float = 90, 
                 model_propo: str = 'Schnider',
                 model_remi: str = 'Minto',
                 Te: float = 1,
                 BIS_param: list = [None]*6,
                 Random_PK: bool = False,
                 Random_PD: bool = False,
                 CO_update: bool = False):
        """ Initialise a patient class for anesthesia simulation
        Inputs:     - age (year)
                    - height (cm)
                    - weight (kg)
                    - gender (0 for female, 1 for male)
                    - CO_base: base Cardiac Output (L/min) default = 6.5 L/min
                    - MAP_base: Mean Arterial Pressure (mmHg), default = 90mmHg
                    - model_propo: author of the propofol PK model 'Schnider', 'Marsh, 'Schuttler' or 'Eleveld', default = 'Schnider'
                    - model_remi: author of the Remifentanil PK model 'Minto', 'Eleveld2', default = 'Minto'
                    - Sampling period (s), default = 1s
                    - BIS_param: Parameter of the BIS model (Propo Remi interaction) list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]
                            - C50p_BIS : Concentration at half effect for propofol effect on BIS
                            - C50r_BIS : Concentration at half effect for remifentanil effect on BIS
                            - gamma_BIS : slope coefficient for the BIS  model,
                            - beta_BIS : interaction coefficient for the BIS model,
                            - E0_BIS : initial BIS,
                            - Emax_BIS : max effect of the drugs on BIS
                    - Random: add uncertainties in the PK and PD models to study intra-patient variability, default = False
                    - CO_update: bool to turn on the option to update PK parameters thanks to the CO value, default = False"""

        self.age = age
        self.height = height
        self.weight = weight
        self.gender = gender
        self.CO_init = CO_base
        self.MAP_init = MAP_base
        self.Te = Te
        self.CO_update = CO_update
        # LBM computation
        if self.gender == 1: # homme
            self.lbm = 1.1 * weight - 128 * (weight / height) ** 2
        elif self.gender == 0: #femme
            self.lbm = 1.07 * weight - 148 * (weight / height) ** 2

        # Init PK models for propofol and remifentanil
        self.PropoPK = PKmodel(age, height, weight, gender, self.lbm,
                               drug="Propofol", Te=self.Te, model=model_propo, random=Random_PK)
        self.RemiPK = PKmodel(age, height, weight, gender, self.lbm,
                              drug="Remifentanil", Te=self.Te, model=model_remi, random=Random_PK)
        
        # Init PD model for BIS
        self.BisPD = Bismodel(model = 'Bouillon', BIS_param = BIS_param, random=Random_PD)
        
        #Ini Hemodynamic model
        self.Hemo = Hemodynamics(CO_init = CO_base,
                                 MAP_init = MAP_base,
                                 CO_param = [3, 12, 4.5, 4.5, -0.5, 0.4],
                                 MAP_param = [3.5, 17.1, 3, 4.56, -0.5, -1],
                                 ke = [self.PropoPK.A[3][0]/2, self.RemiPK.A[3][0]/2],
                                 Te = Te)
        

        
        
    def one_step(self, uP: float = 0, uR : float = 0, Dist: list = [0]*3, noise: bool = True) -> list[float]:
        """Run the simulation on one step time.
        Inputs:     - uP: Propofol infusion rate (mg/ml/min)
                    - uR: Remifentanil infusion rate (mg/ml/min)
                    - Disturbance vector on [BIS (%), MAP (mmHg), CO (L/min)]
                    - noise: bool to add measurement n noise on the outputs
                    
        Outputs:    - current BIS (%)
                    - current MAP (mmHg)
                    - current CO (L/min)"""
        #Hemodynamic
        [self.CO, self.MAP] = self.Hemo.one_step(self.PropoPK.x[0], self.RemiPK.x[0])   
            
        #compute PK model            
        [self.Cp_blood, self.Cp_ES] = self.PropoPK.one_step(uP)
        [self.Cr_blood, self.Cr_ES] = self.RemiPK.one_step(uR)                
        #BIS
        self.BIS = self.BisPD.compute_bis(self.Cp_ES[0], self.Cr_ES[0])
        
        #disturbances
        self.BIS += Dist[0]
        self.MAP += Dist[1]
        self.CO += Dist[2]
        
        #update PK model with CO
        if self.CO_update:
            self.PropoPK.update_coeff_CO(self.CO, self.CO_init)
            self.RemiPK.update_coeff_CO(self.CO, self.CO_init) 
        
        #add noise
        if noise:
            self.BIS += np.random.normal(scale=3)
            self.MAP += np.random.normal(scale=0.5)
            self.CO += np.random.normal(scale=0.1)
            
        return([self.BIS, self.CO, self.MAP])       
        
