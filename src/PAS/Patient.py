from Models import Pkmodel, Bismodel, Rassmodel, Hemodynamics
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
                 C50p_BIS : float = 4.5,
                 C50r_BIS : float = 12,
                 gamma_BIS : float = None,
                 beta_BIS : float = None,
                 E0_BIS  : float = None,
                 Emax_BIS  : float = None,
                 Random: bool = False,
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
        self.PropoPK = Pkmodel(age, height, weight, gender, self.lbm, drug="Propofol", Te=self.Te, model=model_propo)
        self.PropoPK = Pkmodel(age, height, weight, gender, self.lbm, drug="Propofol", Te=self.Te, model=model_remi)
        
        # Init PD model for BIS
        if gamma_BIS is None:
            if model_propo=='Eleveld':
                self.BisPD = Bismodel(C50p_BIS, C50r_BIS, self.PropoPK.gamma, self.PropoPK.sigma)
            else:
                self.BisPD = Bismodel(C50p_BIS, C50r_BIS)
        else:
            self.BisPD = Bismodel(c50p = C50p_BIS, c50r = C50r_BIS, gamma = gamma_BIS, Emax = Emax_BIS, E0 = E0_BIS, beta = beta_BIS)
            
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
            self.BIS += np.random.randn(1)*2
            self.MAP += np.random.randn(1)*0.5
            self.CO += np.random.randn(1)*0.1
 
        return([self.BIS, self.CO, self.MAP])       
        

def compute_disturbances(time: float, dist_profil: str = 'realistic'):
    """Give the value of the distubance profil for a given time
    Inputs: - Time: in seconde
            - disturbance profil, can be: 'realistic', 'simple' or 'step', default = 'realistic'
    Outputs:- dist_bis, dist_map, dist_co: respectively the additive disturbance to add to the BIS, MAP and CO signals
    """
    
    if dist_profil=='realistic':
        Disturb_point = np.array([[0,     0,  0, 0  ], #time, BIS signal, MAP, CO signals
                                       [9.9,   0,  0, 0  ],
                                       [10,   20, 10, 0.6],
                                       [12,   20, 10, 0.6],
                                       [13,    0,  0, 0  ],
                                       [19.9,  0,  0, 0  ],
                                       [20.2, 20, 10, 0.5],
                                       [21,   20, 10, 0.5],
                                       [21.5,  0,  0, 0  ],
                                       [26,  -20,-10,-0.8],
                                       [27,   20, 10, 0.9],
                                       [28,   10,  7, 0.2],
                                       [36,   10,  7, 0.2],
                                       [37,   30, 15, 0.8],
                                       [37.5, 30, 15, 0.8],
                                       [38,   10,  5, 0.2],
                                       [41,   10,  5, 0.2],
                                       [41.5, 30, 10, 0.5],
                                       [42,   30, 10, 0.5],
                                       [43,   10,  5, 0.2],
                                       [47,   10,  5, 0.2],
                                       [47.5, 30, 10, 0.9],
                                       [50,   30,  8, 0.9],
                                       [51,   10,  5, 0.2],
                                       [56,   10,  5, 0.2],
                                       [56.5,  0,  0, 0  ]])    
    elif dist_profil=='simple':
        Disturb_point = np.array([[0,     0,  0, 0  ], #time, BIS signal, MAP, CO signals
                                       [19.9,  0,  0, 0  ],
                                       [20,   20,  5, 0.3],
                                       [23,   20, 10, 0.6],
                                       [24,   15, 10, 0.6],
                                       [26, 12.5,  6, 0.4],
                                       [30, 10.5,  4, 0.3],
                                       [37,   10,  4, 0.3],
                                       [40,    4,  2, 0.1],
                                       [45,  0.5,0.1,0.01],
                                       [50,    0,  0,   0]])
    elif dist_profil=='step':
        Disturb_point = np.array([[0,     0,  0,   0], #time, BIS signal, MAP, CO signals
                                       [4.9,   0,  0,   0],
                                       [5,    10,  5, 0.3],
                                       [15,   10,  5, 0.3],
                                       [15.1,  0,  5,   0],
                                       [20,    0,  0,   0]])

    
    dist_bis = np.interp(time*60, Disturb_point[:,0], Disturb_point[:,1])
    dist_map = np.interp(time*60, Disturb_point[:,0], Disturb_point[:,2])
    dist_co = np.interp(time*60, Disturb_point[:,0], Disturb_point[:,3])
    
    return dist_bis, dist_map, dist_co

def compute_control_metrics(Bis: list, Te: float = 1, phase: str = 'Maintenance', latex_output: bool =False):
    """This function compute the control metrics initially proposed in "C. M. Ionescu, R. D. Keyser, B. C. Torrico,
    T. D. Smet, M. M. Struys, and J. E. Normey-Rico, “Robust Predictive Control Strategy Applied for Propofol Dosing
    Using BIS as a Controlled Variable During Anesthesia,” IEEE Transactions on Biomedical Engineering, vol. 55, no.
    9, pp. 2161–2170, Sep. 2008, doi: 10.1109/TBME.2008.923142."
    
    Inputs: - BIS: list of BIS value over time
            - Te: sampling time in second
            - phase: either 'Maintenance' or 'Induction'
            - latex_output bool to print the latex code to create table of the results
    Outputs: - TT : observed time-to-target (in seconds) required for reaching first time the target interval of [55,45] BIS values
             - BIS-NADIR: the lowest observed BIS value during induction phase
             - ST10: settling time on the reference BIS value, defined within ± 5BIS(i.e., between 45 and 55 BIS) and stay within this BIS range
             - ST20: settling time on the reference BIS value, defined within ± 10BIS(i.e., between 40 and 60 BIS) and stay within this BIS range
             - US: undershoot, defined as the BIS value that exceeds the limit of the defined BIS interval, namely, the 45 BIS value.
             """
    return 