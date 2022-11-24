import numpy as np
import control
from matplotlib import pyplot as plt
from matplotlib import cm
import copy
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()

def Hill_function(x:float, x_50:float, gamma:float):
    """simple Hill function"""
    return x**gamma/(x_50**gamma + x**gamma)


def discretize(A:list ,B:list, C:list, D:list, Te:float, method:str):
    """This method discretize a LTI system with form x+=Ax+Bu, y = Cx+Du. given in min^-1
    3 methods are available : "exact": use exponential matrix to obtain the exact diretize system
                              "Euler": Euler method
                              "Rung-Kuta": use order 4 of Rung kunta method to discretize the system."""
    if method=='exact':
        model = control.ss(np.array(A), B, C, D)
        model = control.sample_system(model, Te/60)
        A_d, B_d, C_d, D_d = model.A, model.B, model.C, model.C
    elif method=='euler':
        A_d = np.eye(4) + Te/60 * A
        B_d = Te/60 * B
        C_d = C
        D_d = D
    return  A_d, B_d, C_d, D_d   
                        




class PKmodel:
    """ This class modelize the PK model of propofol or remifentanil drug using Minto-Schnider or Eleveld model 
    with compartiments model."""
    def __init__(self, age: float, height: float,
                 weight: float, sex: bool, lbm: float,
                 drug: str, Te : float =1, model :str = 'Minto',
                 random: bool = False, x0: list = np.ones([4, 1])*1e-4):
        """Input:
            - age: float, in year
            - height:float, in cm
            - weight:float, in kg
            - sex:bool, 1=male, 0= female
            - lbm:float, lean body mass index
            - drug: string, either "Propofol" or "Remifentanil"
            - Te:float, sampling time, in s
            - model: string could be "Minto", "Eleveld" for both drugs, "Marsh" and "Shuttler"" are also available for Propofol
            - random: bool to introduce uncertainties in the model
            - x0: numpy vector(4x1), initial concentration of the compartement model
            """
        self.Te = Te
        
        self.sigma = None
        self.gamma = None
        self.c50p = None
        self.c50r = None
        self.emax = None
        
        if drug == "Propofol":
            if model =='Schnider':
                # see T. W. Schnider et al., “The Influence of Age on Propofol Pharmacodynamics,”
                # Anesthesiology, vol. 90, no. 6, pp. 1502-1516., Jun. 1999, doi: 10.1097/00000542-199906000-00003.

                # Clearance Rates [l/min]
                cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
                cl2 = 1.29 - 0.024 * (age - 53)
                cl3 = 0.836
                # Volume of the compartments [l]
                self.v1 = 4.27
                v2 = 18.9 - 0.391 * (age - 53)
                v3 = 238
                # drug amount transfer rates [1/min]
                ke0 = 0.456
                k1e = 0.456
                
                #variability
                cv_v1 = self.v1*0.0404
                cv_v2 = v2*0.01
                cv_v3 = v3*0.1435
                cv_cl1 = cl1*0.1005
                cv_cl2 = cl2*0.01
                cv_cl3 = cl3*0.1179
                cv_ke = ke0*0.42 #The real value seems to be not available in the article
                
            elif model=='Marsh':
                # see B. Marsh, M. White, N. morton, and G. N. C. Kenny, “Pharmacokinetic model Driven Infusion of Propofol in Children,”
                # BJA: British Journal of Anaesthesia, vol. 67, no. 1, pp. 41–48, Jul. 1991, doi: 10.1093/bja/67.1.41.
    
                self.v1 = 0.228 * weight
                v2 = 0.463 * weight
                v3 = 2.893 * weight
                cl1 = 0.119 * self.v1
                cl2 = 0.112 * self.v1
                cl3 = 0.042 * self.v1
                ke0 = 1.2
                k1e = ke0
                
                #variability
                cv_v1 = self.v1
                cv_v2 = v2
                cv_v3 = v3
                cv_cl1 = cl1
                cv_cl2 = cl2
                cv_cl3 = cl3
                cv_ke = ke0

            elif model=='Schuttler':
            # J. Schüttler and H. Ihmsen, “Population Pharmacokinetics of Propofol: A Multicenter Study,”
            # Anesthesiology, vol. 92, no. 3, pp. 727–738, Mar. 2000, doi: 10.1097/00000542-200003000-00017.

                self.v1 = 9.3 * (weight/70)**0.71 * (age/30)**(-0.39)
                v2 = 44.2 * (weight/70)**0.61
                v3 = 266
                if age<=60:
                    cl1 = 1.44 * (weight/70)**0.75
                else:
                    cl1 = 1.44 * (weight/70)**0.75 - (age-60)*0.045
                cl2 = 2.25 * (weight/70)**0.62*0.6
                cl3 = 0.92 * (weight/70)**0.55
                #no PD model so we reuse schuttler one
                ke0 = 1.2
                k1e = ke0

                #variability
                cv_v1 = self.v1*0.400
                cv_v2 = v2*0.548
                cv_v3 = v3*0.469
                cv_cl1 = cl1*0.374
                cv_cl2 = cl2*0.516
                cv_cl3 = cl3*0.509
                cv_ke = ke0*1.01
                
                
            elif model=='Eleveld':
                # see D. J. Eleveld, P. Colin, A. R. Absalom, and M. M. R. F. Struys, 
                # “Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation,”
                # British Journal of Anaesthesia, vol. 120, no. 5, pp. 942–959, mai 2018, doi: 10.1016/j.bja.2018.01.018.
                
                #function used in the model
                faging = lambda x: np.exp(x * (age - 35))
                fsig = lambda x,C50,gam :  x**gam/(C50**gam + x**gam)
                fcentral = lambda x: fsig(x, 33.6, 1)
                PMA = age + 40/52
                
                fCLmat = fsig(PMA * 52, 42.3, 9.06)
                fQ3mat = fsig(PMA * 52, 68.3, 1)
                fopiate = lambda x: np.exp(x*age)
                
                BMI = weight/(height/100)**2
                BMIref = 70/1.7**2
    
                #reference: male, 70kg, 35 years and 170cm
                if sex:
                    fal_sallami = lambda weightX,ageX,bmiX:  (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
                else:
                    fal_sallami = lambda weightX,ageX,bmiX:  (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)
                
                self.v1 = 6.28 * fcentral(weight)/fcentral(35)
                #self.v1 = self.v1 * (1 + 1.42 * (1 - fcentral(weight)))
                v2 = 25.5 * weight/70 * faging(-0.0156)
                v2ref = 25.5
                v3 = 273 * fal_sallami(weight,age,BMI)/fal_sallami(70,35,BMIref) * fopiate(-0.0138)
                v3ref = 273*fopiate(-0.0138)
                cl1 = (sex*1.79 + (1-sex)*2.1) * (weight/70)**0.75 * fCLmat/fsig(35*52+40, 42.3, 9.06) * fopiate(-0.00286)
                cl2 = 1.75*(v2/v2ref)**0.75 * (1 + 1.30 * (1 - fQ3mat ))
                #cl2 = cl2*0.68
                cl3 = 1.11 * (v3/v3ref)**0.75 * fQ3mat/fsig(35*52+40, 68.3, 1)
                
                ke0 = 0.146*(weight/70)**(-0.25)
                k1e = ke0
                self.sigma = 0
                self.gamma = (1.89+1.47)/2
                self.c50p = 3.08*faging(-0.00635)
                self.c50r = 15
                self.emax = 93
                
                #variability
                cv_v1 = self.v1*0.917
                cv_v2 = v2*0.871
                cv_v3 = v3*0.904
                cv_cl1 = cl1*0.551
                cv_cl2 = cl2*0.643
                cv_cl3 = cl3*0.482
                cv_ke = ke0*1.01               
                
        if drug == "Remifentanil":
            if model =='Minto':
                #  see C. F. Minto et al., “Influence of Age and Gender on the Pharmacokinetics
                # and Pharmacodynamics of Remifentanil: I. Model Development,”
                # Anesthesiology, vol. 86, no. 1, pp. 10–23, Jan. 1997, doi: 10.1097/00000542-199701000-00004.

                # Clearance Rates [l/min]
                cl1 = 2.6 + 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
                cl2 = 2.05 - 0.0301 * (age - 40)
                cl3 = 0.076 - 0.00113 * (age - 40)
                # Volume of the compartments [l]
                self.v1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
                v2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
                v3 = 5.42
                # drug amount transfer rates [1/min]
                ke0 = 0.595 - 0.007 * (age - 40)
                k1e = ke0 #0.456
                self.c50r = 13.1 - 0.148*(age-40)
                
                #variability
                cv_v1 = self.v1*0.26
                cv_v2 = v2*0.29
                cv_v3 = v3*0.66
                cv_cl1 = cl1*0.14
                cv_cl2 = cl2*0.36
                cv_cl3 = cl3*0.41 
                cv_ke = ke0*0.68
                
            elif model=='Eleveld':
                # see D. J. Eleveld et al., “An Allometric Model of Remifentanil Pharmacokinetics and Pharmacodynamics,”
                # Anesthesiology, vol. 126, no. 6, pp. 1005–1018, juin 2017, doi: 10.1097/ALN.0000000000001634.
                
                #function used in the model
                faging = lambda x: np.exp(x * (age - 35))
                fsig = lambda x,C50,gam :  x**gam/(C50**gam + x**gam)
                
                BMI = weight/(height/100)**2
                BMIref = 70/1.7**2
                if sex:
                    FFM = (0.88 + (1-0.88)/(1+(age/13.4)**(-12.7))) * (9270 * weight)/(6680 + 216*BMI)
                    KSEX = 1
                else:
                    FFM = (1.11 + (1 - 1.11)/(1+(age/7.1)**(-1.1))) * (9270 * weight)/(8780 + 244*BMI)
                    KSEX = 1 + 0.47*fsig(age,12,6)*(1 - fsig(age,45,6))
                FFMref = (0.88 + (1-0.88)/(1+(35/13.4)**(-12.7))) * (9270 * 70)/(6680 + 216*BMIref)
                SIZE = (FFM/FFMref)
                KMAT = fsig(weight, 2.88,2)
                KMATref = fsig(70, 2.88,2)
                self.v1 = 5.81 * SIZE * faging(-0.00554)
                v2 = 8.882 * SIZE * faging(-0.00327)
                V2ref = 8.882
                v3 = 5.03 * SIZE * faging(-0.0315)*np.exp(-0.026*(weight - 70))
                V3ref = 5.03
                cl1 = 2.58 * SIZE**0.75 * (KMAT/KMATref)*KSEX*faging(-0.00327)
                cl2 = 1.72 * (v2/V2ref)**0.75 * faging(-0.00554) * KSEX
                cl3 = 0.124 * (v3/V3ref)**0.75 * faging(-0.00554)
                ke0 = 1.09  * faging(-0.0289)
                k1e = ke0
                
                #variability
                cv_v1 = self.v1*0.33
                cv_v2 = v2*0.35
                cv_v3 = v3*1.12
                cv_cl1 = cl1*0.14
                cv_cl2 = cl2*0.237
                cv_cl3 = cl3*0.575
                cv_ke = ke0*1.26

                
        # drug amount transfer rates [1/min]
        self.k10 = cl1 / self.v1
        self.k12 = cl2 / self.v1
        self.k13 = cl3 / self.v1
        k21 = cl2 / v2
        k31 = cl3 / v3

        # Nominal Matrices system definition
        self.A_nom = np.array([[-(self.k10 + self.k12 + self.k13), k21, k31, 0],
                              [self.k12, -k21, 0, 0],
                              [self.k13, 0, -k31, 0],
                              [k1e, 0, 0, -ke0]]) #1/min

        self.B_nom = np.transpose(np.array([[60/self.v1, 0, 0, 0]])) # min/s/L
        self.C = np.array([[1, 0, 0, 0,],[0, 0, 0, 1]])
        self.D = 0
        
        # Introduce inter-patient variability
        if random==True:
            if model=='Marsh':
                print("Warning: the standard deviation of the Marsh model are not know, it is set to 100% for each variable")
                
            self.v1 = truncated_normal(mean=self.v1, sd=cv_v1, low=self.v1/4, upp=self.v1*4)
            cl2 = truncated_normal(mean=cl2, sd=cv_cl2, low=cl2/4, upp=cl2*4)
            cl3 = truncated_normal(mean=cl3, sd=cv_cl3, low=cl3/4, upp=cl3*4)
            self.k10 = truncated_normal(mean=cl1, sd=cv_cl1, low=cl1/4, upp=cl1*4)/ self.v1
            self.k12 = cl2 / self.v1
            self.k13 = cl3 / self.v1
            k21 = cl2 / truncated_normal(mean=v2, sd=cv_v2, low=v2/4, upp=v2*4)
            k31 = cl3 / truncated_normal(mean=v3, sd=cv_v3, low=v3/4, upp=v3*4)
            ke0 = truncated_normal(mean=ke0, sd=cv_ke, low=ke0/4, upp=ke0*4)
            k1e = ke0
            self.A = np.array([[-(self.k10 + self.k12 + self.k13), k21, k31, 0],
                                  [self.k12, -k21, 0, 0],
                                  [self.k13, 0, -k31, 0],
                                  [k1e, 0, 0, -ke0]])
            self.B = np.transpose(np.array([[60/self.v1, 0, 0, 0]])) # min/s/L
        else:
            self.A = self.A_nom
            self.B = self.B_nom
            
        #discretization of the system    
        self.A_d, self.B_d, self.C_d, self.D_d = discretize(self.A, self.B, self.C, self.D, self.Te, method="exact")
        #init outout
        self.x = x0
        self.y = np.dot(self.C, self.x)

    def one_step(self, u: float):
        """Simulate on step of PK model with infusion rate u (mg/s) and return the actual blood and effect site concentration (mg/L)"""
        self.x = np.dot(self.A_d, self.x) + np.dot(self.B_d, u)
        self.y = np.dot(self.C_d, self.x) + np.dot(self.D_d, u)
        return self.y

    def update_coeff_CO(self, CO: float, CO_init : float = 6.5):
        """Update PK coefficient with a linear function of Cardiac output value"""
        coeff = 1
        Anew = copy.deepcopy(self.A)
        Anew[0][0] +=  coeff * Anew[0][0] * (CO - CO_init) / CO_init
        Anew[1][1] +=  coeff * Anew[1][1] * (CO - CO_init) / CO_init
        Anew[2][2] +=  coeff * Anew[2][2] * (CO - CO_init) / CO_init
        Anew[0][1] +=  coeff * Anew[0][1] * (CO - CO_init) / CO_init
        Anew[0][2] +=  coeff * Anew[0][2] * (CO - CO_init) / CO_init
        Anew[1][0] +=  coeff * Anew[1][0] * (CO - CO_init) / CO_init
        Anew[2][0] +=  coeff * Anew[2][0] * (CO - CO_init) / CO_init
        self.A_d, self.B_d, self.C_d, self.D_d = discretize(Anew, self.B, self.C, self.D, self.Te, method="exact")
        
class Bismodel:
    def __init__(self, model: str, BIS_param: list = [None]*6, random: bool = False):
        """Init a Hill curv model to link BIS to Propofol and Remifentanil effect site concentration
        Inputs: - model: only Bouillon is available now
                - BIS_param: Parameter of the BIS model (Propo Remi interaction) list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]
                    - C50p_BIS : Concentration at half effect for propofol effect on BIS
                    - C50r_BIS : Concentration at half effect for remifentanil effect on BIS
                    - gamma_BIS : slope coefficient for the BIS  model,
                    - beta_BIS : interaction coefficient for the BIS model,
                    - E0_BIS : initial BIS,
                    - Emax_BIS : max effect of the drugs on BIS
                - Random: add uncertainties in the PK and PD models to study intra-patient variability, default = False"""
        if BIS_param[0] is not None:
           self.c50p = BIS_param[0]
           self.c50r = BIS_param[1]
           self.gamma = BIS_param[2]
           self.beta = BIS_param[3]
           self.E0 = BIS_param[4]
           self.Emax = BIS_param[5]
           
        elif model=='Bouillon':
            if not random:
                self.c50p = 4.47
                self.c50r = 19.3
                self.beta = 0
                self.gamma = 1.43
                self.E0 = 97.4
                self.Emax = self.E0
            else:
                self.c50p = truncated_normal(mean=4.47, sd=4.47*0.3, low=2, upp=8)
                self.c50r = truncated_normal(mean=19.3, sd=19.3*0.3, low=10, upp=26)
                self.beta = truncated_normal(mean=0, sd=0.5, low=0, upp=3)
                self.gamma = truncated_normal(mean=2.43, sd=2.43*0.3, low=1, upp=5)
                self.E0 = truncated_normal(mean=97.4, sd=97.4*0.1, low=80, upp=100) #standard deviation not given in the article, arbitrary fixed to 10% 
                self.Emax = truncated_normal(mean=97.4, sd=97.4*0.1, low=75, upp=100) #standard deviation not given in the article, arbitrary fixed to 10%               
        self.BIS_param = [self.c50p, self.c50r, self.gamma, self.beta, self.E0, self.Emax]
        
    def compute_bis(self, cep: float, cer: float):
        """Compute BIS function from Propofol and Remifentanil effect site concentration"""
        up = cep / self.c50p
        ur = cer / self.c50r
        if self.beta is None:
            i = up + ur + self.sigma * up * ur
        else:
            Phi = up/(up + ur + 1e-6)
            U_50 = 1 - self.beta * (Phi - Phi**2)
            i = (up + ur)/U_50
        bis = self.E0 - self.Emax * i ** self.gamma / (1 + i ** self.gamma)  
        return np.clip(bis, a_max = 100, a_min = 0)

    def inverse_hill(self, BIS: float, cer: float = 0):
        """Compute Propofol effect site concentration from BIS and Remifentanil Effect site concentration"""
        effect = 1-BIS/self.emax
        I = (effect/(1 - effect))**(1/self.gamma)
        return (I - cer/self.c50r)/(1/self.c50p + self.sigma*cer/(self.c50r*self.c50p))

    def plot_surface(self):
        """Plot the 3D surface of the BIS related to Propofol and Remifentanil effect site concentration"""
        cer = np.linspace(0, 4, 50)
        cep = np.linspace(0, 6, 50)
        cer, cep = np.meshgrid(cer, cep)
        effect = 100 - self.compute_bis(cep, cer)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(cer, cep, effect, cmap=cm.jet, linewidth=0.1)
        ax.set_xlabel('Remifentanil')
        ax.set_ylabel('Propofol')
        ax.set_zlabel('Effect')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        ax.view_init(12, -72)
        plt.show()
        # fig.tight_layout()
        # savepath = "/home/aubouinb/ownCloud/Anesthesie/Science/Bob/3D-Hill.pdf"
        # fig.savefig(savepath, bbox_inches='tight', format='pdf')



class Rassmodel:
    def __init__(self, k1: float = 0.81, k0: float = 0.81):
        self.k0 = k0
        self.k1 = k1
        self.tf = control.tf([-2], [1, 2])
        self.tfd = control.sample_system(self.tf, 1, method='bilinear')
        self.sys = control.tf2ss(self.tfd)
        self.x = 0

    def one_step(self, cer: float):
        u = cer / (self.k1 * cer + self.k0)
        self.x = np.dot(self.sys.A, self.x) + np.dot(self.sys.B, u)
        rass = np.dot(self.sys.C, self.x) + np.dot(self.sys.D, u)
        return rass


class AtraciumPKPD:
    def __init__(self, k1: float = 1, k2: float = 2, k3: float = 10, c50: float = 3.2435, alpha: float = 0.0374,
                 gamma: float = 2.667):
        self.gamma = gamma
        self.c50 = c50
        self.tf = control.tf([k1 * k2 * k3 * alpha ** 3],
                             [1, (k1 + k2 + k3) * alpha, (k1 * k2 + k2 * k3 + k1 * k3) * alpha ** 2,
                              k1 * k2 * k3 * alpha ** 3])
        self.tfd = control.sample_system(self.tf, 1, method='bilinear')
        self.sys = control.tf2ss(self.tfd)
        self.x = np.zeros(3, 1)
        self.ce = 0

    def one_step(self, u):
        self.x = np.dot(self.sys.A, self.x) + np.dot(self.sys.B, u)
        self.ce = np.dot(self.sys.C, self.x) + np.dot(self.sys.D, u)
        return self.ce

    def compute_bis(self):
        effect = 100 * self.c50 ** self.gamma / (self.ce ** self.gamma + self.c50 ** self.gamma)
        return effect

class Hemodynamics:
    def __init__(self, 
                 CO_init: float, 
                 MAP_init: float, 
                 CO_param: list, 
                 MAP_param: list, 
                 ke: list, 
                 Te: float = 1) -> list:
        
        """ Init the Hemodynamic class:
            Inputs:     - CO_init: Initial Cardiac output (L/min)
                        - MAP_init: Initial Mean Arterial Pressure (mmHg)
                        - CO_param: Parameter of the CO Hill functions [C50_P, C50_R, gamma_P, gamma_R, Emax_P, Emax_R]
                        - MAP_param: Parameter of the MAP Hill functions [C50_P, C50_R, gamma_P, gamma_R, Emax_P, Emax_R]
                        - ke: time constant for propofol and remifentanil to hemodynamic [keP, keR]
                        - Te: sampling period (s), default = 1s"""
        self.Te = Te
        self.CO_init = CO_init
        self.MAP_init = MAP_init
        
        self.C50_CO_P = CO_param[0]
        self.C50_CO_R = CO_param[1]
        self.gamma_CO_P = CO_param[2]
        self.gamma_CO_R = CO_param[3]
        self.Emax_CO_P = CO_param[4]*CO_init
        self.Emax_CO_R = CO_param[5]*CO_init
        
        self.C50_MAP_P = MAP_param[0]
        self.C50_MAP_R = MAP_param[1]
        self.gamma_MAP_P = MAP_param[2]
        self.gamma_MAP_R = MAP_param[3]
        self.Emax_MAP_P = MAP_param[4]*MAP_init
        self.Emax_MAP_R = MAP_param[5]*MAP_init
        
        #init another one order system
        self.AP =  -ke[0]
        self.BP = ke[0]
        self.C = 1
        self.D = 0

        self.AP_d = np.eye(1) + self.Te/60 * np.array(self.AP)
        self.BP_d = self.Te/60 * np.array(self.BP)
        self.CP_d = np.array(self.C)
        self.DP_d = np.array(self.D)
        # self.modelP = control.ss(self.AP, self.BP, self.C, self.D)
        # self.modelP = control.sample_system(self.modelP, Te/60)
        
        self.AR = -ke[1]
        self.BR = ke[1]

        self.AR_d = np.eye(1) + self.Te/60 * np.array(self.AR)
        self.BR_d = self.Te/60 * np.array(self.BR)
        self.CR_d = np.array(self.C)
        self.DR_d = np.array(self.D)
        # self.modelR = control.ss(self.AR, self.BR, self.C, self.D)
        # self.modelR = control.sample_system(self.modelR, Te/60)
        
        self.CeP = 0
        self.CeR = 0
    
    def one_step(self, CP_blood: float, CR_blood: float):
        """Simulate one step time of the hemodynamic system"""
        
        #Dynamic system
        self.CeP = self.AP_d*self.CeP + self.BP_d * CP_blood
        self.CeR = self.AR_d*self.CeR + self.BR_d * CR_blood
        
        #Hill functions
        CO = self.CO_init + self.Emax_CO_P * Hill_function(self.CeP, self.C50_CO_P, self.gamma_CO_P) + self.Emax_CO_R * Hill_function(self.CeR, self.C50_CO_R, self.gamma_CO_R)
        MAP = self.MAP_init + self.Emax_MAP_P * Hill_function(self.CeP, self.C50_MAP_P, self.gamma_MAP_P) + self.Emax_MAP_R * Hill_function(self.CeR, self.C50_MAP_R, self.gamma_MAP_R)
        
        return (CO,MAP)
    
class PBPKmodel:
    """ This class modelize the PBPK model of propofol and remifentanil using Abbiati model.
    Return the actual blood and effect site concentration (mg/ml)"""
    def __init__(self, age: float, height: float,
                 weight: float, sex: bool, lbm: float,
                 drug: str, Te : float =1, model :str = 'Minto',
                 health: str = 'bad', x0: list = np.zeros([4, 1])):
        
        # Patient charactristics
        self.BM = weight
        self.height = height
        #Fraction of drug bound to plasma protein
        self.R = 0.7
        
        # Organ mass fraction
        self.w_blood = 0.079
        self.w_bones = 0.143
        self.w_brain = 0.02
        self.w_fat = 0.214
        self.w_gics = 0.0001766
        self.w_heart = 0.005
        self.w_kidneys = 0.004
        self.w_liver = 0.026
        self.w_muscles = 0.4
        self.w_skin = 0.037
        self.w_spleen = 0.00026
        
        #organ density
        self.p_blood = 1.06
        self.p_bones = 1.6
        self.p_brain = 1.035
        self.p_fat = 0.916
        self.p_gics = 1
        self.p_heart = 1.03
        self.p_kidneys = 1.05
        self.p_liver = 1
        self.p_muscles = 1.041
        self.p_skin = 1.3
        self.p_spleen =  1.05
        
        #organ volumes [cm3]
        self.V_PT = self.BM * (self.w_fat/self.p_fat + self.w_bones/self.p_bones + self.w_skin/self.p_skin + self.w_muscles/self.p_muscles)
        self.V_HO = self.BM * (self.w_brain/self.p_brain + self.w_kidneys/self.p_kidneys + self.w_spleen/self.p_spleen)
        self.V_L = self.BM * self.w_liver/self.p_liver
        self.V_gics = self.BM * self.w_gics/self.p_gics
        self.V_P = 0.54 * self.BM * self.w_blood/self.p_blood
        
        #body surface area from Schlich E, Schumm M, Schlich M:
        #"3-D-Body-Scan als anthropometrisches Verfahren zur Bestimmung der spezifischen Korperoberflache". Ernahrungs Umschau 2010;57:178-183
        if sex:
            BSA = 0.000579479 * self.BM**0.38 * height**1.24
        else:
            BSA = 0.000975482 * self.BM**0.46 * height**1.08
         
        #cardiac output
        self.CO = 3.5 * BSA
        
        #blood rate [ml/min]
        if sex:
            self.Q_PV = 0.54 * 0.185 * self.CO
            self.Q_HA = 0.54 * 0.065 * self.CO
            self.Q_HV = 0.54 * 0.25 * self.CO
            self.Q_K = 0.54 * 0.19 * self.CO
        else:
            self.Q_PV = 0.54 * 0.205 * self.CO
            self.Q_HA = 0.54 * 0.065 * self.CO
            self.Q_HV = 0.54 * 0.27 * self.CO
            self.Q_K = 0.54 * 0.17 * self.CO
        
        #Regression data
        if drug=='Remifentanil':
            self.Eff_H = 0.144
            self.Eff_K = 0.394
            self.k_P = 1.732
            self.k_T = 0.063
            self.j_HO_P = 0.044
            self.J_P_HO = 0.662
            self.j_P_PT = 0.479
            self.j_PT_P = 0.279
            self.ke0 = 0.33
        elif drug=='Propofol':
            self.Eff_H = 0
            self.Eff_K = 0
            self.k_P = 0
            self.k_T = 0
            self.j_HO_P =0
            self.J_P_HO = 0
            self.j_P_PT = 0
            self.j_PT_P = 0
            self.ke0 = 0.125    
        #Clearance rates
        self.CL_H = (self.Q_PV) * self.Eff_H
        self.CL_K = self.Q_K * self.Eff_K
        
        a11 = - ((1-self.R)*(self.j_P_PT + self.j_P_HO + self.k_P) + (self.Q_HA + self.Q_PV + self.CL_K)/self.V_P)
        a12 = self.j_PT_P * self.V_PT / self.V_P
        a14 = self.Q_HV / self.V_P
        a15 = self.j_HO_P * self.V_HO/self.V_P
        
        a21 = (1-self.R) * self.j_P_PT * self.V_P/self.V_PT
        a22 = - (self.j_PT_P + self.k_T)
        
        a31 = self.Q_PV / self.V_gics
        a33 = - (self.Q_PV/self.V_gics)
        
        a41 = self.Q_HA / self.V_L
        a43 = self.Q_PV / self.V_L
        a44 = - (self.Q_HV + self.CL_H)/self.V_L
        
        a51 = (1 - self.R) * self.j_P_HO * self.V_P /self.V_HO
        a55 = - self.j_HO_P
        
        self.A = np.array([[a11, a12, 0, a14, a15, 0],
                           [a21, a22, 0, 0, 0, 0],
                           [a31, 0, a33, 0, 0, 0],
                           [a41, 0, a43, a44, 0, 0],
                           [a51, 0, 0, 0, a55, 0],
                           [self.ke0, 0, 0, 0, 0, self.ke0]])
        self.B = np.array([[1/self.V_P],
                           [0],
                           [0],
                           [0],
                           [0],
                           [0]])
        