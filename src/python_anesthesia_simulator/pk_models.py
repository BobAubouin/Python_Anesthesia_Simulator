# Standard import
import copy
from typing import Optional

# Third party imports
import numpy as np
import control


class CompartmentModel:
    """PKmodel class modelize the PK model of propofol, remifentanil or norepinephrine drug. Simulate the drug distribution in the body.

    Use a 6 compartement model for propofol, a 5 compartement model for remifentanil,
    and a 1 compartement model for norepinephrine.
    The model is a LTI model with the form:

    .. math::  x(k+1)= Ax(k) + Bu(k)
    .. math::  y(k) = Cx(k)

    The state vector is the concentration of the drug in each compartement in the following order:
    blood, muscles, fat, BIS effect-site, MAP effect-site 1, MAP effect-site 2.
    The output of the model is the BIS effect-site concentration.

    Parameters
    ----------
    Patient_characteristic: list
        Patient_characteristic = [age (yr), height(cm), weight(kg), gender(0: female, 1: male)]
    lbm : float
        lean body mass index.
    drug : str
        can be "Propofol", "Remifentanil" or "Norepinephrine".
    model : str, optional
        Could be "Schnider" [Schnider1999]_, "Marsh_initial"[Marsh1991]_, "Marsh_modified"[Struys2000]_,
        "Shuttler"[Schuttler2000]_ or "Eleveld"[Eleveld2018]_ for Propofol.
        "Minto"[Minto1997]_, "Eleveld"[Eleveld2017]_ for Remifentanil.
        only "Beloeil"[Beloeil2005]_ for Norepinephrine.
        The default is "Minto" for Remifentanil and "Schnider" for Propofol.
    ts : float, optional
        Sampling time, in s. The default is 1.
    random : bool, optional
        bool to introduce uncertainties in the model. The default is False.
    x0 : list, optional
        Initial concentration of the compartement model. The default is np.ones([4, 1])*1e-4.
    opiate : bool, optional
        For Elelevd model for propofol, specify if their is a co-administration of opiate (Remifentantil)
        in the same time. The default is True.
    measurement : str, optional
        For Elelevd model for propofol, specify the measuremnt place for blood concentration.
        Can be either 'arterial' or 'venous'. The default is 'arterial'.

    Attributes
    ----------
    ts : float
        Sampling time, in s.
    drug : str
        can be "Propofol", "Remifentanil" or "Norepinephrine".
    A_init : list
        Initial value of the matrix A.
    B_init : list
        Initial value of the matrix B.
    v1 : float
        Volume of the first compartement.
    continuous_sys : control.StateSpace
        Continuous state space model.
    discrete_sys : control.StateSpace
        Discrete state space model.
    x : list
        State vector.
    y : list
        Output vector (hypnotic effect site concentration).

    References
    ----------
    .. [Schnider1999] T. W. Schnider et al., “The Influence of Age on Propofol Pharmacodynamics,”
            Anesthesiology, vol. 90, no. 6, pp. 1502-1516., Jun. 1999, doi: 10.1097/00000542-199906000-00003.
    .. [Marsh1991] B. Marsh, M. White, N. morton, and G. N. C. Kenny,
            “Pharmacokinetic model Driven Infusion of Propofol in Children,”
            BJA: British Journal of Anaesthesia, vol. 67, no. 1, pp. 41–48, Jul. 1991, doi: 10.1093/bja/67.1.41.
    .. [Struys2000] M. M. R. F. Struys et al., “Comparison of Plasma Compartment versus  Two Methods for Effect
            Compartment–controlled Target-controlled Infusion for Propofol,”
            Anesthesiology, vol. 92, no. 2, p. 399, Feb. 2000, doi: 10.1097/00000542-200002000-00021.
    .. [Schuttler2000] J. Schüttler and H. Ihmsen, “Population Pharmacokinetics of Propofol: A Multicenter Study,”
            Anesthesiology, vol. 92, no. 3, pp. 727–738, Mar. 2000, doi: 10.1097/00000542-200003000-00017.
    .. [Eleveld2018] D. J. Eleveld, P. Colin, A. R. Absalom, and M. M. R. F. Struys,
            “Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation”
            British Journal of Anaesthesia, vol. 120, no. 5, pp. 942–959, mai 2018, doi:10.1016/j.bja.2018.01.018.
    .. [Minto1997] C. F. Minto et al., “Influence of Age and Gender on the Pharmacokinetics
            and Pharmacodynamics of Remifentanil: I. Model Development,”
            Anesthesiology, vol. 86, no. 1, pp. 10–23, Jan. 1997, doi: 10.1097/00000542-199701000-00004.
    .. [Eleveld2017] D. J. Eleveld et al., “An Allometric Model of Remifentanil Pharmacokinetics and Pharmacodynamics,”
            Anesthesiology, vol. 126, no. 6, pp. 1005–1018, juin 2017, doi: 10.1097/ALN.0000000000001634.
    .. [Beloeil2005] H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau, “Norepinephrine kinetics and dynamics
            in septic shock and trauma patients,” BJA: British Journal of Anaesthesia,
            vol. 95, no. 6, pp. 782–788, Dec. 2005, doi: 10.1093/bja/aei259.

    """

    def __init__(self, Patient_characteristic: list, lbm: float,
                 drug: str, model: str = None, ts: float = 1,
                 random: bool = False, x0: list = None,
                 opiate=True, measurement="arterial"):
        """Init the class."""
        self.ts = ts
        age = Patient_characteristic[0]
        height = Patient_characteristic[1]
        weight = Patient_characteristic[2]
        gender = Patient_characteristic[3]
        if drug == "Propofol":
            if model is None:
                model = 'Schnider'
            if model == 'Schnider':
                # see T. W. Schnider et al., “The Influence of Age on Propofol Pharmacodynamics,”
                # Anesthesiology, vol. 90, no. 6, pp. 1502-1516., Jun. 1999, doi: 10.1097/00000542-199906000-00003.

                # Clearance Rates [l/min]
                cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * \
                    (lbm - 59) + 0.0264 * (height - 177)
                cl2 = 1.29 - 0.024 * (age - 53)
                cl3 = 0.836
                # Volume of the compartmente [l]
                v1 = 4.27
                v2 = 18.9 - 0.391 * (age - 53)
                v3 = 238
                # drug amount transfer rates [1/min]
                ke0 = 0.456

                # variability
                cv_v1 = v1*0.0404
                cv_v2 = v2*0.01
                cv_v3 = v3*0.1435
                cv_cl1 = cl1*0.1005
                cv_cl2 = cl2*0.01
                cv_cl3 = cl3*0.1179
                cv_ke = ke0*0.42  # The real value seems to be not available in the article

                # estimation of log normal standard deviation
                w_v1 = np.sqrt(np.log(1+cv_v1**2))
                w_v2 = np.sqrt(np.log(1+cv_v2**2))
                w_v3 = np.sqrt(np.log(1+cv_v3**2))
                w_cl1 = np.sqrt(np.log(1+cv_cl1**2))
                w_cl2 = np.sqrt(np.log(1+cv_cl2**2))
                w_cl3 = np.sqrt(np.log(1+cv_cl3**2))
                w_ke0 = np.sqrt(np.log(1+cv_ke**2))

            elif model == 'Marsh_initial' or model == 'Marsh_modified':
                # see B. Marsh, M. White, N. morton, and G. N. C. Kenny,
                # “Pharmacokinetic model Driven Infusion of Propofol in Children,”
                # BJA: British Journal of Anaesthesia, vol. 67, no. 1, pp. 41–48, Jul. 1991, doi: 10.1093/bja/67.1.41.

                v1 = 0.228 * weight
                v2 = 0.463 * weight
                v3 = 2.893 * weight
                cl1 = 0.119 * v1
                cl2 = 0.112 * v1
                cl3 = 0.042 * v1

                if model == 'Marsh_initial':
                    ke0 = 0.26
                else:
                    ke0 = 1.2

                # variability
                # estimation of log normal standard deviation
                # not given in the paper so estimated at 100% for each variable
                w_v1 = np.sqrt(np.log(1+1**2))
                w_v2 = np.sqrt(np.log(1+1**2))
                w_v3 = np.sqrt(np.log(1+1**2))
                w_cl1 = np.sqrt(np.log(1+1**2))
                w_cl2 = np.sqrt(np.log(1+1**2))
                w_cl3 = np.sqrt(np.log(1+1**2))
                w_ke0 = np.sqrt(np.log(1+1**2))

            elif model == 'Schuttler':
                # J. Schüttler and H. Ihmsen, “Population Pharmacokinetics of Propofol: A Multicenter Study,”
                # Anesthesiology, vol. 92, no. 3, pp. 727–738, Mar. 2000, doi: 10.1097/00000542-200003000-00017.

                theta = [None,  # just to get same index than in the paper
                         1.44,  # Cl1 [l/min]
                         9.3,  # v1ref [l]
                         2.25,  # Cl2 [l/min]
                         44.2,  # v2ref [l]
                         0.92,  # Cl3 [l/min]
                         266,  # v3ref [l]
                         0.75,
                         0.62,
                         0.61,
                         0.045,
                         0.55,
                         0.71,
                         -0.39,
                         -0.40,
                         1.61,
                         2.02,
                         0.73,
                         -0.48]

                v1 = theta[2] * (weight/70)**theta[12] * (age/30)**theta[13]
                v2 = theta[4] * (weight/70)**theta[9]
                v3 = theta[6]
                if age <= 60:
                    cl1 = theta[1] * (weight/70)**theta[7]
                else:
                    cl1 = theta[1] * (weight/70)**theta[7] - (age-60)*theta[10]
                cl2 = theta[3] * (weight/70)**theta[8]
                cl3 = theta[5] * (weight/70)**theta[11]

                # no PD model so we reuse Marsh modified one
                ke0 = 1.2

                # variability
                cv_v1 = 0.400
                cv_v2 = 0.548
                cv_v3 = 0.469
                cv_cl1 = 0.374
                cv_cl2 = 0.519
                cv_cl3 = 0.509
                # estimation of log normal standard deviation
                w_v1 = np.sqrt(np.log(1+cv_v1**2))
                w_v2 = np.sqrt(np.log(1+cv_v2**2))
                w_v3 = np.sqrt(np.log(1+cv_v3**2))
                w_cl1 = np.sqrt(np.log(1+cv_cl1**2))
                w_cl2 = np.sqrt(np.log(1+cv_cl2**2))
                w_cl3 = np.sqrt(np.log(1+cv_cl3**2))
                w_ke0 = np.sqrt(np.log(1+1**2))

            elif model == 'Eleveld':
                # see D. J. Eleveld, P. Colin, A. R. Absalom, and M. M. R. F. Struys,
                # “Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation”
                # British Journal of Anaesthesia, vol. 120, no. 5, pp. 942–959, mai 2018, doi:10.1016/j.bja.2018.01.018.

                # reference patient
                AGE_ref = 35
                WGT_ref = 70
                HGT_ref = 1.7
                PMA_ref = (40+AGE_ref*52)/52  # not born prematurely and now 35 yo
                BMI_ref = WGT_ref/HGT_ref**2
                GDR_ref = 1  # 1 male, 0 female

                theta = [None,                    # just to get same index than in the paper
                         6.2830780766822,       # V1ref [l]
                         25.5013145036879,      # V2ref [l]
                         272.8166615043603,     # V3ref [l]
                         1.7895836588902,       # Clref [l/min]
                         1.7500983738779,       # Q2ref [l/min]
                         1.1085424008536,       # Q3ref [l/min]
                         0.191307,              # Typical residual error
                         42.2760190602615,      # CL maturation E50
                         9.0548452392807,       # CL maturation slope [weeks]
                         -0.015633,             # Smaller V2 with age
                         -0.00285709,           # Lower CL with age
                         33.5531248778544,      # Weight for 50 % of maximal V1 [kg]
                         -0.0138166,            # Smaller V3 with age
                         68.2767978846832,      # Maturation of Q3 [weeks]
                         2.1002218877899,       # CLref (female) [l/min]
                         1.3042680471360,       # Higher Q2 for maturation of Q3
                         1.4189043652084,       # V1 venous samples (children)
                         0.6805003109141]       # Higer Q2 venous samples

                # function used in the model
                def faging(x): return np.exp(x * (age - AGE_ref))
                def fsig(x, C50, gam): return x**gam/(C50**gam + x**gam)
                def fcentral(x): return fsig(x, theta[12], 1)

                def fal_sallami(sexX, weightX, ageX, bmiX):
                    if sexX:
                        return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
                    else:
                        return (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)

                PMA = age + 40/52
                BMI = weight/(height/100)**2

                fCLmat = fsig(PMA * 52, theta[8], theta[9])
                fCLmat_ref = fsig(PMA_ref*52, theta[8], theta[9])
                fQ3mat = fsig(PMA * 52, theta[14], 1)
                fQ3mat_ref = fsig(PMA_ref * 52, theta[14], 1)
                fsal = fal_sallami(gender, weight, age, BMI)
                fsal_ref = fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref)

                if opiate:
                    def fopiate(x): return np.exp(x*age)
                else:
                    def fopiate(x): return 1

                # reference: male, 70kg, 35 years and 170cm

                v1 = theta[1] * fcentral(weight)/fcentral(WGT_ref)
                if measurement == "venous":
                    v1 = v1 * (1 + theta[17] * (1 - fcentral(weight)))
                v2 = theta[2] * weight/WGT_ref * faging(theta[10])
                v2ref = theta[2]
                v3 = theta[3] * fsal/fsal_ref * fopiate(theta[13])
                v3ref = theta[3]
                cl1 = (gender*theta[4] + (1-gender)*theta[15]) * (weight/WGT_ref)**0.75 * \
                    fCLmat/fCLmat_ref * fopiate(theta[11])

                cl2 = theta[5]*(v2/v2ref)**0.75 * (1 + theta[16] * (1 - fQ3mat))
                if measurement == "venous":
                    cl2 = cl2*theta[18]

                cl3 = theta[6] * (v3/v3ref)**0.75 * fQ3mat/fQ3mat_ref
                if measurement == "venous":
                    ke0 = 1.24*(weight/70)**(-0.25)
                else:
                    ke0 = 0.146*(weight/70)**(-0.25)

                # Coeff variability
                cv_v1 = v1*0.917
                cv_v2 = v2*0.871
                cv_v3 = v3*0.904
                cv_cl1 = cl1*0.551
                cv_cl2 = cl2*0.643
                cv_cl3 = cl3*0.482

                # log normal standard deviation
                w_v1 = np.sqrt(0.610)
                w_v2 = np.sqrt(0.565)
                w_v3 = np.sqrt(0.597)
                w_cl1 = np.sqrt(0.265)
                w_cl2 = np.sqrt(0.346)
                w_cl3 = np.sqrt(0.209)
                w_ke0 = np.sqrt(0.702)

            # see C. Jeleazcov, M. Lavielle, J. Schüttler, and H. Ihmsen,
            # “Pharmacodynamic response modelling of arterial blood pressure in adult
            # volunteers during propofol anaesthesia,”
            # BJA: British Journal of Anaesthesia, vol. 115, no. 2, pp. 213–226, Aug. 2015, doi: 10.1093/bja/aeu553.
            ke1_map = 0.054
            ke2_map = 0.0695
            w_ke1_map = 1.5
            w_ke2_map = 1.48

        elif drug == "Remifentanil":
            if model is None:
                model = 'Minto'
            if model == 'Minto':
                #  see C. F. Minto et al., “Influence of Age and Gender on the Pharmacokinetics
                # and Pharmacodynamics of Remifentanil: I. Model Development,”
                # Anesthesiology, vol. 86, no. 1, pp. 10–23, Jan. 1997, doi: 10.1097/00000542-199701000-00004.

                # Clearance Rates [l/min]
                cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
                cl2 = 2.05 - 0.0301 * (age - 40)
                cl3 = 0.076 - 0.00113 * (age - 40)

                # Volume of the compartmente [l]
                v1 = 5.1 - 0.0201 * (age-40) + 0.072 * (lbm - 55)
                v2 = 9.82 - 0.0811 * (age-40) + 0.108 * (lbm-55)
                v3 = 5.42

                ke0 = 0.595 - 0.007 * (age - 40)  # [1/min]

                # variability
                cv_v1 = 0.26
                cv_v2 = 0.29
                cv_v3 = 0.66
                cv_cl1 = 0.14
                cv_cl2 = 0.36
                cv_cl3 = 0.41
                cv_ke = 0.68

                # estimation of log normal standard deviation
                w_v1 = np.sqrt(np.log(1+cv_v1**2))
                w_v2 = np.sqrt(np.log(1+cv_v2**2))
                w_v3 = np.sqrt(np.log(1+cv_v3**2))
                w_cl1 = np.sqrt(np.log(1+cv_cl1**2))
                w_cl2 = np.sqrt(np.log(1+cv_cl2**2))
                w_cl3 = np.sqrt(np.log(1+cv_cl3**2))
                w_ke0 = np.sqrt(np.log(1+cv_ke**2))

            elif model == 'Eleveld':
                # see D. J. Eleveld et al., “An Allometric Model of Remifentanil Pharmacokinetics and Pharmacodynamics,”
                # Anesthesiology, vol. 126, no. 6, pp. 1005–1018, juin 2017, doi: 10.1097/ALN.0000000000001634.

                # function used in the model
                def faging(x): return np.exp(x * (age - 35))
                def fsig(x, C50, gam): return x**gam/(C50**gam + x**gam)

                def fal_sallami(sexX, weightX, ageX, bmiX):
                    if sexX:
                        return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
                    else:
                        return (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)

                # reference patient
                AGE_ref = 35
                WGT_ref = 70
                HGT_ref = 1.7
                PMA_ref = (40+AGE_ref*52)/52  # not born prematurely and now 35 yo
                BMI_ref = WGT_ref/HGT_ref**2
                GDR_ref = 1  # 1 male, 0 female

                BMI = weight/(height/100)**2

                SIZE = (fal_sallami(gender, weight, age, BMI)/fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref))

                theta = [None,      # Juste to have the same index as in the paper
                         2.88,
                         -0.00554,
                         -0.00327,
                         -0.0315,
                         0.470,
                         -0.0260]

                KMAT = fsig(weight, theta[1], 2)
                KMATref = fsig(WGT_ref, theta[1], 2)
                if gender:
                    KSEX = 1
                else:
                    KSEX = 1+theta[5]*fsig(age, 12, 6)*(1-fsig(age, 45, 6))

                v1ref = 5.81
                v1 = v1ref * SIZE * faging(theta[2])
                V2ref = 8.882
                v2 = V2ref * SIZE * faging(theta[3])
                V3ref = 5.03
                v3 = V3ref * SIZE * faging(theta[4])*np.exp(theta[6]*(weight - WGT_ref))
                cl1ref = 2.58
                cl2ref = 1.72
                cl3ref = 0.124
                cl1 = cl1ref * SIZE**0.75 * (KMAT/KMATref)*KSEX*faging(theta[3])
                cl2 = cl2ref * (v2/V2ref)**0.75 * faging(theta[2]) * KSEX
                cl3 = cl3ref * (v3/V3ref)**0.75 * faging(theta[2])

                ke0 = 1.09 * faging(-0.0289)

                # log normal standard deviation
                w_v1 = np.sqrt(0.104)
                w_v2 = np.sqrt(0.115)
                w_v3 = np.sqrt(0.810)
                w_cl1 = np.sqrt(0.0197)
                w_cl2 = np.sqrt(0.0547)
                w_cl3 = np.sqrt(0.285)
                w_ke0 = np.sqrt(1.26)
            # see J. F. Standing, G. B. Hammer, W. J. Sam, and D. R. Drover,
            # “Pharmacokinetic–pharmacodynamic modeling of the hypotensive effect of
            # remifentanil in infants undergoing cranioplasty,”
            # Pediatric Anesthesia, vol. 20, no. 1, pp. 7–18, 2010, doi: 10.1111/j.1460-9592.2009.03174.x.
            ke_map = 0.81
            w_ke_map = 0.41
        elif drug == 'Norepinephrine':
            # see H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau, “Norepinephrine kinetics and dynamics
            # in septic shock and trauma patients,” BJA: British Journal of Anaesthesia,
            # vol. 95, no. 6, pp. 782–788, Dec. 2005, doi: 10.1093/bja/aei259.

            v1 = 8.840
            cl1 = 2  # suppose SAPS II = 30 ()

            w_v1 = 1.63
            w_cl1 = 0.974

        if drug == 'Propofol':
            if random:
                if model == 'Marsh':
                    print("Warning: the standard deviation of the Marsh model are not know," +
                          " it is set to 100% for each variable")
                v1 *= np.exp(np.random.normal(scale=w_v1))
                v2 *= np.exp(np.random.normal(scale=w_v2))
                v3 *= np.exp(np.random.normal(scale=w_v3))
                cl1 *= np.exp(np.random.normal(scale=w_cl1))
                cl2 *= np.exp(np.random.normal(scale=w_cl2))
                cl3 *= np.exp(np.random.normal(scale=w_cl3))
                ke0 *= np.exp(np.random.normal(scale=w_ke0))
                ke1_map *= np.exp(np.random.normal(scale=w_ke1_map))
                ke2_map *= np.exp(np.random.normal(scale=w_ke2_map))
            # drug amount transfer rates [1/min]
            k10 = cl1 / v1
            k12 = cl2 / v1
            k13 = cl3 / v1
            k21 = cl2 / v2
            k31 = cl3 / v3

            # Matrices system definition
            A = np.array([[-(k10 + k12 + k13), k12, k13, 0, 0, 0],
                          [k21, -k21, 0, 0, 0, 0],
                          [k31, 0, -k31, 0, 0, 0],
                          [ke0, 0, 0, -ke0, 0, 0],
                          [ke1_map, 0, 0, 0, -ke1_map, 0],
                          [ke2_map, 0, 0, 0, 0, -ke2_map]])/60  # 1/s

            B = np.transpose(np.array([[1/v1, 0, 0, 0, 0, 0]]))  # 1/L
            C = np.array([[0, 0, 0, 1, 0, 0]])
            D = np.array([[0]])

        elif drug == 'Remifentanil':
            if random:
                v1 *= np.exp(np.random.normal(scale=w_v1))
                v2 *= np.exp(np.random.normal(scale=w_v2))
                v3 *= np.exp(np.random.normal(scale=w_v3))
                cl1 *= np.exp(np.random.normal(scale=w_cl1))
                cl2 *= np.exp(np.random.normal(scale=w_cl2))
                cl3 *= np.exp(np.random.normal(scale=w_cl3))
                ke0 *= np.exp(np.random.normal(scale=w_ke0))
                ke_map *= np.exp(np.random.normal(scale=w_ke_map))

            # drug amount transfer rates [1/min]
            k10 = cl1 / v1
            k12 = cl2 / v1
            k13 = cl3 / v1
            k21 = cl2 / v2
            k31 = cl3 / v3

            # Matrices system definition
            A = np.array([[-(k10 + k12 + k13), k12, k13, 0, 0],
                          [k21, -k21, 0, 0, 0],
                          [k31, 0, -k31, 0, 0],
                          [ke0, 0, 0, -ke0, 0],
                          [ke_map, 0, 0, 0, -ke_map]])/60  # 1/s

            B = np.transpose(np.array([[1/v1, 0, 0, 0, 0]]))  # 1/L
            C = np.array([[0, 0, 0, 1, 0]])
            D = np.array([[0]])

        elif drug == 'Norepinephrine':
            if random:
                v1 *= np.exp(np.random.normal(scale=w_v1))
                cl1 *= np.exp(np.random.normal(scale=w_cl1))
            # drug amount transfer rates [1/min]
            k10 = cl1 / v1

            # Matrices system definition
            A = np.array([-k10])/60  # 1/s

            B = np.array([1/v1])  # 1/L
            C = np.array([1])
            D = np.array([0])

        self.A_init = A
        self.B_init = B
        self.v1 = v1
        self.drug = drug
        # Continuous system with blood concentration as output
        self.continuous_sys = control.ss(A, B, C, D)
        # Discretization of the system
        self.discretize_sys = self.continuous_sys.sample(self.ts)

        # init output
        if x0 is None:
            x0 = np.zeros(len(A))  # np.ones(len(A))*1e-3
        self.x = x0
        self.y = np.dot(C, self.x)

    def one_step(self, u: float) -> list:
        """Simulate one step of PK model.

        .. math:: x^+ = Ax + Bu
        .. math:: y = Cx

        Parameters
        ----------
        u : float
            Infusion rate (mg/s for Propofol, µg/s for Remifentanil).

        Returns
        -------
        numpy array
            Actual effect site concentration (µg/mL for Propofol and ng/mL
            for Remifentanil and Norepinephrine).

        """
        self.x = self.discretize_sys.dynamics(None, self.x, u=u)  # first input is ignored
        self.y = self.discretize_sys.output(None, self.x, u=u)  # first input is ignored
        return self.y

    def full_sim(self, u: list, x0: Optional[np.array] = None) -> list:
        """ Simulate PK model with a given input.

        Parameters
        ----------
        u : list
            Infusion rate (mg/s for Propofol, µg/s for Remifentanil and Norepinephrine).
        x0 : numpy array, optional
            Initial state. The default is None.

        Returns
        -------
        numpy array
            List of the states value during the simulation.
            (µg/mL for Propofol and ng/mL for Remifentanil and Norepinephrine).

        """
        if x0 is None:
            x0 = np.zeros(len(self.A_init))
        _, _, x = control.forced_response(self.discretize_sys, U=u, X0=x0, return_x=True)
        return x

    def update_param_CO(self, CO_ratio: float):
        """Update PK coefficient with a linear function of Cardiac output value.

        Parameters
        ----------
        CO : float
            Ratio of Current CO relatively to initial CO.

        Returns
        -------
        None.

        """
        coeff = 1
        Anew = copy.deepcopy(self.A_init)
        if self.drug == 'Propofol' or self.drug == 'Remifentanil':
            Anew[:3, :3] = coeff * Anew[:3, :3] * CO_ratio
        else:
            Anew = Anew * coeff * CO_ratio
        # Continuous system with blood concentration as output
        self.continuous_sys = control.ss(Anew, self.continuous_sys.B, self.continuous_sys.C, self.continuous_sys.D)
        # Discretization of the system
        self.discretize_sys = self.continuous_sys.sample(self.ts)

    def update_param_blood_loss(self, v_ratio: float):
        """Update PK coefficient to mimic a blood loss.

            Update the blodd volume compartment

        Parameters
        ----------
        v_ratio : float
            blood volume as a fraction of init volume, 1 mean no loss, 0 mean 100% loss.

        Returns
        -------
        None.

        """
        Bnew = copy.deepcopy(self.B_init)
        Anew = copy.deepcopy(self.A_init)
        if self.drug == 'Propofol' or self.drug == 'Remifentanil':
            Anew[0][0] /= v_ratio
            Anew[1][0] /= v_ratio
            Anew[2][0] /= v_ratio
        else:
            Anew /= v_ratio
        Bnew /= v_ratio

        # Continuous system with blood concentration as output
        self.continuous_sys = control.ss(Anew, Bnew, self.continuous_sys.C, self.continuous_sys.D)
        # Discretization of the system
        self.discretize_sys = self.continuous_sys.sample(self.ts)
