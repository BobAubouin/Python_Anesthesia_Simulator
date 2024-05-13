# Third party imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def fsig(x, c50, gam): return x**gam/(c50**gam + x**gam)  # quick definition of sigmoidal function


class BIS_model:
    r"""Surface Response model to link Propofol and Remifentanil blood concentration to BIS.

    equation:

    .. math:: BIS = E0 + Emax * \frac{U^\gamma}{1+U^\gamma}
    .. math:: U = \frac{U_p + U_r}{1 - \beta \theta + \beta \theta^2}
    .. math:: U_p = \frac{C_{p,es}}{C_{p,50}}
    .. math:: U_r = \frac{C_{r,es}}{C_{r,50}}
    .. math:: \theta = \frac{U_p}{U_r+U_p}

    Parameters
    ----------
    hill_model : str, optional
        'Bouillon' [Bouillon2004]_ and 'Aubouin' [Aubouin2023] are available.
        Ignored if hill_param is specified. Default is 'Bouilllon'.
    hill_param : list, optional
        Parameter of the Hill model (Propo Remi interaction)
        list [c50p_BIS, c50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]:
            - c50p_BIS : Concentration at half effect for propofol effect on BIS (µg/mL)
            - c50r_BIS : Concentration at half effect for remifentanil effect on BIS (ng/mL)
            - gamma_BIS : slope coefficient for the BIS  model,
            - beta_BIS : interaction coefficient for the BIS model,
            - E0_BIS : initial BIS,
            - Emax_BIS : max effect of the drugs on BIS.
        The default is None.
    random : bool, optional
        Add uncertainties in the parameters. Ignored if Hill_cruv is specified. The default is False.
    ts : float, optional
        Sampling time, in s. The default is 1.

    Attributes
    ----------
    c50p : float
        Concentration at half effect for propofol effect on BIS (µg/mL).
    c50r : float
        Concentration at half effect for remifentanil effect on BIS (ng/mL).
    gamma : float
        slope coefficient for the BIS  model.
    beta : float
        interaction coefficient for the BIS model.
    E0 : float
        initial BIS.
    Emax : float
        max effect of the drugs on BIS.
    hill_param : list
        Parameter of the Hill model (Propo Remi interaction)
        list [c50p_BIS, c50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]
    c50p_init : float
        Initial value of c50p, used for blood loss modelling.

    References
    ----------
    .. [Bouillon2004]  T. W. Bouillon et al., “Pharmacodynamic Interaction between Propofol and Remifentanil
            Regarding Hypnosis, Tolerance of Laryngoscopy, Bispectral Index, and Electroencephalographic
            Approximate Entropy,” Anesthesiology, vol. 100, no. 6, pp. 1353–1372, Jun. 2004,
            doi: 10.1097/00000542-200406000-00006.
    .. [Aubouin2023]  A. Aubouin et al., “Comparison of Multiple Kalman Filter and Moving Horizon
            Estimator for the Anesthesia Process” draft 2023.

    """

    def __init__(self, hill_model: str = 'Bouillon', hill_param: list = None,
                 random: bool = False):
        """
        Init the class.

        Returns
        -------
        None.

        """
        if hill_param is not None:  # Parameter given as an input
            self.c50p = hill_param[0]
            self.c50r = hill_param[1]
            self.gamma = hill_param[2]
            self.beta = hill_param[3]
            self.E0 = hill_param[4]
            self.Emax = hill_param[5]

        elif hill_model == 'Bouillon':
            # See [1] T. W. Bouillon et al., “Pharmacodynamic Interaction between Propofol and Remifentanil
            # Regarding Hypnosis, Tolerance of Laryngoscopy, Bispectral Index, and Electroencephalographic
            # Approximate Entropy,” Anesthesiology, vol. 100, no. 6, pp. 1353–1372, Jun. 2004,
            # doi: 10.1097/00000542-200406000-00006.

            self.c50p = 4.47
            self.c50r = 19.3
            self.gamma = 1.43
            self.beta = 0
            self.E0 = 97.4
            self.Emax = self.E0

            # coefficient of variation
            cv_c50p = 0.182
            cv_c50r = 0.888
            cv_gamma = 0.304
            cv_beta = 0
            cv_E0 = 0
            cv_Emax = 0

        if random and hill_param is None:
            # estimation of log normal standard deviation
            w_c50p = np.sqrt(np.log(1+cv_c50p**2))
            w_c50r = np.sqrt(np.log(1+cv_c50r**2))
            w_gamma = np.sqrt(np.log(1+cv_gamma**2))
            w_beta = np.sqrt(np.log(1+cv_beta**2))
            w_E0 = np.sqrt(np.log(1+cv_E0**2))
            w_Emax = np.sqrt(np.log(1+cv_Emax**2))

        if random and hill_param is None:
            self.c50p *= np.exp(np.random.normal(scale=w_c50p))
            self.c50r *= np.exp(np.random.normal(scale=w_c50r))
            self.beta *= np.exp(np.random.normal(scale=w_beta))
            self.gamma *= np.exp(np.random.normal(scale=w_gamma))
            self.E0 *= min(100, np.exp(np.random.normal(scale=w_E0)))
            self.Emax *= np.exp(np.random.normal(scale=w_Emax))

        self.hill_param = [self.c50p, self.c50r, self.gamma, self.beta, self.E0, self.Emax]
        self.c50p_init = self.c50p  # for blood loss modelling

    def compute_bis(self, c_es_propo: float, c_es_remi: float) -> float:
        """Compute BIS function from Propofol and Remifentanil effect site concentration.

        Parameters
        ----------
        cep : float
            Propofol effect site concentration µg/mL.
        cer : float
            Remifentanil effect site concentration ng/mL

        Returns
        -------
        BIS : float
            Bis value.

        """
        up = c_es_propo / self.c50p
        ur = c_es_remi / self.c50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - self.beta * (Phi - Phi**2)
        interaction = (up + ur)/U_50
        bis = self.E0 - self.Emax * interaction ** self.gamma / (1 + interaction ** self.gamma)

        return bis

    def update_param_blood_loss(self, v_ratio: float):
        """Update PK coefficient to mimic a blood loss.

        Update the c50p parameters thanks to the blood volume ratio. The values are estimated from [Johnson2003]_.

        Parameters
        ----------
        v_loss : float
            blood volume as a fraction of init volume, 1 mean no loss, 0 mean 100% loss.

        Returns
        -------
        None.

        References
        ----------
        .. [Johnson2003]  K. B. Johnson et al., “The Influence of Hemorrhagic Shock on Propofol: A Pharmacokinetic
                and Pharmacodynamic Analysis,” Anesthesiology, vol. 99, no. 2, pp. 409–420, Aug. 2003,
                doi: 10.1097/00000542-200308000-00023.

        """
        self.c50p = self.c50p_init - 3/0.5*(1-v_ratio)

    def inverse_hill(self, BIS: float, c_es_remi: float = 0) -> float:
        """Compute Propofol effect site concentration from BIS and Remifentanil effect site concentration.

        Parameters
        ----------
        BIS : float
            BIS value.
        cer : float, optional
            Effect site Remifentanil concentration (ng/mL). The default is 0.

        Returns
        -------
        cep : float
            Effect site Propofol concentration (µg/mL).

        """
        temp = (max(0, self.E0-BIS)/(self.Emax-self.E0+BIS))**(1/self.gamma)
        Yr = c_es_remi / self.c50r
        b = 3*Yr - temp
        c = 3*Yr**2 - (2 - self.beta) * Yr * temp
        d = Yr**3 - Yr**2*temp

        p = np.poly1d([1, b, c, d])

        real_root = 0
        try:
            for el in np.roots(p):
                if np.real(el) == el and np.real(el) > 0:
                    real_root = np.real(el)
                    break
            cep = real_root*self.c50p
        except Exception as e:
            print(f'bug: {e}')

        return cep

    def plot_surface(self):
        """Plot the 3D-Hill surface of the BIS related to Propofol and Remifentanil effect site concentration."""
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


class TOL_model():
    r"""Hierarchical model to link druf effect site concentration to Tolerance of Laringoscopy.

    The equation are:


    .. math:: postopioid = preopioid * \left(1 - \frac{C_{r,es}^{\gamma_r}}{C_{r,es}^{\gamma_r} + (C_{r,50} preopioid)^{\gamma_r}}\right)
    .. math:: TOL = \frac{C_{p,es}^{\gamma_p}}{C_{p,es}^{\gamma_p} + (C_{p,50} postopioid)^{\gamma_p}}

    Parameters
    ----------
    model : str, optional
        Only 'Bouillon' is available. Ignored if model_param is specified. The default is 'Bouillon'.
    model_param : list, optional
        Model parameters, model_param = [c50p, c50p, gammaP, gammaR, Preopioid intensity].
        The default is None.
    random : bool, optional
        Add uncertainties in the parameters. Ignored if model_param is specified. The default is False.

    Attributes
    ----------
    c50p : float
        Concentration at half effect for propofol effect on BIS (µg/mL).
    c50r : float
        Concentration at half effect for remifentanil effect on BIS (ng/mL).
    gamma_p : float
        Slope of the Hill function for propofol effect on TOL.
    gamma_r : float
        Slope of the Hill function for remifentanil effect on TOL.
    pre_intensity : float
        Preopioid intensity.

    """

    def __init__(self, model: str = 'Bouillon', model_param: list = None, random: bool = False):
        """
        Init the class.

        Returns
        -------
        None.

        """
        if model == "Bouillon":
            # See [Bouillon2004] T. W. Bouillon et al., “Pharmacodynamic Interaction between Propofol and Remifentanil
            # Regarding Hypnosis, Tolerance of Laryngoscopy, Bispectral Index, and Electroencephalographic
            # Approximate Entropy,” Anesthesiology, vol. 100, no. 6, pp. 1353–1372, Jun. 2004,
            # doi: 10.1097/00000542-200406000-00006.
            self.c50p = 8.04
            self.c50r = 1.07
            self.gamma_r = 0.97
            self.gamma_p = 5.1
            self.pre_intensity = 1.05  # Here we choose to use the value from laringoscopy

            cv_c50p = 0
            cv_c50r = 0.26
            cv_gamma_p = 0.9
            cv_gamma_r = 0.23
            cv_pre_intensity = 0
            w_c50p = np.sqrt(np.log(1+cv_c50p**2))
            w_c50r = np.sqrt(np.log(1+cv_c50r**2))
            w_gamma_p = np.sqrt(np.log(1+cv_gamma_p**2))
            w_gamma_r = np.sqrt(np.log(1+cv_gamma_r**2))
            w_pre_intensity = np.sqrt(np.log(1+cv_pre_intensity**2))

        if random and model_param is None:
            self.c50p *= np.exp(np.random.normal(scale=w_c50p))
            self.c50r *= np.exp(np.random.normal(scale=w_c50r))
            self.gamma_r *= np.exp(np.random.normal(scale=w_gamma_p))
            self.gamma_p *= np.exp(np.random.normal(scale=w_gamma_r))
            self.pre_intensity *= np.exp(np.random.normal(scale=w_pre_intensity))

    def compute_tol(self, c_es_propo: float, c_es_remi: float) -> float:
        """Return TOL from Propofol and Remifentanil effect site concentration.

        Compute the output of the Hirarchical model to predict TOL
        from Propofol and Remifentanil effect site concentration.
        TOL = 1 mean very relaxed and will tolerate laryngoscopie while TOL = 0 mean fully awake and will not tolerate.

        Parameters
        ----------
        cep : float
            Propofol effect site concentration µg/mL.
        cer : float
            Remifentanil effect site concentration ng/mL

        Returns
        -------
        TOL : float
            TOL value.

        """
        post_opioid = self.pre_intensity * (1 - fsig(c_es_remi, self.c50r*self.pre_intensity, self.gamma_r))
        tol = fsig(c_es_propo, self.c50p*post_opioid, self.gamma_p)
        return tol

    def plot_surface(self):
        """Plot the 3D-Hill surface of the BIS related to Propofol and Remifentanil effect site concentration."""
        cer = np.linspace(0, 20, 50)
        cep = np.linspace(0, 8, 50)
        cer, cep = np.meshgrid(cer, cep)
        effect = self.compute_tol(cep, cer)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(cer, cep, effect, cmap=cm.jet, linewidth=0.1)
        ax.set_xlabel('Remifentanil')
        ax.set_ylabel('Propofol')
        ax.set_zlabel('TOL')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        ax.view_init(12, -72)
        plt.show()


class Hemo_PD_model():
    """Modelize the effect of Propofol, Remifentanil, Norepinephrine on Mean Arterial Pressure and Cardiac Output.

    Use the addition of sigmoid curve to model the effect of each drugs on MAP and CO.
    The following articles are used to define the parameters of the model:

    - Norepinephrine to MAP: [Beloeil2005]_
    - Noepinephrine to CO: [Monnet2011]_
    - Propofol to MAP: [Jeleazcov2011]_
    - Propofol to CO: [Fairfield1991]_
    - Remifentanil to MAP: [Standing2010]_
    - Remifentanil to CO: [Chanavaz2005]_

    Parameters
    ----------
    nore_param : list, optional
        List of hill curve parameters for Norepinephrine action 
        [Emax_map, c50_map, gamma_map, Emax_co, c50_co, gamma_co].
        The default is None.
    propo_param : list, optional
        List of hill curve parameters for Propofol action 
        [emax_SAP, emax_DAP, c50_map_1, c50_map_2, gamma_map_1, gamma_map_2, Emax_co, c50_co, gamma_co].
        The default is None.
    remi_param : list, optional
        List of hill curve parameters for Relifentanil action 
        [Emax_map, c50_map, gamma_map, Emax_co, c50_co, gamma_co].
        The default is None.
    random : bool, optional
        Add uncertainties in the parameters. The default is False.
    co_base: float, optional
        Baseline Cardiac output (L/min). The default is 6.5 L/min.
    map_base: float, optional
        Baseline mean arterial pressure (mmHg). The default is 90mmHg.

    Attributes
    ----------
    co_base : float
        Baseline cardiac output.
    map_base : float
        Baseline mean arterial pressure.
    emax_nore_map : float
        Maximal effect of Norepinephrine on MAP.
    c50_nore_map : float
        Concentration of Norepinephrine that produce half of the maximal effect on MAP.
    gamma_nore_map : float
        Slope of the sigmoid curve for Norepinephrine effect on MAP.
    emax_nore_co : float
        Maximal effect of Norepinephrine on CO.
    c50_nore_co : float
        Concentration of Norepinephrine that produce half of the maximal effect on CO.
    gamma_nore_co : float
        Slope of the sigmoid curve for Norepinephrine effect on CO.
    emax_propo_SAP : float
        Maximal effect of Propofol on SAP.
    emax_propo_DAP : float
        Maximal effect of Propofol on DAP.
    emax_propo_co : float
        Maximal effect of Propofol on CO.
    c50_propo_map_1 : float
        Concentration of Propofol that produce half of the maximal effect on MAP.
    c50_propo_map_2 : float
        Concentration of Propofol that produce half of the maximal effect on MAP.
    gamma_propo_map_1 : float
        Slope of the sigmoid curve for Propofol effect on MAP.
    gamma_propo_map_2 : float
        Slope of the sigmoid curve for Propofol effect on MAP.
    c50_propo_co : float
        Concentration of Propofol that produce half of the maximal effect on CO.
    gamma_propo_co : float
        Slope of the sigmoid curve for Propofol effect on CO.
    emax_remi_map : float
        Maximal effect of Remifentanil on MAP.
    emax_remi_co : float
        Maximal effect of Remifentanil on CO.
    c50_remi_map : float
        Concentration of Remifentanil that produce half of the maximal effect on MAP.
    gamma_remi_map : float
        Slope of the sigmoid curve for Remifentanil effect on MAP.
    c50_remi_co : float
        Concentration of Remifentanil that produce half of the maximal effect on CO.
    gamma_remi_co : float
        Slope of the sigmoid curve for Remifentanil effect on CO.
    map : float
        Mean arterial pressure.
    co : float
        Cardiac output.

    References
    ----------
    .. [Beloeil2005]  H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau, 
            “Norepinephrine kinetics and dynamics in septic shock and trauma patients,”
            BJA: British Journal of Anaesthesia, vol. 95, no. 6, pp. 782–788, Dec. 2005,
            doi: 10.1093/bja/aei261.
    .. [Monnet2011]  X. Monnet, J. Jabot, J. Maizel, C. Richard, and J.-L. Teboul,
            “Norepinephrine increases cardiac preload and reduces preload dependency assessed by passive leg
            raising in septic shock patients”
            Critical Care Medicine, vol. 39, no. 4, p. 689, Apr. 2011, doi: 10.1097/CCM.0b013e318206d2a3.
    .. [Jeleazcov2011]  C. Jeleazcov, M. Lavielle, J. Schüttler, and H. Ihmsen,
            “Pharmacodynamic response modelling of arterial blood pressure in adult
            volunteers during propofol anaesthesia,”
            BJA: British Journal of Anaesthesia,
            vol. 115, no. 2, pp. 213–226, Aug. 2015, doi: 10.1093/bja/aeu553.
    .. [Fairfield1991]  J. E. Fairfield, A. Dritsas, and R. J. Beale,
            “HAEMODYNAMIC EFFECTS OF PROPOFOL: INDUCTION WITH 2.5 MG KG-1,”
            British Journal of Anaesthesia, vol. 67, no. 5, pp. 618–620, Nov. 1991, doi: 10.1093/bja/67.5.618.
    .. [Standing2010]  J. F. Standing, G. B. Hammer, W. J. Sam, and D. R. Drover,
            “Pharmacokinetic–pharmacodynamic modeling of the hypotensive effect of
            remifentanil in infants undergoing cranioplasty,”
            Pediatric Anesthesia, vol. 20, no. 1, pp. 7–18, 2010, doi: 10.1111/j.1460-9592.2009.03174.x.
    .. [Chanavaz2005]  C. Chanavaz et al.,
            “Haemodynamic effects of remifentanil in children with and
            without intravenous atropine. An echocardiographic study,”
            BJA: British Journal of Anaesthesia, vol. 94, no. 1, pp. 74–79, Jan. 2005, doi: 10.1093/bja/aeh293.

    """

    def __init__(self, nore_param: list = None, propo_param: list = None,
                 remi_param: list = None, random: bool = False,
                 co_base: float = 6.5, map_base: float = 90):
        """
        Initialize the class.

        Returns
        -------
        None.

        """
        self.co_base = co_base
        self.map_base = map_base
        w_not_known = 0.4
        std_not_known = 1
        if nore_param is None:
            # see H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau, “Norepinephrine kinetics and dynamics
            # in septic shock and trauma patients,” BJA: British Journal of Anaesthesia,
            # vol. 95, no. 6, pp. 782–788, Dec. 2005, doi: 10.1093/bja/aei259.
            self.emax_nore_map = 98.7
            self.c50_nore_map = 70.4
            self.gamma_nore_map = 1.8
            w_emax_nore_map = 0
            w_c50_nore_map = 1.64
            w_gamma_nore_map = 0

            # see X. Monnet, J. Jabot, J. Maizel, C. Richard, and J.-L. Teboul,
            # “Norepinephrine increases cardiac preload and reduces preload dependency assessed by passive leg
            # raising in septic shock patients*,”
            # Critical Care Medicine, vol. 39, no. 4, p. 689, Apr. 2011, doi: 10.1097/CCM.0b013e318206d2a3.

            self.emax_nore_co = 0.3 * self.co_base
            self.c50_nore_co = 0.36
            self.gamma_nore_co = 2.3  # to have an increase of 11% for a change between 0.24 and 0.48 of concentration
            std_emax_nore_co = std_not_known
            w_c50_nore_co = w_not_known
            w_gamma_nore_co = w_not_known

        else:
            self.emax_nore_map = nore_param[0]
            self.c50_nore_map = nore_param[1]
            self.gamma_nore_ma = nore_param[2]
            self.emax_nore_co = nore_param[3]
            self.c50_nore_co = nore_param[4]
            self.gamma_nore_co = nore_param[5]

            # variability set to 0 if value are given
            w_emax_nore_map = 0
            w_c50_nore_map = 0
            w_gamma_nore_map = 0
            std_emax_nore_co = 0
            w_c50_nore_co = 0
            w_gamma_nore_co = 0

        if propo_param is None:
            # see C. Jeleazcov, M. Lavielle, J. Schüttler, and H. Ihmsen,
            # “Pharmacodynamic response modelling of arterial blood pressure in adult
            # volunteers during propofol anaesthesia,”
            # BJA: British Journal of Anaesthesia, vol. 115, no. 2, pp. 213–226, Aug. 2015, doi: 10.1093/bja/aeu553.

            self.emax_propo_SAP = 54.8
            self.emax_propo_DAP = 18.1
            self.c50_propo_map_1 = 1.96
            self.gamma_propo_map_1 = 4.77
            self.c50_propo_map_2 = 2.20
            self.gamma_propo_map_2 = 8.49
            w_emax_propo_SAP = 0.0871
            w_emax_propo_DAP = 0.207
            w_c50_propo_map_1 = 0.165
            w_c50_propo_map_2 = 0.148
            w_gamma_propo_map_1 = np.sqrt(np.log(1+5.59**2))
            w_gamma_propo_map_2 = np.sqrt(np.log(1+6.33**2))

            # see J. E. Fairfield, A. Dritsas, and R. J. Beale,
            # “HAEMODYNAMIC EFFECTS OF PROPOFOL: INDUCTION WITH 2.5 MG KG−1,”
            # British Journal of Anaesthesia, vol. 67, no. 5, pp. 618–620, Nov. 1991, doi: 10.1093/bja/67.5.618.

            self.emax_propo_co = -2
            self.c50_propo_co = 2.6
            self.gamma_propo_co = 2
            std_emax_propo_co = std_not_known
            w_c50_propo_co = w_not_known
            w_gamma_propo_co = w_not_known
        else:
            self.emax_propo_SAP = propo_param[0]
            self.emax_propo_DAP = propo_param[1]
            self.c50_propo_map_1 = propo_param[2]
            self.gamma_propo_map_1 = propo_param[3]
            self.c50_propo_map_2 = propo_param[4]
            self.gamma_propo_map_2 = propo_param[5]
            self.emax_propo_co = propo_param[6]
            self.c50_propo_co = propo_param[7]
            self.gamma_propo_co = propo_param[8]

            # variability set to 0 if value are given
            w_emax_propo_SAP = 0
            w_emax_propo_DAP = 0
            w_c50_propo_map_1 = 0
            w_c50_propo_map_2 = 0
            w_gamma_propo_map_1 = 0
            w_gamma_propo_map_2 = 0
            std_emax_propo_co = 0
            w_c50_propo_co = 0
            w_gamma_propo_co = 0

        if remi_param is None:
            # see J. F. Standing, G. B. Hammer, W. J. Sam, and D. R. Drover,
            # “Pharmacokinetic–pharmacodynamic modeling of the hypotensive effect of
            # remifentanil in infants undergoing cranioplasty,”
            # Pediatric Anesthesia, vol. 20, no. 1, pp. 7–18, 2010, doi: 10.1111/j.1460-9592.2009.03174.x.

            self.emax_remi_map = -map_base
            self.c50_remi_map = 17.1
            self.gamma_remi_map = 4.56
            w_emax_remi_map = 0
            w_c50_remi_map = 0.09
            w_gamma_remi_map = 0

            # see C. Chanavaz et al.,
            # “Haemodynamic effects of remifentanil in children with and
            # without intravenous atropine. An echocardiographic study,”
            # BJA: British Journal of Anaesthesia, vol. 94, no. 1, pp. 74–79, Jan. 2005, doi: 10.1093/bja/aeh293.

            self.emax_remi_co = -1.5
            self.c50_remi_co = 5
            self.gamma_remi_co = 2
            w_emax_remi_co = w_not_known
            w_c50_remi_co = w_not_known
            w_gamma_remi_co = w_not_known
        else:
            self.emax_remi_map = remi_param[0]
            self.c50_remi_map = remi_param[1]
            self.gamma_remi_ma = remi_param[2]
            self.emax_remi_co = remi_param[3]
            self.c50_remi_co = remi_param[4]
            self.gamma_remi_co = remi_param[5]

            # variability set to 0 if value are given
            w_emax_remi_map = 0
            w_c50_remi_map = 0
            w_gamma_remi_map = 0
            w_emax_remi_co = 0
            w_c50_remi_co = 0
            w_gamma_remi_co = 0

        if random:
            # Norepinephrine
            self.emax_nore_map *= np.exp(np.random.normal(scale=w_emax_nore_map))
            self.c50_nore_map *= np.exp(np.random.normal(scale=w_c50_nore_map))
            self.gamma_nore_map *= np.exp(np.random.normal(scale=w_gamma_nore_map))

            self.emax_nore_co += np.random.normal(scale=std_emax_nore_co)
            self.c50_nore_co *= np.exp(np.random.normal(scale=w_c50_nore_co))
            self.gamma_nore_co *= np.exp(np.random.normal(scale=w_gamma_nore_co))

            # Propofol
            self.emax_propo_SAP *= np.exp(np.random.normal(scale=w_emax_propo_SAP))
            self.emax_propo_DAP *= np.exp(np.random.normal(scale=w_emax_propo_DAP))
            self.c50_propo_map_1 *= np.exp(np.random.normal(scale=w_c50_propo_map_1))
            self.gamma_propo_map_1 *= min(3, np.exp(np.random.normal(scale=w_gamma_propo_map_1)))
            self.c50_propo_map_2 *= np.exp(np.random.normal(scale=w_c50_propo_map_2))
            self.gamma_propo_map_2 *= min(3, np.exp(np.random.normal(scale=w_gamma_propo_map_2)))

            self.emax_propo_co += np.random.normal(scale=std_emax_propo_co)
            self.c50_propo_co *= np.exp(np.random.normal(scale=w_c50_propo_co))
            self.gamma_propo_co *= np.exp(np.random.normal(scale=w_gamma_propo_co))

            # Remifentanil
            self.emax_remi_map *= np.exp(np.random.normal(scale=w_emax_remi_map))
            self.c50_remi_map *= np.exp(np.random.normal(scale=w_c50_remi_map))
            self.gamma_remi_map *= np.exp(np.random.normal(scale=w_gamma_remi_map))

            self.emax_remi_co *= np.exp(np.random.normal(scale=w_emax_remi_co))
            self.c50_remi_co *= np.exp(np.random.normal(scale=w_c50_remi_co))
            self.gamma_remi_co *= np.exp(np.random.normal(scale=w_gamma_remi_co))

    def compute_hemo(self, c_es_propo: list, c_es_remi: float, c_es_nore: float) -> tuple[float, float]:
        """
        Compute current MAP and CO using addition of hill curv, one for each drugs.

        Parameters
        ----------
        c_es_propo : list
            Propofol concentration on both hemodynamic effect site concentration µg/mL.
        c_es_remi : float
            Remifentanil hemodynamic effect site concentration µg/mL.
        c_es_nore : float
            Norepinephrine hemodynamic effect site concentration µg/mL.

        Returns
        -------
        map : float
            Mean arterial pressure (mmHg), without blood loss.
        co : float
            Cardiac output (L/min), without blood loss.

        """
        map_nore = self.emax_nore_map * fsig(c_es_nore, self.c50_nore_map, self.gamma_nore_map)
        u_propo = ((c_es_propo[0]/self.c50_propo_map_1)**self.gamma_propo_map_1 +
                   (c_es_propo[1]/self.c50_propo_map_2)**self.gamma_propo_map_2)
        map_propo = - (self.emax_propo_DAP + (self.emax_propo_SAP + self.emax_propo_DAP) / 3) * u_propo/(1+u_propo)
        map_remi = self.emax_remi_map * fsig(c_es_remi, self.c50_remi_map, self.gamma_remi_map)

        self.map = self.map_base + map_nore + map_propo + map_remi

        co_nore = self.emax_nore_co * fsig(c_es_nore, self.c50_nore_co, self.gamma_nore_co)
        co_propo = self.emax_propo_co * fsig((c_es_propo[0] + c_es_propo[1])/2, self.c50_propo_co, self.gamma_propo_co)
        co_remi = self.emax_remi_co * fsig(c_es_remi, self.c50_remi_co, self.gamma_remi_co)

        self.co = self.co_base + co_nore + co_propo + co_remi

        return self.map, self.co
