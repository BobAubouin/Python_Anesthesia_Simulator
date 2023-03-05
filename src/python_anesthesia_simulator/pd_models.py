"""Includes class for different PD models.

@author: Aubouin--Pairault Bob 2023, bob.aubouin@tutanota.com
"""

# Third party imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def fsig(x, C50, gam): return x**gam/(C50**gam + x**gam)  # quick definition of sigmoidal function


class BIS_model:
    """efect site +Hill curv model to link Propofol and Remifentanil blood concentration to BIS."""

    def __init__(self, hill_model: str = 'Bouillon', hill_param: list = None,
                 random: bool = False):
        """
        Init the class.

        Parameters
        ----------
        hill_model : str, optional
            Only 'Bouillon' is available. Ignored if  is specified.
        hill_param : list, optional
            Parameter of the Hill model (Propo Remi interaction)
            list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]:
                - C50p_BIS : Concentration at half effect for propofol effect on BIS (µg/mL)
                - C50r_BIS : Concentration at half effect for remifentanil effect on BIS (ng/mL)
                - gamma_BIS : slope coefficient for the BIS  model,
                - beta_BIS : interaction coefficient for the BIS model,
                - E0_BIS : initial BIS,
                - Emax_BIS : max effect of the drugs on BIS.
            The default is None.
        random : bool, optional
            Add uncertainties in the parameters. Ignored if Hill_cruv is specified. The default is False.
        ts : float, optional
            Sampling time, in s. The default is 1.

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

    def update_param_blood_loss(self, v_loss_ratio: float):
        """Update PK coefficient to mimic a blood loss.

        Update the c50p parameters thanks to the blood volume loss ratio.

        Parameters
        ----------
        v_loss : float
            loss volume as a fraction of total volume, 0 mean no loss, 1 mean 100% loss.

        Returns
        -------
        None.

        """
        # value estimated from K. B. Johnson et al., “The Influence of Hemorrhagic Shock on Propofol: A Pharmacokinetic
        # and Pharmacodynamic Analysis,” Anesthesiology, vol. 99, no. 2, pp. 409–420, Aug. 2003,
        # doi: 10.1097/00000542-200308000-00023.

        self.c50r = self.c50p_init - 3/0.5*v_loss_ratio

    def inverse_hill(self, BIS: float, cer: float = 0) -> float:
        """Compute Propofol effect site concentration from BIS and Remifentanil Effect site concentration.

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
        Yr = cer / self.c50r
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
        except:
            print('bug')

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
    """Hierarchical model to link druf effect site concentration to Tolerance of Laringoscopy."""

    def __init__(self, model: str = 'Bouillon', model_param: list = None, random: bool = False):
        """
        Init the class.

        Parameters
        ----------
        model : str, optional
            Only 'Bouillon' is available. Ignored if model_param is specified. The default is 'Bouillon'.
        model_param : list, optional
            Model parameters, model_param = [C50p, C50p, gammaP, gammaR, Preopioid intensity].
            The default is None.
        random : bool, optional
            Add uncertainties in the parameters. Ignored if model_param is specified. The default is False.

        Returns
        -------
        None.

        """
        if model == "Bouillon":
            # See [1] T. W. Bouillon et al., “Pharmacodynamic Interaction between Propofol and Remifentanil
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
        tol = 1 - fsig(c_es_propo, self.c50p*post_opioid, self.gamma_p)
        return tol

    def plot_surface(self):
        """Plot the 3D-Hill surface of the BIS related to Propofol and Remifentanil effect site concentration."""
        cer = np.linspace(0, 4, 50)
        cep = np.linspace(0, 6, 50)
        cer, cep = np.meshgrid(cer, cep)
        effect = 100 - self.compute_tol(cep, cer)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(cer, cep, effect, cmap=cm.jet, linewidth=0.1)
        ax.set_xlabel('Remifentanil')
        ax.set_ylabel('Propofol')
        ax.set_zlabel('Effect')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        ax.view_init(12, -72)
        plt.show()

# class Hemo_PD_model():
#     """Modelize the effect of Propofol, Remifentanil, Norepinephrine on Meanrterial pressure and Cardiac Output."""

#     def __init__(self, )
