#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:47:11 2022

@author: aubouinb
"""
import numpy as np


def compute_control_metrics(Bis: list, Ts: float = 1, phase: str = 'maintenance',
                            start_step: float = 600, end_step: float = 1200):
    """Compute metrics for closed loop anesthesia.

    This function compute the control metrics initially proposed in "C. M. Ionescu, R. D. Keyser, B. C. Torrico,
    T. D. Smet, M. M. Struys, and J. E. Normey-Rico, “Robust Predictive Control Strategy Applied for Propofol Dosing
    Using BIS as a Controlled Variable During Anesthesia,” IEEE Transactions on Biomedical Engineering, vol. 55, no.
    9, pp. 2161–2170, Sep. 2008, doi: 10.1109/TBME.2008.923142."


    Parameters
    ----------
    Bis : liste
        Llist of BIS value over time.
    Ts : float, optional
        Sampling time in second. The default is 1.
    phase : str, optional
        Control phase, can be "maintenance", 'induction" or "total". The default is 'maintenance'.
    start_step: float, optional
        Start time of the step disturbance, for maintenance and total phase. The default is 600s.
    end_step: float, optional
        End time of the step disturbance, for maintenance and total phase. The default is 1200s.

    Returns
    -------
    for "induciton" phase:
    TT : float
        Observed time-to-target (in seconds) required for reaching first time the target interval of [55,45] BIS values
    BIS_NADIR: float
        The lowest observed BIS value during induction phase
    ST10: float
        Settling time on the reference BIS value, defined within ± 5BIS(i.e., between 45 and 55 BIS)
        and stay within this BIS range
    ST20: float
        Settling time on the reference BIS value, defined within ± 10BIS(i.e., between 40 and 60 BIS)
        and stay within this BIS range
    US: float
        Undershoot, defined as the BIS value that exceeds the limit of the defined BIS interval,
        namely, the 45 BIS value.

    For "maintenance" phase:
    TTp : float
        Time to target after the positive step disturbance.
    BIS_NADIRp: float
        Minimum BIS vamue after the positive step disturbance.
    TTpn: float
         Time to target after the negative step disturbance.
     BIS_NADIRn: float
         Maximum BIS vamue after the negative step disturbance.

    For total phase: both indction and maintenance phase.
    """
    if phase == 'induction':
        BIS_NADIR = min(Bis)
        US = max(0, 45 - BIS_NADIR)
        TT, ST10, ST20 = np.nan, np.nan, np.nan
        for j in range(len(Bis)):
            if Bis[j] < 55:
                if np.isnan(TT):
                    TT = j*Ts/60
            if Bis[j] < 55 and Bis[j] > 45:
                if np.isnan(ST10):
                    ST10 = j*Ts/60
            else:
                ST10 = np.nan

            if Bis[j] < 60 and Bis[j] > 40:
                if np.isnan(ST20):
                    ST20 = j*Ts/60
            else:
                ST20 = np.nan
        return TT, BIS_NADIR, ST10, ST20, US

    elif phase == 'maintenance':
        BIS_NADIRp = min(Bis[int(start_step/Ts):int(end_step/Ts)])
        BIS_NADIRn = max(Bis[int(end_step/Ts):])
        TTp, TTn = np.nan, np.nan
        for j in range(int(start_step/Ts), int(end_step/Ts)):
            if Bis[j] < 55:
                TTp = (j*Ts-60)/60
                break

        for j in range(int(end_step/Ts), len(Bis)):
            if Bis[j] > 45:
                TTn = (j*Ts-5*60)/60
                break

        return TTp, BIS_NADIRp, TTn, BIS_NADIRn

    elif phase == 'total':
        BIS_NADIR = min(Bis)
        US = max(0, 45 - BIS_NADIR)
        TT, ST10, ST20 = np.nan, np.nan, np.nan
        for j in range(int(10*60/Ts)):
            if Bis[j] < 55:
                if np.isnan(TT):
                    TT = j*Ts/60
            if Bis[j] < 55 and Bis[j] > 45:
                if np.isnan(ST10):
                    ST10 = j*Ts/60
            else:
                ST10 = np.nan

            if Bis[j] < 60 and Bis[j] > 40:
                if np.isnan(ST20):
                    ST20 = j*Ts/60
            else:
                ST20 = np.nan

        BIS_NADIRp = min(Bis[int(start_step/Ts):int(end_step/Ts)])
        BIS_NADIRn = max(Bis[int(end_step/Ts):])
        TTp, TTn = np.nan, np.nan
        for j in range(int(start_step/Ts), int(end_step/Ts)):
            if Bis[j] < 55:
                TTp = (j*Ts-60)/60
                break

        for j in range(int(end_step/Ts), len(Bis)):
            if Bis[j] > 45:
                TTn = (j*Ts-5*60)/60
                break

        return TT, BIS_NADIR, ST10, ST20, US, TTp, BIS_NADIRp, TTn, BIS_NADIRn
