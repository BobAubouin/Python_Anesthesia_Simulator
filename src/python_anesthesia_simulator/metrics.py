"""
Created on Mon Oct 10 09:47:11 2022

@author: aubouinb
"""
import numpy as np


def compute_control_metrics(time: list, bis: list, phase: str = 'maintenance',
                            start_step: float = 600, end_step: float = 1200):
    """Compute metrics for closed loop anesthesia.

    This function compute the control metrics initially proposed in "C. M. Ionescu, R. D. Keyser, B. C. Torrico,
    T. D. Smet, M. M. Struys, and J. E. Normey-Rico, “Robust Predictive Control Strategy Applied for Propofol Dosing
    Using BIS as a Controlled Variable During Anesthesia,” IEEE Transactions on Biomedical Engineering, vol. 55, no.
    9, pp. 2161–2170, Sep. 2008, doi: 10.1109/TBME.2008.923142."


    Parameters
    ----------
    time : list
        List of time value.
    bis : list
        List of BIS value over time.
    ts : float, optional
        Sampling time in second. The default is 1.
    phase : str, optional
        Control phase, can be "maintenance", 'induction" or "total". The default is 'maintenance'.
    start_step: float, optional
        Start time of the step disturbance, for maintenance and total phase. The default is 600s.
    end_step: float, optional
        End time of the step disturbance, for maintenance and total phase. The default is 1200s.

    Returns
    -------
    for "induction" phase:
    TT : float
        Observed time-to-target (in minute) required for reaching first time the target interval of [55,45] BIS values
    BIS_NADIR: float
        The lowest observed BIS value during induction phase
    ST10: float
        Settling time (in minute) on the reference BIS value, defined within ± 5BIS(i.e., between 45 and 55 BIS)
        and stay within this BIS range
    ST20: float
        Settling time (in minute) on the reference BIS value, defined within ± 10BIS(i.e., between 40 and 60 BIS)
        and stay within this BIS range
    US: float
        Undershoot, defined as the BIS value that exceeds the limit of the defined BIS interval,
        namely, the 45 BIS value.

    For "maintenance" phase:
    TTp : float
        Time to target (in minute) after the positive step disturbance.
    BIS_NADIRp: float
        Minimum BIS vamue after the positive step disturbance.
    TTpn: float
         Time to target (in minute) after the negative step disturbance.
     BIS_NADIRn: float
         Maximum BIS vamue after the negative step disturbance.

    For total phase: both induction and maintenance phase.
    """
    if phase == 'induction':
        BIS_NADIR = min(bis)
        US = max(0, 45 - BIS_NADIR)
        TT, ST10, ST20 = np.nan, np.nan, np.nan
        for j in range(len(bis)):
            if bis[j] < 55:
                if np.isnan(TT):
                    TT = time[j]/60
            if bis[j] < 55 and bis[j] > 45:
                if np.isnan(ST10):
                    ST10 = time[j]/60
            else:
                ST10 = np.nan

            if bis[j] < 60 and bis[j] > 40:
                if np.isnan(ST20):
                    ST20 = time[j]/60
            else:
                ST20 = np.nan
        return TT, BIS_NADIR, ST10, ST20, US

    elif phase == 'maintenance':
        # find start step index
        index_start = np.where(np.array(time) == start_step)[0][0]
        index_end = np.where(np.array(time) == end_step)[0][0]

        BIS_NADIRp = min(bis[index_start:index_end])
        BIS_NADIRn = max(bis[index_end:])
        TTp, TTn = np.nan, np.nan
        for j in range(index_start, index_end):
            if bis[j] < 55:
                TTp = (time[j]-start_step)/60
                break

        for j in range(index_end, len(bis)):
            if bis[j] > 45:
                TTn = (time[j]-end_step)/60
                break

        return TTp, BIS_NADIRp, TTn, BIS_NADIRn

    elif phase == 'total':
        # consider induction as the first 10 minutes
        index_10 = np.where(np.array(time) == 10*60)[0][0]
        bis_induction = bis[:index_10]
        BIS_NADIR = min(bis_induction)
        US = max(0, 45 - BIS_NADIR)
        TT, ST10, ST20 = np.nan, np.nan, np.nan
        for j in range(index_10):
            if bis_induction[j] < 55:
                if np.isnan(TT):
                    TT = time[j]/60
            if bis_induction[j] < 55 and bis_induction[j] > 45:
                if np.isnan(ST10):
                    ST10 = time[j]/60
            else:
                ST10 = np.nan

            if bis_induction[j] < 60 and bis_induction[j] > 40:
                if np.isnan(ST20):
                    ST20 = time[j]/60
            else:
                ST20 = np.nan
        # Maintenance phase
        # find start step index
        index_start = np.where(np.array(time) == start_step)[0][0] + 1
        index_end = np.where(np.array(time) == end_step)[0][0] + 1
        BIS_NADIRp = min(bis[index_start:index_end])
        BIS_NADIRn = max(bis[index_end:])
        TTp, TTn = np.nan, np.nan
        for j in range(index_start, index_end):
            if bis[j] < 55:
                TTp = (time[j]-start_step)/60
                break

        for j in range(index_end, len(bis)):
            if bis[j] > 45:
                TTn = (time[j]-time[index_end])/60
                break

        return TT, BIS_NADIR, ST10, ST20, US, TTp, BIS_NADIRp, TTn, BIS_NADIRn
