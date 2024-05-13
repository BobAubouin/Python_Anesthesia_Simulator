import numpy as np
import pandas as pd


def compute_control_metrics(time: list, bis: list, phase: str = 'maintenance',
                            start_step: float = 600, end_step: float = 1200):
    """Compute metrics for closed loop anesthesia.

    This function compute the control metrics initially proposed in [Ionescu2008]_.


    Parameters
    ----------
    time : list
        List of time value (s).
    bis : list
        List of BIS value over time.
    phase : str, optional
        Control phase, can be "maintenance", 'induction" or "total". The default is 'maintenance'.
    start_step: float, optional
        Start time of the step disturbance, for maintenance and total phase. The default is 600s.
    end_step: float, optional
        End time of the step disturbance, for maintenance and total phase. The default is 1200s.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the computed metrics:
        TT : float
            Observed time-to-target (in minute) required for reaching first time the target interval of [55,45] BIS values.
        BIS_NADIR: float
            for "induction" or "total" phase. The lowest observed BIS value during induction phase.
        ST10: float
            for "induction" or "total" phase. Settling time (in minute) on the reference BIS value,
            defined within ± 5BIS(i.e., between 45 and 55 BIS)and stay within this BIS range.
        ST20: float
            for "induction" or "total" phase. Settling time (in minute) on the reference BIS value,
            defined within ± 10BIS(i.e., between 40 and 60 BIS) and stay within this BIS range.
        US: float
            for "induction" or "total" phase. Undershoot, defined as the BIS value that exceeds the
            limit of the defined BIS interval, namely, the 45 BIS value.
        TTp : float
            Time to target (in minute) after the positive step disturbance.
        BIS_NADIRp: float
            for "maintenance" or "total" phase. Minimum BIS vamue after the positive step disturbance.
        TTpn: float
            for "maintenance" or "total" phase. Time to target (in minute) after the negative step disturbance.
        BIS_NADIRn: float
            for "maintenance" or "total" phase. Maximum BIS vamue after the negative step disturbance.

    References
    ----------
    .. [Ionescu2008]  C. M. Ionescu, R. D. Keyser, B. C. Torrico, T. D. Smet, M. M. Struys, and J. E. Normey-Rico,
            “Robust Predictive Control Strategy Applied for Propofol Dosing Using BIS as a Controlled
            Variable During Anesthesia,” IEEE Transactions on Biomedical Engineering, vol. 55, no.
            9, pp. 2161–2170, Sep. 2008, doi: 10.1109/TBME.2008.923142.

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
        df = pd.DataFrame([{'TT': TT,
                            'BIS_NADIR': BIS_NADIR,
                            'ST10': ST10,
                            'ST20': ST20,
                            'US': US}])
        return df

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
        df = pd.DataFrame([{'TTp': TTp,
                            'BIS_NADIRp': BIS_NADIRp,
                            'TTn': TTn,
                            'BIS_NADIRn': BIS_NADIRn}])
        return df

    elif phase == 'total':
        # consider induction as the first 10 minutes
        index_10 = np.where(np.array(time) == start_step)[0][0]-1
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
        df = pd.DataFrame([{'TT': TT,
                            'BIS_NADIR': BIS_NADIR,
                            'ST10': ST10,
                            'ST20': ST20,
                            'US': US,
                            'TTp': TTp,
                            'BIS_NADIRp': BIS_NADIRp,
                            'TTn': TTn,
                            'BIS_NADIRn': BIS_NADIRn}])
        return df


def intergal_absolut_error(time: list, bis: list, bis_target: float = 50):
    """Compute the integral of the absolute error.

    This function compute the integral of the absolute error between the BIS value and the target value.

    Parameters
    ----------
    time : list
        List of time value (s).
    bis : list
        List of BIS value over time.
    bis_target : float, optional
        Target BIS value. The default is 50.

    Returns
    -------
    IAE : float
        Integral of the absolute error.

    """
    iae = np.trapz(np.abs(np.array(bis)-bis_target), time)
    return iae


def new_metrics_induction(time: list, bis: list):
    """Compute new metrics for induction of closed loop anesthesia.

    This function compute new metrics for closed loop anesthesia.

    Parameters
    ----------
    time : list
        List of time value (s).
    bis : list
        List of BIS value over time.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the computed metrics:
        IAE : float
            Integral of the absolute error.
        Sleep_Time : float
            Time to reach BIS < 60 and stay below 60 (minutes).
        Low BIS time : float
            Time passed with BIS < 40 (seconds).
        Lowest BIS : float
            Lowest BIS value.
        Settling time : float
            Time to reach BIS < 60 and stay within [40, 60] (minutes).
    """
    results = {}
    # Integral of the absolute error
    iae = intergal_absolut_error(time, bis)
    results['IAE'] = iae
    # Sleep time
    sleep_time = np.nan
    for j in range(len(bis)-1, -1, -1):
        if bis[j] > 60:
            if j == len(bis)-1:
                sleep_time = time[j]/60
            else:
                sleep_time = time[j+1]/60
            break
    results['Sleep_Time'] = sleep_time
    # Low BIS time
    ts = time[1] - time[0]
    low_bis_index = np.where(np.array(bis) < 40)[0]
    low_bis_time = len(low_bis_index)*ts
    results['Low BIS time'] = low_bis_time
    # Lowest BIS
    lowest_bis = min(bis)
    results['Lowest BIS'] = lowest_bis
    # Settling time
    settling_time = np.nan
    for j in range(len(bis)-1, -1, -1):
        if bis[j] > 60 or bis[j] < 40:
            if j == len(bis)-1:
                settling_time = time[j]/60
            else:
                settling_time = time[j+1]/60
            break
    results['Settling time'] = settling_time
    df = pd.DataFrame([results])
    return df


def new_metrics_maintenance(time: list, bis: list):
    """Compute new metrics for maintenance of closed loop anesthesia.

    Parameters
    ----------
    time : list
        List of time value (s).
    bis : list
        List of BIS value over time.


    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the computed metrics:
        IAE : float
            Integral of the absolute error.
        Time out of range : float
            Time passed with BIS out of [40, 60] (seconds).
        Lowest BIS : float
            Lowest BIS value.
        Highest BIS : float
            Highest BIS value.
    """
    IAE = intergal_absolut_error(time, bis)
    results = {}
    results['IAE'] = IAE
    # Time out of range
    ts = time[1] - time[0]
    out_range_index = np.where(np.array(bis) < 40)[0]
    out_range_time = len(out_range_index)*ts
    out_range_index = np.where(np.array(bis) > 60)[0]
    out_range_time += len(out_range_index)*ts
    results['Time out of range'] = out_range_time
    # Lowest BIS
    lowest_bis = min(bis)
    results['Lowest BIS'] = lowest_bis
    # Highest BIS
    highest_bis = max(bis)
    results['Highest BIS'] = highest_bis
    df = pd.DataFrame([results])
    return df
