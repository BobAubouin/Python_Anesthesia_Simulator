#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:47:11 2022

@author: aubouinb
"""
import numpy as np
def compute_control_metrics(Bis: list, Te: float = 1, phase: str = 'maintenance', latex_output: bool =False):
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
    if phase=='induction':
        BIS_NADIR = min(Bis)
        US = max(0, 45 - BIS_NADIR)
        TT, ST10, ST20 = np.nan, np.nan, np.nan
        for j in range(len(Bis)):
            if Bis[j]<55:
                if np.isnan(TT):
                    TT = j*Te/60
            if Bis[j]<55 and Bis[j]>45:
                if np.isnan(ST10):
                    ST10 = j*Te/60
            else:
                ST10 = np.nan
                
            if Bis[j]<60 and Bis[j]>40:
                if np.isnan(ST20):
                    ST20 = j*Te/60
            else:
                ST20 = np.nan
        return TT, BIS_NADIR, ST10, ST20, US
    
    elif phase=='maintenance':
        BIS_NADIRp = min(Bis[:round(len(Bis))-1])
        BIS_NADIRn = max(Bis[round(len(Bis)):])
        TTp, TTn = np.nan, np.nan
        for j in range(int(60/Te)+1, int(5*60/Te)):
            if Bis[j]<55:
                TTp = (j*Te-60)/60
                break
            
        for j in range(int(5*60/Te)+1, len(Bis)):
            if Bis[j]<45:
                TTn = (j*Te-5*60)/60
                break   
        
        return TTp, BIS_NADIRp, TTn, BIS_NADIRn
    