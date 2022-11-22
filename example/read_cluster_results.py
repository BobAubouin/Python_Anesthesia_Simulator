#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:16:03 2022

@author: aubouinb
"""
import pandas as pd
import numpy as np
from src.PAS import Patient, disturbances, metrics
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
import matplotlib.pyplot as plt
from bokeh.io import export_svg

Number_of_patient = 32
Data = pd.read_csv("result_NMPC_n="+str(Number_of_patient)+'.csv')

IAE_list = []
TT_list = []
ST10_list = []
ST20_list = []
US_list = []
BIS_NADIR_list = []

p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)

for i in range(Number_of_patient):
    

    BIS = Data[str(i)+'_BIS']
    Time = np.arange(0,len(BIS))*5/60
    p1.line(Time, BIS, legend_label='internal target')
    # p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
    p2.line(Time, Data[str(i)+'_MAP'], legend_label='MAP (mmgh)')
    p2.line(Time, Data[str(i)+'_CO']*10, legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(Time, Data[str(i)+'_Up'], line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(Time, Data[str(i)+'_Ur'], line_color="#f46d43", legend_label='remifentanil (ng/min)')
    
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(Data[str(i)+'_BIS'], Te = 5, phase = 'induction')
    TT_list.append(TT)
    BIS_NADIR_list.append(BIS_NADIR)
    ST10_list.append(ST10)
    ST20_list.append(ST20)
    US_list.append(US)

p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3,p1,p2))

show(grid)

result_table = pd.DataFrame()
result_table.insert(len(result_table.columns),"", ['mean','std','min','max'])
result_table.insert(len(result_table.columns),"TT (min)", [np.round(np.nanmean(TT_list),2),
                                                   np.round(np.nanstd(TT_list),2),
                                                   np.round(np.nanmin(TT_list),2),
                                                   np.round(np.nanmax(TT_list),2)])
result_table.insert(len(result_table.columns),"BIS_NADIR", [np.round(np.nanmean(BIS_NADIR_list),2),
                                                   np.round(np.nanstd(BIS_NADIR_list),2),
                                                   np.round(np.nanmin(BIS_NADIR_list),2),
                                                   np.round(np.nanmax(BIS_NADIR_list),2)])
result_table.insert(len(result_table.columns),"ST10 (min)", [np.round(np.nanmean(ST10_list),2),
                                                   np.round(np.nanstd(ST10_list),2),
                                                   np.round(np.nanmin(ST10_list),2),
                                                   np.round(np.nanmax(ST10_list),2)])
result_table.insert(len(result_table.columns),"ST20 (min)", [np.round(np.nanmean(ST20_list),2),
                                                   np.round(np.nanstd(ST20_list),2),
                                                   np.round(np.nanmin(ST20_list),2),
                                                   np.round(np.nanmax(ST20_list),2)])
result_table.insert(len(result_table.columns),"US", [np.round(np.nanmean(US_list),2),
                                                   np.round(np.nanstd(US_list),2),
                                                   np.round(np.nanmin(US_list),2),
                                                   np.round(np.nanmax(US_list),2)])

print(result_table.to_latex(index=False))