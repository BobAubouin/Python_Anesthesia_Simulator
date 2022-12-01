#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:08:08 2022

@author: aubouinb
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


Bis_target = 50
L = []
x = []
for bis in range(30,150):
    x.append(bis)
    L.append((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32)

    
fig, ax = plt.subplots(figsize=[5, 4.5])
ax.plot(x,L)
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.plot(x,L)

ax.set_xlim(28,152)
ax.set_ylim(-5,105)
axins.set_xlim(40, 60)
axins.set_ylim(-0.5,3)
ax.set_xlabel('BIS',  fontsize=14)
ax.set_ylabel('Cost',  fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
axins.tick_params(axis="x", labelsize=14)
axins.tick_params(axis="y", labelsize=14)
# axins.set_xticks([])
# axins.set_yticks([])

# axins.grid()
# ax.grid()
ax.indicate_inset_zoom(axins, edgecolor="black")
plt.savefig('Non_linear_cost.pdf', format='pdf')

#%% delta U NL cost
L = []
x = []
for dU in range(-10,10):
    x.append(dU)
    L.append((dU)**2/100 + ((dU+10)/10)**8)

    
fig, ax = plt.subplots(figsize=[5, 4.5])
ax.plot(x,L)
ax.set_xlabel('$\Delta U$',  fontsize=14)
ax.set_ylabel('Cost',  fontsize=14)
# axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
# axins.plot(x,L)

# ax.set_xlim(-10, 10)
# ax.set_ylim(-0.5,1)
# # axins.set_xlim(-3, 3)
# # axins.set_ylim(-0.5,0.2)

# ax.tick_params(axis="x", labelsize=14)
# ax.tick_params(axis="y", labelsize=14)
# axins.tick_params(axis="x", labelsize=14)
# axins.tick_params(axis="y", labelsize=14)
# # axins.set_xticks([])
# # axins.set_yticks([])

# # axins.grid()
# # ax.grid()
# # ax.indicate_inset_zoom(axins, edgecolor="black")
# plt.savefig('Non_linear_cost.pdf', format='pdf')