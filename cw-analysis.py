#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from matplotlib import rc
from matplotlib.gridspec import GridSpec
rc("font", family="sans", size=11)

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.transforms
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools
import scipy.stats as stat


# Variable names in file `UHIdata2013.xlsx`:
# 
# - W001_TAIR -	Air temperature (dry bulb) in degrees Celcius at Station W001
# - W006_TAIR -	Air temperature (dry bulb) in degrees Celcius at Station W006
# - W018_TAIR -	Air temperature (dry bulb) in degrees Celcius at Station W018
# - W026_TAIR -	Air temperature (dry bulb) in degrees Celcius at Station W026
# - PC_TAIR -		Air temperature (dry bulb) in degrees Celcius at Station Paradise Circus
# - CH_TAIR -		Air temperature (dry bulb) in degrees Celcius at Station Coleshill
# - CH_WD	- 		Wind direction in degrees at Station Coleshill
# - CH_WS	- 		Wind speed in knots at Station Coleshill
# - CH_RH	- 		Relative Humidity in % at Station Coleshill
# - CH_CLOUD - 	Cloud cover in octas at Station Coleshill
# - CH_HCLOUD -	Cloud base heightin decametres (i.e. *10m)
# 
# The data is imported and a month-long period selected for analysis, including metadata about each recording station.

dateparse = lambda x: pd.datetime.strptime(x, "%d/%m/%Y %H:%M")
df = pd.read_csv("UHIdata2013.csv", skiprows=12, parse_dates=['Time'], date_parser=dateparse)
df['CH_CLOUD'] = df['CH_CLOUD'].astype('Int64')
dt_start = pd.Timestamp(2013,7,11,0,0,0)
dt_fin = pd.Timestamp(2013,8,10,1,59,59)
df = df[(df['Time'] > dt_start) & (df['Time'] < dt_fin)]
df = df[(df['Time'].dt.hour==1)]

stations = {'IDs' : ['W001', 'W006', 'W018', 'W026', 'PC', 'CH'],
            'Names' : ['Sutton Park', 'Handsworth', 'Hodge Hill',
                       'Edgbaston', 'Paradise Circus', 'Coleshill'],
            'Lat' : [52.566353, 52.50451, 52.49249,
                     52.45638, 52.4806, 52.48015],
            'Lon' : [-1.83791, -1.92291, -1.80763, 
                     -1.92767, -1.90493, -1.69075]}

for i in stations['IDs']:
    hisland = str(i+'_TI')
    df[hisland] = df[str(i+'_TAIR')] - df['CH_TAIR']


#display(df)

# From the initially selected data, two weeks are selected to represent periods where northerly or southerly winds are predominant. From this average conditions for the UHI at each site are calculated to provide a mean during each timeframe.

windnord = [pd.Timestamp(2013,7,14),pd.Timestamp(2013,7,21)]
windsud = [pd.Timestamp(2013,7,29),pd.Timestamp(2013,8,5)]

dfnord = df[(df['Time'] > windnord[0]) & (df['Time'] < windnord[1])].reset_index()
dfsud = df[(df['Time'] > windsud[0]) & (df['Time'] < windsud[1])].reset_index()

#display(dfnord)
#display(dfsud)
dfnord.to_csv('northerly_cond.csv')
dfsud.to_csv('southerly_cond.csv')

descriptives = {'N_mean' : dfnord.mean(),
                'N_std': np.std(dfnord, ddof=1),
                'S_mean': dfsud.mean(),
                'S_std': np.std(dfsud, ddof=1)}

df_summ = pd.DataFrame(descriptives)
df_summ = df_summ.loc[[str(i+'_TI') for i in stations['IDs']][0:5],]
display(df_summ)


# Using this data we may usefully plot all our data to provide a record of how the primary variables of interest change with time (ie. temperature, wind etc.). The following plots also include a wind rose for the total period of interest and a comparison between UHI intensity at different sites during northerly/southerly wind conditions.

fig, ax = plt.subplots(figsize=(12,12))
gs = GridSpec(3, 1, height_ratios=[2,2,1], hspace=0.1)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
axs = [ax1, ax2, ax3]

months = mdates.MonthLocator()
days = mdates.DayLocator()
month_fmt = mdates.DateFormatter('%Y/%m/%d')
day_fmt = mdates.DateFormatter('%d')
compass = ['', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', '']
colours = itertools.cycle(('sandybrown', 'salmon', 'indianred',
                           'darkred', 'steelblue', 'darkgreen'))
linestyles = itertools.cycle(('-.', '--', '-.', ':', '-', '-'))

for i in stations['IDs']:
    colour = next(colours)
    line = next(linestyles)
    kwargs = dict(color=colour, linestyle=line, alpha=0.9)
    ax1.plot(df['Time'], df[str(i+'_TAIR')], **kwargs)
    ax2.plot(df['Time'], df[str(i+'_TI')], **kwargs)

cls = ax3.scatter(df['Time'], df['CH_WD'], marker='o', edgecolor='k',
                  c=df['CH_WS'], cmap='Greens', vmin=0)

for j in axs:
    j.xaxis.set_major_locator(months)
    j.xaxis.set_major_formatter(month_fmt)
    j.xaxis.set_minor_locator(days)
    j.xaxis.set_minor_formatter(day_fmt)
    j.set_xlim(dt_start, dt_fin)
    j.axvspan(windnord[0], pd.Timestamp(2013,7,20,2,0,0), color='steelblue', alpha=0.1, lw=0)
    j.axvspan(windsud[0], pd.Timestamp(2013,8,4,2,0,0), color='C2', alpha=0.1, lw=0) 

for k in [ax1, ax2]: 
    k.xaxis.set_major_formatter(day_fmt)
    k.xaxis.set_major_locator(plt.NullLocator())
    #k.set_xticklabels([], minor=True)
    #k.set_xticklabels([], major=True)

offset = matplotlib.transforms.ScaledTranslation(0, -0.2, fig.dpi_scale_trans)
for tick in ax3.xaxis.get_majorticklabels():
        tick.set_transform(tick.get_transform() + offset)
        
ax1.set_ylabel(r'Temperature ($^\circ$C)')
ax2.set_ylabel(r'Heat Island mag. ($T_{site}-T_{CH}$) ($^\circ$C)')
ax3.set_ylabel(r'Wind Bearing')

ax1.legend(stations['IDs'])
ax3.set_yticks(np.arange(0,370,45))
ax3.set_ylim(0,360)
ax3.set_xlabel('Date')
cbaxes = fig.add_axes([0.985, 0.07, 0.01, 0.16]) 
cb = plt.colorbar(cls, cax=cbaxes)
cb.set_label(r'ms$^{-1}$', rotation=0, position=(0, 1.12), ha='right')


sax = ax3.twinx()
sax.set_yticks(np.arange(0,370,45))
sax.set_ylim(0,360)
sax.set_yticklabels(compass, ha='right')
sax.tick_params(direction='in', pad=-5)

gs.tight_layout(fig)
plt.savefig('bham_temperatures.pdf', bbox_inches='tight')
plt.show()


bins = 8

fig = plt.figure(figsize=(5,5))

polar_ax = fig.add_subplot(1, 1, 1, projection="polar")
compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

count, bins = np.histogram(df['CH_WD']+22.5, bins=bins, range=(0,360))
rads = (bins*np.pi)/180
colours = cm.winter_r(count / float(max(count)))
polar_ax.bar(rads[:-1], count, align='center', color='darkgreen', alpha=0.8)

polar_ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
polar_ax.set_xticklabels(compass)
polar_ax.set_theta_zero_location('N')
polar_ax.set_theta_direction(-1)
polar_ax.set_rticks(np.arange(0,max(count),2))
polar_ax.set_rlabel_position(180)

fig.tight_layout()
plt.savefig('bham_winds.pdf', bbox_inches='tight')
plt.show()


x = np.arange(len(stations['IDs'][0:5]))  # the label locations
width = 0.35

fig, ax = plt.subplots(figsize=(10,4))
rects1 = ax.bar(x - width/2, df_summ['N_mean'], width, capsize=3,
                yerr=df_summ['N_std']/np.sqrt(len(dfnord)),
                label='Northerly', color='steelblue')
rects2 = ax.bar(x + width/2, df_summ['S_mean'], width, capsize=3,
                yerr=df_summ['S_std']/np.sqrt(len(dfnord)),
                label='Southerly', color='C2')

ax.axhline(0, color='k', linewidth=1, linestyle=':')
ax.set_ylabel('Mean Temperature Difference ($^\circ$C)')
ax.set_xlabel('Recording Station')
ax.set_xticks(x)
ax.set_xticklabels(stations['IDs'])
ax.legend(title='Wind Direction:')
plt.savefig('UHI_wind_comp.pdf', bbox_inches='tight')
plt.show()


# The following code provides the means to perform a independent samples t-test, either using `paired_t` for equal variances between samples or `welchs_t` for unequal variances. Welch's test is then performed on between each group for every station (seen above) to attain a critical value $t$.

def paired_t(s1,s2):
    means = np.round([np.mean(s1), np.mean(s2)], 4)
    diff = s1-s2
    stddev = np.std(diff, ddof=1)
    t = round(np.mean(diff)/(stddev/np.sqrt(len(s1))), 4)
    #stdvs = [3.1710,3.1710]
    #sdtp = np.sqrt((stdvs[0]**2 + stdvs[1]**2)/2)
    #t = d/(sdtp*np.sqrt(2/len(s1)))
    return t, means, stddev

def welchs_t(s1,s2):
    means = np.round([np.mean(s1), np.mean(s2)], 4)
    diff = s1-s2
    stds = np.round([np.std(s1, ddof=1), np.std(s2, ddof=1)], 4)
    denom = np.sqrt(stds[0]**2/len(s1) + stds[1]**2/len(s2))
    #denom_nu = stds[0]**4/(len(s1)*(len(s1)+1)) + stds[1]**4/(len(s2)*(len(s2)+1))
    t = round(np.mean(diff)/denom, 4)
    #nu = denom**4/(denom_nu)
    
    return t, means, stds#, nu


pairs1 = [dfnord[str(i+'_TI')].values for i in stations['IDs']][0:5]
pairs2 = [dfsud[str(i+'_TI')].values for i in stations['IDs']][0:5]
t_vals = [welchs_t(pairs1[idx],pairs2[idx])[0] for idx, i in enumerate(pairs1)]


print(t_vals)
print([f'p={i}: {stat.t.ppf(1-i/2, 7):.3f}' for i in [0.01, 0.02, 0.05, 0.1]])



