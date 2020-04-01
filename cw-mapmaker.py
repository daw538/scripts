#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
rc("font", family="sans", size=11)

# Import mapping packages
import geopandas as gpd
from shapely.geometry import Polygon
from OSGridConverter import latlong2grid
import rasterio as ras
from matplotlib_scalebar.scalebar import ScaleBar


# ## Making Maps
# 
# This notebook will import shapefiles from the OS and UK Land Use sruvey 2015 to produce a map of the Birmingham area.
# 
# ### Location Maps
# 
# Files from the Ordnance Survey Open Roads and Boundary Layers packages are loaded into a geo-dataframe.

# set the filepath and load a shapefile into a geodatabase
fp_maps = '/home/daniel/Documents/maps/'
df_roads = gpd.read_file(fp_maps + 'roads_SP/SP_RoadLink.shp')
df_bound = gpd.read_file(fp_maps + 'bound/district_borough_unitary_region.shp')

polygon = Polygon([(400000,270000),(400000,300000),(430000,300000),(430000,270000),(430000,270000)])
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=df_roads.crs)

df_roads = df_roads[df_roads.geometry.notnull()]
df_bound = df_bound[df_bound.geometry.notnull()]

# Classify roads
df_mway = df_roads[(df_roads['class'] == 'Motorway')]
df_aroad = df_roads[(df_roads['class'] == 'A Road')]
df_broad = df_roads[(df_roads['class'] == 'B Road')]
brum = df_bound[(df_bound['NAME'] == 'Birmingham District (B)')]

# We can take the two full databases and select the features we wish to use in our
# map, for example having layers for the major road network only. Since the
# shapefiles are provided in the OS UK 100km grid system, we need to convert the 
# co-ordinates provided for the meteorological stations so they can be plotted on
# the same graph.

stations = {'IDs' : ['W001', 'W006', 'W018', 'W026', 'PC', 'CH'],
            'Names' : ['Sutton Park', 'Handsworth', 'Hodge Hill',
                       'Elms Road', 'Paradise Circus', 'Coleshill'],
            'Lat' : [52.566353, 52.50451, 52.49249,
                     52.45638, 52.4806, 52.48015],
            'Lon' : [-1.83791, -1.92291, -1.80763, 
                     -1.92767, -1.90493, -1.69075]}

# Transform co-ordinates for met stations
gridn = []
gride = []
for i,j in zip(stations['Lat'], stations['Lon']):
    g = latlong2grid(i,j)
    gridn.append(g.N)
    gride.append(g.E)
    
stations['gridn'] = gridn
stations['gride'] = gride
print(gridn,gride)


# Now we have sufficient information for be able to plot our overview map.

fig, ax = plt.subplots(figsize=(4,4))
ax.set_aspect('equal')

df_bound.plot(ax=ax, zorder=-1, alpha=1,color='white', edgecolors='grey')
brum.plot(ax=ax, zorder=-1, alpha=0.7, color='lightgrey', edgecolors='grey')
df_broad.plot(ax=ax, color='tan')
df_aroad.plot(ax=ax, color='forestgreen')
df_mway.plot(ax=ax, color='dodgerblue')
ax.scatter(stations['gride'], stations['gridn'], color='red', zorder=5, marker='^')
for i, txt in enumerate(stations['IDs']):
    ax.annotate(txt, (stations['gride'][i]+200, stations['gridn'][i]),
                horizontalalignment='left', verticalalignment='bottom',
                family='sans-serif')
ax.set_xlim(400000,425000)
ax.set_ylim(280000,300000)
ax.set_xticks([])
ax.set_yticks([])

scalebar = ScaleBar(1, location='lower right') # 1 pixel = 0.2 meter
ax.add_artist(scalebar)

plt.tight_layout()
#plt.savefig('bham_sites.pdf', bbox_inches='tight')
plt.savefig('bham_sites.png', bbox_inches='tight', dpi=300)
plt.show()


# ### Landuse Maps
# 
# Land use data can be downloaded from the EU Copernicus mission as part of their 
# environmental observations division. Following a short amount of processing we can 
# plot landuse map. Note however that the Copernicus data uses a different 
# co-ordinate system (ETRS 1989 LAEA) which appears to be slightly rotated with
# respect to the OS UK Grid. https://mygeodata.cloud/cs2cs/


fp_maps = '/home/daniel/Documents/maps/'
lnd_wmids = gpd.read_file(fp_maps +
                          'copernicus_wmids/Shapefiles/UK002L3_WEST_MIDLANDS_URBAN_AREA_UA2012.shp')
lnd_wmids['CODE2012'] = lnd_wmids['CODE2012'].astype(int)
cp_cols = pd.read_csv('copernicus_landcolours.csv', delimiter=';', names=['CODE2012', 'colour', 'ITEM2012'])
cp_cols['CODE2012'] = cp_cols['CODE2012'].astype(int)

display(cp_cols.head(10))
display(lnd_wmids.head(5))


landcodes = lnd_wmids['CODE2012'].unique()
landtypes = [lnd_wmids[(lnd_wmids['CODE2012'] == code)] for code in sorted(landcodes)]
stations['ETRS89n'] = [3338224.01652, 3332359.27589, 3329764.77047,
                       3327117.95295, 3329530.00128, 3327129.14958]
stations['ETRS89e'] = [3522375.47415, 3515570.65252, 3523061.3448,
                       3514372.8453, 3516336.64735, 3530658.84985]


fig, ax = plt.subplots(figsize=(12,16))

#ax = fig.add_axes([0, 0.02, 1, 1.1])
for i in landtypes:
    num = int(i['CODE2012'].unique()[0])
    c = cp_cols[(cp_cols['CODE2012'] == num)]['colour'].values[0]
    i.plot(ax=ax, color=c)

legend_patches = []
for i, colour in enumerate(cp_cols['colour']):
    p = mpatches.Patch(color=colour, label=f"{cp_cols['ITEM2012'][i]}")
    legend_patches.append(p)

ax.scatter(stations['ETRS89e'], stations['ETRS89n'], marker='^',
           color='k', edgecolor='w', s=140, linewidth=3)

plt.legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1, -0.01),
           loc='upper left', ncol=2, mode="expand", borderaxespad=0.)
ax.set_xlim(3510000,3535000)
ax.set_ylim(3315000,3340000)
ax.set_xticks([])
ax.set_yticks([])

scalebar = ScaleBar(1, location='lower right') # 1 pixel = 0.2 meter
ax.add_artist(scalebar)

plt.tight_layout()
plt.savefig('bham_landuse.pdf', bbox_inches='tight')
plt.savefig('bham_landuse.png', bbox_inches='tight', dpi=300)
plt.show()


fp_maps = '/home/daniel/Documents/meteorology/m8/maps/'
w001 = ras.open(fp_maps + 'svf_w001_c.tif')
w006 = ras.open(fp_maps + 'svf_w006_c.tif')
w018 = ras.open(fp_maps + 'svf_w018_c.tif')
w026 = ras.open(fp_maps + 'svf_w026_c.tif')
pc = ras.open(fp_maps + 'svf_pc_c.tif')
ch = ras.open(fp_maps + 'svf_ch_c.tif')


svf_plot = {'sta_e' : [ 83,211,179,158,157, 99],
            'sta_n' : [185,178,224,146,142,141],
            'lab_e' : [0.03,0.03,0.03,0.53,0.53,0.53],
            'lab_n' : [0.975,0.640,0.305,0.975,0.640,0.305],
            'svf' : [0.95,0.97,0.98,0.93,0.82,0.998]}

fig, ax = plt.subplots(figsize=(8,12),constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax5 = plt.subplot(gs[4])
ax6 = plt.subplot(gs[5])
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

kwcmap = dict(cmap='Greys_r', vmin=0.65, vmax=1.0)
kwargs = dict(color='red', marker='^', s=80)
props = dict(boxstyle='round', facecolor='white')

ax1.imshow(w001.read(1), **kwcmap)
ax2.imshow(w026.read(1), **kwcmap)
ax3.imshow(w006.read(1), **kwcmap)
cls = ax4.imshow(pc.read(1), **kwcmap)
ax5.imshow(w018.read(1), **kwcmap)
ax6.imshow(ch.read(1), **kwcmap)


for i in range(len(axs)):
    axs[i].scatter(svf_plot['sta_e'][i], svf_plot['sta_n'][i], **kwargs)
    fig.text(svf_plot['lab_e'][i], svf_plot['lab_n'][i],
             f"{stations['IDs'][i]} ({stations['Names'][i]})", bbox=props)
    fig.text(1-svf_plot['lab_e'][5-i], svf_plot['lab_n'][i],
             f"SVF = {svf_plot['svf'][i]}", bbox=props, ha='right')
    axs[i].set_xticks([])
    axs[i].set_yticks([])

cbaxes = fig.add_axes([0.25, -0.02, 0.5, 0.01]) 
cb = plt.colorbar(cls, cax=cbaxes, orientation='horizontal', extend='min')
cb.set_label('Sky View Factor')#, rotation=0, position=(0, 1.12), ha='right')

scalebar = ScaleBar(1, location='lower right') # 1 pixel = 0.2 meter
ax6.add_artist(scalebar)

plt.savefig('maps/skyviewfactors.pdf', bbox_inches='tight', dpi=300)
plt.show()

