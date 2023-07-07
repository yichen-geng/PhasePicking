import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cartopy.feature as cfeature

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize

import matplotlib.ticker as mticker

from matplotlib.lines import Line2D

from matplotlib import cm

from basic.basic_functions import calculate_distaz

data_dir = "../results/picks"
plot_dir = "../results/plots"


def plot_helper_function(st_lats, st_lons, st_vals, ax, cb_label, distance_contours=None, color_map=None, center=False, st_qualities=None):
    min_val = np.min(st_vals)
    max_val = np.max(st_vals)
    if center:
        # center the color bar at 0
        if abs(min_val) > max_val:
            max_val = abs(min_val)
        else:
            min_val = -max_val
    minmax_scale = max_val - min_val
    st_vals_normalized = (st_vals - min_val) / minmax_scale

    if st_qualities is not None:
        # sort
        sort_indices = np.argsort(st_qualities)
        st_lats = st_lats[sort_indices]
        st_lons = st_lons[sort_indices]
        st_vals = st_vals[sort_indices]
        st_vals_normalized = st_vals_normalized[sort_indices]
        st_qualities = st_qualities[sort_indices]

    # ax.set_extent([125, 150, 30, 46])
    ax.set_extent([128, 147, 30, 46])
    # ax.coastlines(facecolor=[0.7, 0.7, 0.7])
    ax.add_feature(cfeature.LAND, facecolor=[0.5, 0.5, 0.5])
    ax.add_feature(cfeature.OCEAN, facecolor=[0.8, 0.8, 0.8])
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([130, 135, 140, 145])
    gl.ylocator = mticker.FixedLocator([32, 36, 40, 44])

    # plot distance contours
    if distance_contours:
        LON, LAT, Z = distance_contours[0], distance_contours[1], distance_contours[2]
        # plot distance contours
        CS = ax.contour(LON, LAT, Z, levels=np.arange(np.floor(np.min(Z)), np.ceil(np.max(Z)) + 1, 1)[::2],
                        colors='grey', linewidths=0.5)
        ax.clabel(CS, CS.levels, inline=True)

    # # plot stations
    # ax.scatter(st_lons, st_lats, s=200, marker='^', c='gold', edgecolors='k', transform=ccrs.PlateCarree())
    # legend_elements = [
    #     Line2D([0], [0], ls='none', marker='^', color='gold', markeredgecolor='k', label='Hi-Net', markersize=24)]
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=32, framealpha=1)

    # plot station corrections
    if st_qualities is not None:
        ax.scatter(st_lons, st_lats, s=st_qualities * 50, color=color_map(st_vals_normalized), alpha=0.8, transform=ccrs.PlateCarree())
    else:
        ax.scatter(st_lons, st_lats, s=5, color=color_map(st_vals_normalized),
                   transform=ccrs.PlateCarree())

    # add color bar
    norm = Normalize(min_val, max_val)

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, orientation='vertical', pad=0.01, location='right')
    cb.set_label(label=cb_label, size=18, labelpad=-10)
    cb.ax.tick_params(labelsize=18)


def plot_station_times_on_map(st_lats, st_lons, st_times, st_qualities,
                              ev_lat=None, ev_lons=None, distance_contours=None, title=None):
    # # plot distribution of station avg. rel. times
    # plt.figure()
    # plt.hist(st_times, bins=np.arange(-10, 10.1, 0.1))
    # plt.xlabel("Station avg. rel. time (s)")

    # keep stations with nonnan time and quality > 0 (nan > 0 is False)
    flags = (~np.isnan(st_times)) & (st_qualities > 0)
    st_lats_selected = st_lats[flags]
    st_lons_selected = st_lons[flags]
    st_times_selected = st_times[flags]
    st_qualities_selected = st_qualities[flags]
    # st_lats_discarded = st_lats[~flags]
    # st_lons_discarded = st_lons[~flags]

    # saturate the outliers
    q1 = np.percentile(st_times_selected, 25)
    q3 = np.percentile(st_times_selected, 75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    # # print(lb, ub)
    # # plot distribution of station avg. rel. times
    # plt.figure()
    # plt.hist(st_times_selected, bins=np.arange(-10, 10.1, 0.1))
    # plt.axvline(lb, c='r', lw=1)
    # plt.axvline(ub, c='r', lw=1)
    # plt.xlabel("Station avg. rel. time (s)")
    # plt.show()

    st_lats_final = st_lats_selected
    st_lons_final = st_lons_selected
    st_times_final = np.where(st_times_selected < lb, lb, st_times_selected)
    st_times_final = np.where(st_times_final > ub, ub, st_times_final)
    st_qualities_final = st_qualities_selected

    fig1 = plt.figure(figsize=(20, 8))
    ax1 = fig1.add_subplot(
        projection=ccrs.PlateCarree(central_longitude=140))  # ccrs stands for cartopy coordinate reference system
    ax1.set_extent([60, 180, -20, 60])
    ax1.coastlines()
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                       linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    data_crs = ccrs.PlateCarree()

    # plot earthquake location
    ax1.scatter(ev_lon, ev_lat, s=100, marker=(5, 1), c='r', edgecolors='k', transform=data_crs)

    fig2 = plt.figure(figsize=(20, 8))
    ax21 = fig2.add_subplot(111, projection=ccrs.PlateCarree())  # ccrs stands for cartopy coordinate reference system
    # create color map
    color_map = cm.get_cmap('seismic')
    plot_helper_function(st_lats_final, st_lons_final, st_times_final, ax21, "Time (s)",
                         distance_contours=distance_contours, color_map=color_map, center=True, st_qualities=st_qualities_final)

    if title:
        plt.suptitle(title)


# load station info
stations = pd.read_csv("../data/metadata/stations.csv")
station_lats = stations['latitude'].values
station_lons = stations['longitude'].values
n_stations = len(stations)

# load event info
events = pd.read_csv("../data/metadata/catalog.csv")
ev_time_strs = events['time'].values

phase = 'PcP'

ev_time_str = "20070429124157400000"
ev_idx = np.where(ev_time_strs == ev_time_str)[0][0]
ev = events.iloc[ev_idx]
origin_time = datetime.datetime.strptime(ev['time'], "%Y%m%d%H%M%S%f")
ev_lat = ev['latitude']
ev_lon = ev['longitude']
ev_dep = ev['depth']
ev_mag = ev['mag']
print("UTC time:", origin_time)
print("Latitude: {:.2f} degrees".format(ev_lat))
print("Longitude: {:.2f} degrees".format(ev_lon))
print("Depth: {:.2f} km".format(ev_dep))
print("Moment magnitude: {:.1f} Mw".format(ev_mag))
print("")

# load results
with open(data_dir + "/" + ev_time_str + ".json", 'r') as fp:
    results = json.load(fp)

# create distance contours
LON = np.arange(125, 150.1, 0.1)
LAT = np.arange(30, 46.1, 0.1)
LON, LAT = np.meshgrid(LON, LAT)
Z = np.zeros_like(LON)
for i in range(LON.shape[0]):
    for j in range(LON.shape[1]):
        lon, lat = LON[i][j], LAT[i][j]
        distance, _, _ = calculate_distaz(lat, lon, ev_lat, ev_lon)
        Z[i][j] = distance

st_lats = []
st_lons = []
st_times =[]
st_qualities = []

# for each station calculate the average of the relative times of the stations within a given distance range
for j in range(n_stations):
    st_name = st_names[j][-4:]
    if st_name in results.keys():
        st_time = results[st_name]['time_deviate'][0]
        st_quality = results[st_name]['qf']
        idx_tmp = np.where(station_elevations['name'].values == ("N." + st_name))[0]
        if len(idx_tmp) != 0:
            idx_tmp = idx_tmp[0]
            st_lats.append(station_lats[j])
            st_lons.append(station_lons[j])
            st_times.append(st_time)
            st_qualities.append(st_quality)
        # else:
        #     print(st_name1)

st_lats = np.array(st_lats)
st_lons = np.array(st_lons)
st_eles = np.array(st_eles)
st_times = np.array(st_times)
st_qualities = np.array(st_qualities)

# # exclude nans
# flags = ~np.isnan(st_times)
#
# st_lats = st_lats[flags]
# st_lons = st_lons[flags]
# st_eles = st_eles[flags]
# st_times = st_times[flags]
# st_qualities = st_qualities[flags]

# title = str(origin_time) + ", " + str(ev_lat) + ", " + str(ev_lon) + ", " + str(ev_dep) + "km, Mw" + str(ev_mag)
plot_station_times_on_map(ev_lat, ev_lon, st_lats, st_lons, st_times, st_qualities,
                          distance_contours=(LON, LAT, Z))
plt.tight_layout()
# plt.show()
plt.savefig(plot_dir + "/station_times_filled2.png", dpi=150)
plt.close('all')
