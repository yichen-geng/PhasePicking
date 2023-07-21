import datetime
import json

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from obspy.taup import TauPyModel

from basic_functions import calculate_distaz

data_dir = "../results/picks"
plot_dir = "../results/plots"

# define PREM/IASP91 model
taup_model = TauPyModel("prem")


def plot_station_times_on_map(results, fig, plot_type='residual', quality_type='score', distance_contours=None, title=None):
    # TODO: add doc
    st_lats = []
    st_lons = []
    st_picks = []
    st_qualities = []
    tts = []
    st_lats_discarded = []
    st_lons_discarded = []

    for _, st_results in results.items():
        st_lat = st_results['latitude']
        st_lon = st_results['longitude']
        # distance = st_results['distance']
        travel_time = st_results['travel_time']
        picks = st_results['picks']
        if len(picks) == 3:
            st_pick = picks[1]
            # calculate uncertainty
            st_unc = picks[2] - picks[0]
            st_score = int(st_results['quality'])
            if quality_type == 'score':
                st_quality = st_score
            elif quality_type == 'uncertainty':
                st_quality = 1 / (st_unc + 1)
            else:
                raise Exception  # create exception message
            st_lats.append(st_lat)
            st_lons.append(st_lon)
            st_picks.append(st_pick)
            st_qualities.append(st_quality)
            tts.append(travel_time)
        else:
            st_lats_discarded.append(st_lat)
            st_lons_discarded.append(st_lon)

    st_lats = np.array(st_lats)
    st_lons = np.array(st_lons)
    st_picks = np.array(st_picks)
    st_qualities = np.array(st_qualities)
    tts = np.array(tts)
    st_lats_discarded = np.array(st_lats_discarded)
    st_lons_discarded = np.array(st_lons_discarded)

    # saturate the outliers
    q1 = np.percentile(st_picks, 25)
    q3 = np.percentile(st_picks, 75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    st_picks = np.where(st_picks < lb, lb, st_picks)
    st_picks = np.where(st_picks > ub, ub, st_picks)

    if plot_type == 'residual':
        st_times = st_picks
        color_map = cm.get_cmap('seismic')
        cb_label = "Residual time (s)"
    elif plot_type == 'arrival':
        st_times = st_picks + tts
        color_map = cm.get_cmap('inferno_r')
        cb_label = "Arrival time (s)"
    else:
        raise Exception  # create exception message

    # normalize
    min_time = np.min(st_times)
    max_time = np.max(st_times)
    # if center:
    #     # center the color bar at 0
    #     if abs(min_time) > max_time:
    #         max_time = abs(min_time)
    #     else:
    #         min_time = -max_time
    minmax_scale = max_time - min_time
    st_times_normalized = (st_times - min_time) / minmax_scale

    # sort by ascending quality
    sort_indices = np.argsort(st_qualities)
    st_lats = st_lats[sort_indices]
    st_lons = st_lons[sort_indices]
    st_times_normalized = st_times_normalized[sort_indices]
    st_qualities = st_qualities[sort_indices]

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())  # ccrs stands for cartopy coordinate reference system
    # ax.set_extent([125, 150, 30, 46])
    ax.set_extent([128, 147, 30, 46])
    # ax.coastlines(facecolor=[0.7, 0.7, 0.7])
    ax.add_feature(cfeature.LAND, facecolor=[0.5, 0.5, 0.5])
    ax.add_feature(cfeature.OCEAN, facecolor=[0.8, 0.8, 0.8])
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color=[0.65, 0.65, 0.65], alpha=0.5, linestyle='--', zorder=9)
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
                        colors='w', linewidths=0.5, zorder=8)
        ax.clabel(CS, CS.levels, inline=True, zorder=8)

    scale = 3

    # # plot discarded stations
    # ax.scatter(st_lons_discarded, st_lats_discarded, s=scale, marker='^', color='k', edgecolors='none', alpha=0.8, transform=ccrs.PlateCarree(), zorder=9)

    # plot station times
    sc = ax.scatter(st_lons, st_lats, s=st_qualities * scale, color=color_map(st_times_normalized), edgecolors='none', alpha=0.8, transform=ccrs.PlateCarree(), zorder=10)
    legend = ax.legend(*sc.legend_elements("sizes", color='k', markeredgecolor='none', alpha=0.8, func=lambda s: s / scale),
                       title="Quality", fontsize=6, loc='lower right', framealpha=1)
    legend.get_title().set_fontsize('6')
    legend.get_frame().set_edgecolor('grey')
    legend.set_zorder(11)

    # add color bar
    norm = Normalize(min_time, max_time)

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, orientation='vertical', pad=0.02, location='right', shrink=0.8)
    cb.set_label(label=cb_label, size=6)
    cb.ax.tick_params(labelsize=6)

    if title:
        plt.title(title, fontsize=8)


def plot_earthquake_location(st_lat, st_lon, ev_lat, ev_lon, fig):
    ax = fig.add_subplot(
        projection=ccrs.PlateCarree(central_longitude=140))  # ccrs stands for cartopy coordinate reference system
    ax.set_extent([60, 200, -20, 60])
    # ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor=[0.5, 0.5, 0.5])
    ax.add_feature(cfeature.OCEAN, facecolor=[0.8, 0.8, 0.8])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color=[0.65, 0.65, 0.65], alpha=0.5, linestyle='--', zorder=9)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}

    data_crs = ccrs.PlateCarree()

    # plot station location
    ax.scatter(st_lon, st_lat, s=50, marker='^', c='gold', edgecolors='k', transform=data_crs, zorder=10, label="avg. station loc.")

    # plot earthquake location
    ax.scatter(ev_lon, ev_lat, s=100, marker=(5, 1), c='r', edgecolors='k', transform=data_crs, zorder=10, label="earthquake")

    # plot the box
    ax.add_patch(mpatches.Rectangle(xy=(128, 30), width=19, height=16, facecolor='none', edgecolor='b', transform=ccrs.PlateCarree()))

    legend = ax.legend(fontsize=6, loc='lower left', labelspacing=1)
    legend.get_frame().set_edgecolor('grey')
    legend.set_zorder(11)


# # load event info
# events = pd.read_csv("../data/metadata/catalog.csv")
# ev_time_strs = events['time'].values
#
# ev_time_str = "20070429124157400000"
# ev_idx = np.where(ev_time_strs == ev_time_str)[0][0]
# ev = events.iloc[ev_idx]
# origin_time = datetime.datetime.strptime(ev['time'], "%Y%m%d%H%M%S%f")
# ev_lat = ev['latitude']
# ev_lon = ev['longitude']
# ev_dep = ev['depth']
# ev_mag = ev['mag']
# print("UTC time:", origin_time)
# print("Latitude: {:.2f} degrees".format(ev_lat))
# print("Longitude: {:.2f} degrees".format(ev_lon))
# print("Depth: {:.2f} km".format(ev_dep))
# print("Moment magnitude: {:.1f} Mw".format(ev_mag))
# print("")
#
# # load station metadata
# stations = pd.read_csv("../data/metadata/stations.csv")
#
# station_lat_avg = np.mean(stations['latitude'].values)
# station_lon_avg = np.mean(stations['longitude'].values)
#
# # TODO: change longitude and latitude labels
# fig = plt.figure(figsize=(5, 4), dpi=300)
# plot_earthquake_location(station_lat_avg, station_lon_avg, ev_lat, ev_lon, fig)
# plt.show()
#
# # create distance contours
# LON = np.arange(125, 150.1, 0.1)
# LAT = np.arange(30, 46.1, 0.1)
# LON, LAT = np.meshgrid(LON, LAT)
# Z = np.zeros_like(LON)
# for i in range(LON.shape[0]):
#     for j in range(LON.shape[1]):
#         lon, lat = LON[i][j], LAT[i][j]
#         distance, _, _ = calculate_distaz(lat, lon, ev_lat, ev_lon)
#         Z[i][j] = distance
# distance_contours = (LON, LAT, Z)
#
# # load results
# with open(data_dir + "/" + ev_time_str + ".json", 'r') as fp:
#     results = json.load(fp)
#
# fig = plt.figure(figsize=(5, 4), dpi=300)
# title = str(origin_time) + ", " + str(ev_lat) + ", " + str(ev_lon) + ", " + str(ev_dep) + "km, Mw" + str(ev_mag)
# plot_station_times_on_map(results, fig, plot_type='arrival', distance_contours=distance_contours, title=title)
# plt.tight_layout()
# plt.show()
