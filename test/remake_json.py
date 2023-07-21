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
from obspy.taup import TauPyModel

from picking.basic_functions import calculate_distaz


# load station info
stations = pd.read_csv("../data/metadata/stations.csv")

# load event info
events = pd.read_csv("../data/metadata/catalog.csv")
ev_time_strs = events['time'].values

ev_time_str = "20161006155159200000"
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

data_dir = "../results/picks"

# define PREM/IASP91 model
taup_model = TauPyModel("prem")

ev_time_str = "20161006155159200000"

# load results
with open(data_dir + "/" + ev_time_str + ".json", 'r') as fp:
    results = json.load(fp)

# keys = ['number', 'latitude', 'longitude', 'distance', 'travel_time', 'pick', 'quality']
#
# for st_name in results:
#     st_results = results[st_name]
#     st_info = stations[stations['name'] == st_name]
#     st_lat = st_info['latitude'].values[0]
#     st_lon = st_info['longitude'].values[0]
#     distance, _, baz = calculate_distaz(st_lat, st_lon, ev_lat, ev_lon)
#     arrivals = taup_model.get_travel_times(source_depth_in_km=ev_dep, distance_in_degree=distance,
#                                            phase_list=['PcP'])
#     st_tt = arrivals[0].time
#     results[st_name]['latitude'] = st_lat
#     results[st_name]['longitude'] = st_lon
#     results[st_name]['distance'] = distance
#     results[st_name]['travel_time'] = st_tt
#
#     # reorder dictionary
#     st_results_reordered = {k: st_results[k] for k in keys}
#
#     results[st_name] = st_results_reordered
#
# # save
# with open('../results/picks/' + ev_time_str + '.json', 'w') as fp:
#     json.dump(results, fp, indent=4)

for st_name in results:
    st_results = results[st_name]

    # rename key
    st_results_renamed = {'picks' if k == 'pick' else k:v for k,v in st_results.items()}

    results[st_name] = st_results_renamed

# save
with open('../results/picks/' + ev_time_str + '.json', 'w') as fp:
    json.dump(results, fp, indent=4)
