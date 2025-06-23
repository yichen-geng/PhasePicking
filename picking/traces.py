import datetime
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from obspy import Trace
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.taup import TauPyModel

from basic_functions import calculate_distaz, calculate_signal_to_noise_ratio

# define global variables
half_win_len = 30
decimation = True
if decimation:
    sampling_rate = 10
else:
    sampling_rate = 100
# define PREM/IASP91 model
taup_model = TauPyModel("prem")


def preprocess_traces(event_dir, phase='PcP'):
    ev_time_str = os.path.basename(event_dir)
    events = pd.read_csv("../data/metadata/catalog.csv")
    ev_time_strs = events['time'].values
    ev_idx = np.where(ev_time_strs == ev_time_str)[0][0]
    ev = events.iloc[ev_idx]
    ev_time_JST = UTCDateTime(datetime.datetime.strptime(ev_time_str, "%Y%m%d%H%M%S%f")) + timedelta(hours=9)
    ev_lat = ev['latitude']
    ev_lon = ev['longitude']
    ev_dep = ev['depth']
    ev_locs = (ev_lat, ev_lon)

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
    distance_contours = (LON, LAT, Z)

    # load station metadata
    stations = pd.read_csv("../data/metadata/stations.csv")

    st_names = []
    st_locs = []
    distances = []
    tts = []
    sn_ratios = []
    stream_long_data = []
    stream_short_data = []

    # load automatic picks if exist
    autopick = False
    autopicks_path = '../data/autopicks/' + ev_time_str + '_results_' + phase + '.json'
    if os.path.isfile(autopicks_path):
        with open(autopicks_path, 'r') as fp:
            autopicks = json.load(fp)
            autopick = True

    for i in range(len(stations)):
        st = stations.iloc[i]
        st_name = st['name']

        # get station latitude and longitude
        st_lat = st['latitude']
        st_lon = st['longitude']

        # calculate the angular distance between the event and the station
        distance, _, baz = calculate_distaz(st_lat, st_lon, ev_lat, ev_lon)

        # check whether the phase can be observed at this distance
        # calculate predicted arrival time
        arrivals = taup_model.get_travel_times(source_depth_in_km=ev_dep, distance_in_degree=distance,
                                               phase_list=[phase])

        if len(arrivals) != 0:
            travel_time = arrivals[0].time
            # TODO: SAC files are velocity in nm/s or accelaration in nm/s/s
            file_dir = event_dir + "/" + st['name'] + ".U.SAC"
            download_status = os.path.isfile(file_dir)

            # proceed if the waveform data exist
            if download_status:
                trace_short_data, trace_long_data, data_status = \
                    preprocess_single_trace(file_dir, decimation, ev_time_JST, travel_time)

                # load quality factor if exists
                if autopick and st_name[-4:] in autopicks:
                    sn_ratio = autopicks[st_name[-4:]]['qf']
                else:
                    # a quick & dirty way to calculate the signal-to-noise ratio
                    sn_ratio = calculate_signal_to_noise_ratio(trace_short_data, sampling_rate)

                st_names.append(st_name)
                st_locs.append((st_lat, st_lon))
                distances.append(distance)
                tts.append(travel_time)
                sn_ratios.append(sn_ratio)
                stream_long_data.append(trace_long_data)
                stream_short_data.append(trace_short_data)

    # convert to numpy.array
    stream_short_data = np.array(stream_short_data)
    stream_long_data = np.array(stream_long_data)
    st_names = np.array(st_names)
    st_locs = np.array(st_locs)
    distances = np.array(distances)
    tts = np.array(tts)
    sn_ratios = np.array(sn_ratios)
    # sort by signal-to-noise ratio
    sort_indices = np.argsort(sn_ratios)[::-1]
    stream_short_data_sorted = stream_short_data[sort_indices]
    stream_long_data_sorted = stream_long_data[sort_indices]
    st_names_sorted = st_names[sort_indices]
    st_locs_sorted = st_locs[sort_indices]
    distances_sorted = distances[sort_indices]
    tts_sorted = tts[sort_indices]
    sn_ratios_sorted = sn_ratios[sort_indices]

    return ev_locs, distance_contours, \
           st_names_sorted, st_locs_sorted, distances_sorted, tts_sorted, sn_ratios_sorted, \
           stream_long_data_sorted, stream_short_data_sorted


def preprocess_single_trace(file_dir, decimation, ev_time_JST, travel_time,
                            min_freq='raw', max_freq='raw', dt1=-100., dt2=100., min_time=None, max_time=None):
    """
    This function preprocesses a single trace (vertical component).
    :param file_dir: str, directory at which the trace is stored
    :param decimation: boolean, whether to decimate the trace, default: True
    :param ev_time_JST: obspy.core.UTCDateTime, event time in JST
    :param travel_time: float, travel time of the phase in seconds
    :param min_freq: str or float, minimum corner frequency, default: 'raw'
    :param max_freq: str or float, maximum corner frequency, default: 'raw'
    :param dt1: float, trace start time with respect to the arrival time in seconds
    :param dt2: float, trace end time with respect to the arrival time in seconds
    :param min_time: # TODO
    :param max_time: # TODO
    :return:
    trace_BHZ: obspy.core.trace.Trace, 200s normalized trace centered at the predicted arrival time
    data_status: boolean, True if preprocessing succeeds, False otherwise
    """
    # initialize
    # trace_short = Trace()
    trace_long = Trace()
    data_status = False

    # 1. read trace
    stream = read(file_dir)
    trace_BHZ = stream[0]

    sr_old = trace_BHZ.stats.sampling_rate

    # some traces have sampling rate < 100 Hz or total length < 25 mins or a flat response
    if (sr_old == 100) and (len(trace_BHZ.data) == 25 * 60 * sr_old):
        # 2. (optional) decimate by a factor of 10
        if decimation:
            trace_BHZ.decimate(factor=10, no_filter=True)
            # sanity check
            # the original sampling rate of Hinet is 100 Hz, after decimating by a factor of 10,
            # the sampling rate should be 10 Hz
            sr = int(trace_BHZ.stats.sampling_rate)  # cast to int
            assert sr == 10
        else:
            # sanity check
            # the original sampling rate of Hinet is 100 Hz
            sr = int(trace_BHZ.stats.sampling_rate)  # cast to int
            assert sr == 100

        # 3. (optional) filter
        if min_freq != 'raw' and max_freq != 'raw':
            # apply band-pass filter
            # TODO: ask about zerophase
            trace_BHZ.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=2)
        elif min_freq != 'raw':
            # apply high-pass filter
            trace_BHZ.filter('highpass', freq=min_freq, corners=2)

        # 4. select between dt1 and dt2 after the phase arrival
        arrival_time = ev_time_JST + travel_time
        trace_BHZ.trim(starttime=arrival_time + dt1, endtime=arrival_time + dt2)

        # check whether the trace has a flat response
        trace_data_tmp = trace_BHZ.data
        flat = np.all(trace_data_tmp == trace_data_tmp[0])

        if not flat:
            # 5. detrend
            trace_BHZ.detrend(type='linear')

            # 6. normalize
            trace_BHZ.normalize()

            # make long trace
            trace_long = trace_BHZ.copy()

            # make short trace
            trace_BHZ.trim(starttime=arrival_time - half_win_len, endtime=arrival_time + half_win_len)

            data_status = True

    return trace_BHZ.data, trace_long.data, data_status


def plot_traces(trace_data, ax):
    half_win_len_tmp = int(len(trace_data) / 2)
    x = np.arange(-half_win_len_tmp, half_win_len_tmp + 1) / sampling_rate
    y = trace_data
    line, = ax.plot(x, y, color='k', linewidth=0.8)
    ax.set_xlim(-half_win_len_tmp / sampling_rate, half_win_len_tmp / sampling_rate)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    return line


def select_traces(distances, sn_ratios, n_bins=200):
    """
    Select a number of traces with the highest signal-to-noise ratios.
    :param distances: distances between the event and stations
    :param sn_ratios: signal-to-noise ratios of the phase
    :param n_bins: # TODO
    :return: an array of trace indices
    """
    # divide the whole distance range into n bins
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    dist_bins = np.linspace(np.floor(min_distance), np.ceil(max_distance), n_bins)
    indices_max = []
    dist_bin_counts = []

    for j in range(len(dist_bins) - 1):
        # find indices of traces within the distance range
        indices = np.where((distances >= dist_bins[j]) & (distances < dist_bins[j + 1]))[0]
        dist_bin_counts.append(len(indices))
        if len(indices) != 0:
            sn_ratios_select = sn_ratios[indices]
            # find the trace index with the highest signal-to-noise ratio
            indices_max.append(indices[np.argmax(sn_ratios_select)])
    dist_bin_counts = np.array(dist_bin_counts)
    indices_max = np.array(indices_max)

    return dist_bins, dist_bin_counts, indices_max


def plot_record_section(stream_data, distances, station_idx, sn_ratios, ax1, ax2):
    """
    Plot the record section of selected traces. Color the trace to be examined in a different color.
    :param stream_data: a stream of traces centered at the predicted phase arrivals
    :param distances: distances between the event and stations
    :param station_idx: index of the trace to be examined
    :param sn_ratios: signal-to-noise ratios of the phase
    :param ax: axes to be plotted on
    :return: None
    """
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    dist_bins, dist_bin_counts, indices_max = select_traces(distances, sn_ratios, 100)

    # plot station counts as a function of distance
    ax1.plot(dist_bins[:-1], dist_bin_counts, c='k', linewidth=0.8)

    scale = 0.15
    for idx_max in indices_max:
        ax2.plot(distances[idx_max] - stream_data[idx_max] / max(abs(stream_data[idx_max])) * scale,
                 np.arange(-half_win_len * sampling_rate, half_win_len * sampling_rate + 1) / sampling_rate,
                 color='grey', linewidth=0.7)
    trace_data = stream_data[station_idx]

    # flip the trace for easier comparison with waveform plots
    ax2.plot(distances[station_idx] - trace_data / max(abs(trace_data)) * scale,
             np.arange(-half_win_len * sampling_rate, half_win_len * sampling_rate + 1) / sampling_rate, color='r',
             linewidth=0.7)
    # # plot reference time
    # ax2.axhline(y=0, ls='-', color='grey', linewidth=0.5)
    ax2.set_xticks(np.arange(int(np.around(min_distance / 5) * 5), int(np.around(max_distance / 5) * 5) + 5, 5))
    ax2.set_xticklabels(
        np.arange(int(np.around(min_distance / 5) * 5), int(np.around(max_distance / 5) * 5) + 5, 5))
    ax2.set_ylim([-half_win_len, half_win_len])
    ax2.set_xlabel(r"Distance ($^\circ$)")
    ax2.set_ylabel("Time (s)")

# TODO: UserWarning: Warning: converting a masked element to nan.
#   xys = np.asarray(xys)
# TODO: analysis
# TODO: creat git for the project (structures, dependency...)
