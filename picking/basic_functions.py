"""
This script defines some basic functions.
Created on: 2021/04/13
Last update: 2022/06/23
"""

import numpy as np
from obspy.taup import TauPyModel


class Constants:
    max_distance = 40  # maximum distance of travel time
    epsilon = 1  # increment in distance


def get_travel_time(phase, event_depth, distances=None, min_d=0., max_d=90., epsilon=Constants.epsilon, first=False,
                    all=False):
    """
    This function calculates the predicted travel times for the given phase at given distances/distance range.
    :param phase: str, name of the phase
    :param event_depth: float, event depth
    :param distances: array, distances at which travel times are calculated
    :param min_d: float, minimum distance, default: 0
    :param max_d: float, maximum distance, default: 90
    :param epsilon: float, step size for distance, default: Constants.epsilon
    :param first: boolean, if true, at a given distance, only the time for the first arrival is returned,
    useful for phases with multiple arrivals at a given distance (e.g., P in the triplication region), default: False
    :param all: boolean, if true, return travel times for all distances;
    if the travel time at a given distance is undefined, it is set to NaN;
    otherwise, return travel times at distance ranges only at which they are valid, default: False
    :return:
    distances_new: array, distances
    travel_times: array, travel times at given distances
    """
    # define PREM model
    model = TauPyModel("prem")

    if distances is None:
        distances = np.arange(min_d, max_d + epsilon, epsilon)
    distances_new = []
    travel_times = []
    ray_params = []
    for distance in distances:
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance,
                                          phase_list=[phase])
        if len(arrivals) > 0:
            if first:
                distances_new.append(distance)
                travel_times.append(arrivals[0].time)
                ray_params.append(arrivals[0].ray_param)
            else:
                for arrival in arrivals:
                    distances_new.append(distance)
                    travel_times.append(arrival.time)
                    ray_params.append(arrival.ray_param)
        else:
            if all:
                distances_new.append(distance)
                travel_times.append(np.nan)
                ray_params.append(np.nan)

    return np.array(distances_new), np.array(travel_times), np.array(ray_params)


def get_travel_time_for_plotting(phase, event_depth, distances=None, min_d=0., max_d=90., epsilon=Constants.epsilon):
    """
    This function calculates the predicted travel times for the given phase at given distances/distance range.
    :param phase: str, name of the phase
    :param event_depth: float, event depth
    :param distances: array, distances at which travel times are calculated
    :param min_d: float, minimum distance, default: 0
    :param max_d: float, maximum distance, default: 90
    :param epsilon: float, step size for distance, default: Constants.epsilon
    :return:
    distances_new: array, distances
    travel_times: array, travel times at given distances
    """
    # define PREM model
    model = TauPyModel("prem")

    if distances is None:
        distances = np.arange(min_d, max_d + epsilon, epsilon)
    distances_first = []
    travel_times_first = []
    distances_all = []
    travel_times_all = []
    for distance in distances:
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance,
                                          phase_list=[phase])
        if len(arrivals) > 0:
            distances_first.append(distance)
            travel_times_first.append(arrivals[0].time)
            for arrival in arrivals:
                distances_all.append(distance)
                travel_times_all.append(arrival.time)

    return np.array(distances_first), np.array(travel_times_first), np.array(distances_all), np.array(travel_times_all)


def least_square_fit(x_data, y_data, weights=None):
    """

    :param x_data:
    :param y_data:
    :param weights:
    :return:
    """
    # non-matrix form
    if weights is not None:
        x_mean = np.sum(x_data * weights) / np.sum(weights)
        y_mean = np.sum(y_data * weights) / np.sum(weights)
        beta1 = np.sum((x_data - x_mean) * (y_data - y_mean) * weights) / np.sum((x_data - x_mean) ** 2 * weights)
    else:
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)
        beta1 = np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2)
    beta0 = y_mean - beta1 * x_mean
    return beta0, beta1


def detrend(y_data):
    """
    # TODO: least-square fit
    :param y_data:
    :return:
    """
    x_data = np.arange(len(y_data))
    beta0, beta1 = least_square_fit(x_data, y_data)
    y_data_detrend = y_data - (beta0 + beta1 * x_data)
    # # matrix form
    # X = x_data[:, np.newaxis]**[0, 1]
    # Y = y_data[:, np.newaxis]
    # p = scipy.linalg.inv(X.T @ X) @ X.T @ Y
    # y_data_detrend = y_data - (p[0] + p[1] * x_data)
    return y_data_detrend


def calculate_distaz(st_lat, st_lon, ev_lat, ev_lon):
    """
    This function calculates the distance, azimuth and backazimuth for an event-station pair.
    :param st_lat: float, station latitude in degrees
    :param st_lon: float, station longitude in degrees
    :param ev_lat: float, event latitude in degrees
    :param ev_lon: float, event longitude in degrees
    :return:
    delta: float, distance in degrees
    azi: float, azimuth in radians
    baz: float, backazimuth in radians
    """
    # convert degree to radian
    st_lat = st_lat / 180 * np.pi
    st_lon = st_lon / 180 * np.pi

    ev_lat = ev_lat / 180 * np.pi
    ev_lon = ev_lon / 180 * np.pi

    phi_s = st_lat
    lambda_s = st_lon

    phi_e = ev_lat
    lambda_e = ev_lon

    d_lambda = abs(lambda_s - lambda_e)

    # great-circle distance
    delta = np.arccos(np.sin(phi_s) * np.sin(phi_e) + np.cos(phi_s) * np.cos(phi_e) * np.cos(d_lambda))

    # azimuth [-pi, pi]
    azi = np.arctan2(np.cos(phi_s) * np.cos(phi_e) * np.sin(lambda_s - lambda_e),
                     np.sin(phi_s) - np.cos(delta) * np.sin(phi_e))

    # convert to [0, 2*pi]
    azi = np.where(azi > 0, azi, azi + 2 * np.pi)

    baz = np.arctan2(-np.cos(phi_s) * np.cos(phi_e) * np.sin(lambda_s - lambda_e),
                     np.sin(phi_e) - np.cos(delta) * np.sin(phi_s))
    # convert to [0, 2*pi]
    baz = np.where(baz > 0, baz, baz + 2 * np.pi)

    # convert back to degree
    delta = delta / np.pi * 180

    return delta, azi, baz


def calculate_distance(lat1, lon1, lat2, lon2):
    """

    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    # convert degree to radian
    lat1 = lat1 / 180 * np.pi
    lon1 = lon1 / 180 * np.pi

    lat2 = lat2 / 180 * np.pi
    lon2 = lon2 / 180 * np.pi

    phi_s = lat1
    lambda_s = lon1

    phi_e = lat2
    lambda_e = lon2

    d_lambda = abs(lambda_s - lambda_e)

    # great-circle distance
    delta = np.arccos(np.sin(phi_s) * np.sin(phi_e) + np.cos(phi_s) * np.cos(phi_e) * np.cos(d_lambda))

    # convert back to degree
    delta = delta / np.pi * 180

    return delta


def calculate_p_signal_to_noise_ratio_naive(trace, taup_model, ev_time, ev_depth, distance):
    """
    This function implements a naive calculation of signal-to-noise ratio of the phase.
    It takes 5s before and after the predicted arrival of the phase as the signal window and 15-5s before the predicted
    arrival of the phase as the noise window and then calculates the sn ratio as the ratio between the mean absolute
    amplitude of two windows.
    :param trace: obspy.core.trace.Trace, raw trace
    :param phase: str, seismic phase
    :param taup_model: obspy.taup.TauPyModel, taup model to calculate the predicted travel time for the phase
    :param ev_time: obspy.core.utcdatetime.UTCDateTime, event time
    :param ev_depth: float, event depth in km
    :param distance: float, angular distance between the event and the station in degrees
    :return: sn_ratio: float, signal-to-noise ratio of the phase
    """
    # get the phase travel time
    arrivals = taup_model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=distance,
                                           phase_list=['P'])
    phase_travel_time = arrivals[0].time
    phase_arrival_time = ev_time + phase_travel_time

    trace1 = trace.copy()
    trace2 = trace.copy()

    # get the noise window, 20-10s before phase arrival
    noise_window = trace1.trim(starttime=phase_arrival_time - 20, endtime=phase_arrival_time - 10)
    # calculate the mean amplitude
    noise_amplitude = np.mean(abs(noise_window.data))

    # get the signal window, 10s after phase arrival
    signal_window = trace2.trim(starttime=phase_arrival_time, endtime=phase_arrival_time + 10)
    # calculate the mean amplitude
    signal_amplitude = np.mean(abs(signal_window.data))

    # calculate the signal-to-noise ratio
    sn_ratio = signal_amplitude / noise_amplitude

    return sn_ratio


def select_trace_window(trace_data, sampling_rate, init_window_len=10, selected_window_len=10, predicted=False,
                        noise=False, delta_t=0):
    """
    Select a signal window centered at the maximum absolute amplitude or at the predicted arrival time.
    :param trace_data: array, raw trace
    :param sampling_rate: int, sampling rate of the raw trace
    :param init_window_len: int, length of the initial window
    :param selected_window_len: int, length of the final window
    :param predicted: boolean, whether to center the trace at the predicted arrival time
    :param noise: boolean, if True, the predicted arrival time is randomized; otherwise, the predicted arrival time is
    calculated by TauP, default: False
    :param delta_t: int, the time shift from the predicted arrival time in seconds, used only when noise=True,
    default: 0
    :return:
    time_shift: int, difference in indices between max abs amplitude and predicted arrival
    start: int, index of the start of window
    end: int, index of the end of window
    """
    trace_len = len(trace_data)
    # trace is centered at the predicted arrival time, so idx_pred should be the midpoint
    idx_pred = int(trace_len / 2)  # midpoint

    if noise:
        # pick a point within +-20s of the midpoint as the PcP arrival time
        idx_pred += delta_t * sampling_rate

    # calculate start and end indices in the raw trace
    idx_start = idx_pred - int(np.around(
        init_window_len / 2 * sampling_rate))  # TODO: double check whether np.around() is needed here and same for the rest
    idx_end = idx_pred + int(np.around(init_window_len / 2 * sampling_rate))
    if predicted:
        return 0, idx_start, idx_end
    else:
        trace_window = trace_data[idx_start:idx_end]
        # find the index of the maximum absolute amplitude in the trace window
        # note that this index is defined with respect to the window not the full trace
        idx_max_rel = np.argmax(abs(trace_window))
        # calculate the difference between index of max abs amplitude (actual?) and index of predicted PcP arrival time
        time_shift = idx_max_rel - int(np.around(init_window_len / 2 * sampling_rate))
        # calculate the index of the maximum absolute amplitude with respect to the full trace
        idx_max = time_shift + idx_pred
        # center the trace window at the index of max abs amplitude
        idx_start = idx_max - int(np.around(selected_window_len / 2 * sampling_rate))
        idx_end = idx_max + int(np.around(selected_window_len / 2 * sampling_rate))
        return time_shift, idx_start, idx_end


def calculate_signal_to_noise_ratio(trace_data, sampling_rate,
                                    init_window_len=10, signal_window_len=4, noise_window_len=10, noise=False,
                                    delta_t=0):
    """
    This function calculates the signal-to-noise ratio for a phase.
    It first finds a signal window centered at the maximum absolute amplitude and then
    takes a noise window either before the P arrival or the signal window.
    Signal-to-noise ratio is calculated as the ratio between the mean absolute amplitude of two windows.
    :param trace_data: obspy.core.trace.Trace, raw trace
    :param phase: str, seismic phase
    :param sampling_rate: int, sampling rate of the raw trace
    :param taup_model: obspy.taup.TauPyModel, taup model to calculate the predicted travel time for the phase
    :param ev_time: obspy.core.utcdatetime.UTCDateTime, event time
    :param ev_depth: float, event depth in km
    :param distance: float, angular distance between the event and the station in degrees
    :param init_window_len: int, length of the initial window
    :param signal_window_len: int, length of the signal window
    :param noise_window_len: int, length of the noise window
    :param noise: boolean, if True, the predicted arrival time is randomized; otherwise, the predicted arrival time is
    calculated by TauP, default: False
    :param delta_t: int, the time shift from the predicted arrival time in seconds, used only when noise=True,
    default: 0
    :return: sn_ratio: float, signal-to-noise ratio of the phase
    """
    # _s signal window
    _, idx_start, idx_end = select_trace_window(trace_data, sampling_rate, init_window_len, signal_window_len,
                                                noise=noise, delta_t=delta_t)
    # get the signal window
    signal_data = trace_data[idx_start:idx_end]
    signal_data = detrend(signal_data)
    # calculate the mean amplitude
    signal_amplitude = np.mean(abs(signal_data))
    # get the noise window
    noise_data = trace_data[idx_start - int(noise_window_len * sampling_rate):idx_start]
    noise_data = detrend(noise_data)
    # calculate the mean amplitude
    noise_amplitude = np.mean(abs(noise_data))

    # calculate the signal-to-noise ratio
    sn_ratio = signal_amplitude / noise_amplitude

    return sn_ratio


def calculate_weighted_mean_std(x, errs):
    """
    Calculate the weighted mean and the standard deviation of the weighted mean based on Taylor, 1996.
    :param x: array, measurements
    :param errs: array, errors of the measurements
    :return:
    weighted_mean: float, weighted mean
    std: float, standard deviation of the weighted mean
    """
    weights = 1. / errs ** 2
    weighted_mean = np.sum(weights * x) / np.sum(weights)
    std = 1 / np.sqrt(np.sum(weights))
    return weighted_mean, std


def get_last_number(numbers):
    """
    Takes a list of numbers and returns the largest number in the smallest consecutive group
    :param numbers: a list or array of numbers
    :return: the largest number in the smallest consecutive group
    """
    # sort by ascending values
    numbers = np.sort(numbers)

    last_number = -1

    for number in numbers:
        if number == last_number + 1:
            last_number = number
        else:
            break

    last_number = int(last_number)  # convert from numpy.int to int

    return last_number
