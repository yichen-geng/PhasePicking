"""
This script contains functions useful for multiple packages.
Created on: 2021/04/13
Last update: 2022/06/23
"""

import random

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.linalg
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
from matplotlib.colors import Normalize
from obspy.signal.invsim import cosine_taper
from obspy.taup import TauPyModel


class Constants:
    R = 6371
    r_core_mantle = 3480
    r_inner_core = 1221.5
    max_distance = 40  # maximum distance of travel time
    epsilon = 1  # increment in distance


def calculate_p_wave_radiation_amplitude(moment_tensor, takeoff_angle, azimuth):
    """
    Calculate the P wave amplitude due to radiation pattern from moment tensor, takeoff angle and azimuth.
    Equation is based on Fitch et al., 1980.
    :param moment_tensor: array, 3x3 moment tensor defined in the geographic coordinates N, E and down
    :param takeoff_angle: float, takeoff angle measured up from the downward vertical in radians
    :param azimuth: float, azimuth measured clockwise from N in radians
    :return: A: float, P wave radiation amplitude
    """
    m = moment_tensor
    theta = takeoff_angle
    phi = azimuth

    m_xx = m[0, 0]
    m_yy = m[1, 1]
    m_zz = m[2, 2]
    m_xy = m[0, 1]
    m_xz = m[0, 2]
    m_yz = m[1, 2]

    a = m_xx * np.sin(theta) ** 2 * np.cos(phi) ** 2 + \
        m_yy * np.sin(theta) ** 2 * np.sin(phi) ** 2 + \
        m_zz * np.cos(theta) ** 2 + \
        m_xy * np.sin(theta) ** 2 * np.sin(2 * phi) + \
        m_xz * np.sin(2 * theta) * np.cos(phi) + \
        m_yz * np.sin(2 * theta) * np.sin(phi)
    return a


def calculate_amplitude_decay(takeoff_angle, dt1, dt2):
    """
    Calculate the scaling factor for the body-wave amplitude decay due to geometrical spreading.
    Equation for geometrical spreading is based on Lay & Wallace, 1995 and Dahlen & Tromp, 1998.
    :param takeoff_angle: float, incidence angle at the source in radians
    :param dt1: float, first derivative of the travel time in s/radian
    :param dt2: float, second derivative of the travel time in s/radian^2  # TODO: double check the unit is correct
    :return: a, float, the scaling factor for the body-wave amplitude decay
    """
    # incidence angle at the source
    i_s = takeoff_angle

    # calculate the incidence angle at the receiver
    r_0 = 6368
    v_0 = 5.800
    i_0 = np.arcsin(dt1 * v_0 / r_0)

    # calculate the scaling factor for the body-wave amplitude decay
    a = np.sqrt((np.tan(i_s) / np.cos(i_0)) * abs(dt2))
    return a


def calculate_derivatives_of_travel_time(distances, travel_times, distance=None):
    """
    Calculate the first and second derivatives of the travel time.
    This function first obtains the travel time curve for a given phase and event depth.
    It then fits a 6th-degree polynomial to the travel time curve and calculates the first and second analytical
    derivatives of the polynomial.
    Finally it returns the first and second derivatives at distance closest to the event distance.
    # TODO: insert equations p_s = dT/dDelta
    # TODO: be careful when calling this function, works fine for PcP but not other phases; use TauP to get a more
        accurate first derivative/ray parameter/slowness
    :param distances: array, distances at which travel times are calculated in degrees
    :param travel_times: array, travel times in seconds
    :param distance: float, distance from the event to the station in degrees
    :return:
    z: array, value(s) of the fitted travel time(s) at given distance(s)
    dt1s: float or array, first derivative(s) of the fitted travel time(s) at given distance(s)
    dt2s: float or array, second derivative(s) of the fitted travel time(s) at given distance(s)
    """
    # fit a 6th order polynomial to the travel time
    z = np.polyfit(distances, travel_times, 6)

    # calculate the zero, first and second analytical derivatives of fitted travel time
    # zero derivative (fitted travel time)
    dt0s = z[0] * distances ** 6 + z[1] * distances ** 5 + z[2] * distances ** 4 + z[3] * distances ** 3 + z[
        4] * distances ** 2 + z[5] * distances ** 1 + z[6]

    # first derivative
    dt1s = 6 * z[0] * distances ** 5 + 5 * z[1] * distances ** 4 + 4 * z[2] * distances ** 3 + 3 * z[
        3] * distances ** 2 + 2 * z[4] * distances ** 1 + z[5]
    # second derivative
    dt2s = 30 * z[0] * distances ** 4 + 20 * z[1] * distances ** 3 + 12 * z[2] * distances ** 2 + 6 * z[
        3] * distances ** 1 + 2 * z[4]

    # convert to radians
    dt1s = dt1s * 180 / np.pi
    dt2s = dt2s * (180 / np.pi) ** 2

    if distance:
        # get the index of the closest distance
        idx = np.argmin(abs(distance - distances))
        return dt0s[idx], dt1s[idx], dt2s[idx]
    else:
        return dt0s, dt1s, dt2s


def calculate_p_incidence_angle_from_radial_distance(ray_parameter, radial_distance):
    """
    Calculate the incidence angle given the ray parameter at a particular radial distance.
    :param ray_parameter: float, ray parameter p in s/radian
    :param radial_distance: float, radial distance at which P wave velocity and incidence angle is calculated in km
    :return: i: float, incidence angle in radians
    # TODO: insert equations p_s = rsin(i)/v => i = arcsin(p_s*v/r)
    """
    v = calculate_p_velocity_from_radial_distance(radial_distance)
    i = np.arcsin(ray_parameter * v / radial_distance)
    return i


def calculate_p_incidence_angle_from_boundary(ray_parameter, boundary, radial_distance):
    """
    Calculate the incidence angle given the ray parameter at a particular boundary and radial distance.
    :param ray_parameter: float, ray parameter p in s/radian
    :param boundary: string, "mantle-core", "inner-core" or "core-mantle",
    boundary at which incidence angle is calculated,
    see calculate_P_velocity_from_boundary() for clarification on boundary
    :param radial_distance: float, radial distance at which incidence angle is calculated in km;
    :return: i: float, incidence angle in radians
    """
    v = calculate_p_velocity_from_boundary(boundary)
    # TODO: get rid of the radial distance because it depends on the boundary?
    i = np.arcsin(ray_parameter * v / radial_distance)
    return i


def calculate_p_wave_reflection_transmission_coefficients(boundary, incidence_angle):
    """
    Calculate P wave reflection and transmission coefficients at the core-mantle boundary and inner-core boundary
    for a given incidence angle.
    :param boundary: string, "mantle-core", "inner-core" or "core-mantle",
    boundary at which P wave reflection and transmission coefficients are calculated
    :param incidence_angle: float, P wave incidence angle at the boundary in radians
    :return: x: list, reflection and transmission coefficients at the boundary,
    for "mantle-core", x = [R_P, R_SV, T_P],
    for "inner-core", x = [R_P, T_P, T_SV],
    for "core-mantle", x = [R_P, T_P, T_SV]
    """
    i1 = incidence_angle

    if boundary == "mantle-core":
        r = Constants.r_core_mantle
        x = r / Constants.R

        # lower mantle
        rho1 = 7.9565 - 6.4761 * x + 5.5283 * x ** 2 - 3.0807 * x ** 3
        alpha1 = 15.3891 - 5.3181 * x + 5.5242 * x ** 2 - 2.5514 * x ** 3
        beta1 = 6.9254 + 1.4672 * x - 2.0834 * x ** 2 + 0.9783 * x ** 3

        # outer core
        rho2 = 12.5815 - 1.2638 * x - 3.6426 * x ** 2 - 5.5281 * x ** 3
        alpha2 = 11.0487 - 4.0362 * x + 4.8023 * x ** 2 - 13.5732 * x ** 3
        # beta2 = 0

        # calculate elastic constants
        mu1 = (beta1 * 10 ** 5) ** 2 * rho1  # convert to cm
        lambda1 = (alpha1 * 10 ** 5) ** 2 * rho1 - 2 * mu1
        # mu2 = 0
        lambda2 = (alpha2 * 10 ** 5) ** 2 * rho2

        # calculate reflected and transmitted angles according to Snell's law
        j1 = np.arcsin(np.sin(i1) * beta1 / alpha1)  # reflected SV
        i2 = np.arcsin(np.sin(i1) * alpha2 / alpha1)  # transmitted P

        # calculate the wavenumber ratio (as a function of i1)
        r_alpha1 = 1 / np.tan(i1)
        r_beta1 = 1 / np.tan(j1)
        r_alpha2 = 1 / np.tan(i2)

        # solve for reflection coefficients of reflected P and SV waves and
        # transmission coefficient of transmitted P wave
        a = np.array(
            [[-r_alpha1, 1, -r_alpha2],
             [lambda1 * (1 + r_alpha1 ** 2) + 2 * mu1 * r_alpha1 ** 2, -2 * mu1 * r_beta1,
              -lambda2 * (1 + r_alpha2 ** 2)],
             [-2 * r_alpha1, 1 - r_beta1 ** 2, 0]])
        b = np.array(
            [[-r_alpha1],
             [-lambda1 * (1 + r_alpha1 ** 2) - 2 * mu1 * r_alpha1 ** 2],
             [-2 * r_alpha1]])

        x = np.linalg.solve(a, b)
    elif boundary == "inner-core":
        r = Constants.r_inner_core
        x = r / Constants.R

        # outer core
        rho1 = 12.5815 - 1.2638 * x - 3.6426 * x ** 2 - 5.5281 * x ** 3
        alpha1 = 11.0487 - 4.0362 * x + 4.8023 * x ** 2 - 13.5732 * x ** 3
        # beta1 = 0

        # inner core
        rho2 = 13.0885 - 8.8381 * x ** 2
        alpha2 = 11.2622 - 6.3640 * x ** 2
        beta2 = 3.6678 - 4.4475 * x ** 2

        # calculate elastic constants
        # mu1 = 0
        lambda1 = (alpha1 * 10 ** 5) ** 2 * rho1  # convert to cm
        mu2 = (beta2 * 10 ** 5) ** 2 * rho2
        lambda2 = (alpha2 * 10 ** 5) ** 2 * rho2 - 2 * mu2

        # calculate transmitted angles according to Snell's law
        i2 = np.arcsin(np.sin(i1) * alpha2 / alpha1)
        j2 = np.arcsin(np.sin(i1) * beta2 / alpha1)

        # calculate the wavenumber ratio (as a function of i1)
        r_alpha1 = 1 / np.tan(i1)
        r_alpha2 = 1 / np.tan(i2)
        r_beta2 = 1 / np.tan(j2)

        # solve for reflection coefficient of reflected P wave and
        # transmission coefficients of transmitted P and SV waves
        a = np.array(
            [[r_alpha1, r_alpha2, 1],
             [lambda1 * (1 + r_alpha1 ** 2), -lambda2 - (lambda2 + 2 * mu2) * r_alpha2 ** 2, -2 * mu2 * r_beta2],
             [0, 2 * r_alpha2, 1 - r_beta2 ** 2]])

        b = np.array(
            [[r_alpha1],
             [-lambda1 * (1 + r_alpha1 ** 2)],
             [0]])

        x = np.linalg.solve(a, b)
    elif boundary == "core-mantle":
        r = Constants.r_core_mantle
        x = r / Constants.R

        # outer core
        rho1 = 12.5815 - 1.2638 * x - 3.6426 * x ** 2 - 5.5281 * x ** 3
        alpha1 = 11.0487 - 4.0362 * x + 4.8023 * x ** 2 - 13.5732 * x ** 3
        # beta1 = 0

        # lower mantle
        rho2 = 7.9565 - 6.4761 * x + 5.5283 * x ** 2 - 3.0807 * x ** 3
        alpha2 = 15.3891 - 5.3181 * x + 5.5242 * x ** 2 - 2.5514 * x ** 3
        beta2 = 6.9254 + 1.4672 * x - 2.0834 * x ** 2 + 0.9783 * x ** 3

        # calculate elastic constants
        # mu1 = 0
        lambda1 = (alpha1 * 10 ** 5) ** 2 * rho1  # convert to cm
        mu2 = (beta2 * 10 ** 5) ** 2 * rho2
        lambda2 = (alpha2 * 10 ** 5) ** 2 * rho2 - 2 * mu2

        # calculate transmitted angles according to Snell's law
        i2 = np.arcsin(np.sin(i1) * alpha2 / alpha1)
        j2 = np.arcsin(np.sin(i1) * beta2 / alpha1)

        # calculate the wavenumber ratio (as a function of i1)
        r_alpha1 = 1 / np.tan(i1)
        r_alpha2 = 1 / np.tan(i2)
        r_beta2 = 1 / np.tan(j2)

        # solve for reflection coefficient of reflected P wave and
        # transmission coefficients of transmitted P and SV waves
        a = np.array(
            [[r_alpha1, r_alpha2, 1],
             [lambda1 * (1 + r_alpha1 ** 2), -lambda2 - (lambda2 + 2 * mu2) * r_alpha2 ** 2, -2 * mu2 * r_beta2],
             [0, 2 * r_alpha2, 1 - r_beta2 ** 2]])

        b = np.array(
            [[r_alpha1],
             [-lambda1 * (1 + r_alpha1 ** 2)],
             [0]])

        x = np.linalg.solve(a, b)
    else:
        return
    return x


def calculate_p_velocity_from_radial_distance(radial_distance):
    """
    Calculate the P-wave velocity from the PREM model at a given radial distance.
    :param radial_distance: float, radial distance from the center of the Earth in km,
    assuming that earthquakes have a maximum depth at the mantle-core boundary (radial_distance is at least 3480 km)
    :return: v_p: float, P-wave velocity from the PREM model at the radial distance from the center of the Earth in km/s
    """
    r = radial_distance
    # # inner core
    # if 0 <= r < 1221.5:
    #     x = r/R
    #     v_p = 11.2622 - 6.3640*x**2
    #
    # # outer core
    # elif 1221.5 <= r < 3480:
    #     x = r/R
    #     v_p = 11.0487 - 4.0362*x + 4.8023*x**2 - 13.5732*x**3

    # lower mantle
    if 3480 <= r < 3630:
        x = r / Constants.R
        v_p = 15.3891 - 5.3181 * x + 5.5242 * x ** 2 - 2.5514 * x ** 3
    elif 3630 <= r < 5600:
        x = r / Constants.R
        v_p = 24.9520 - 40.4673 * x + 51.4832 * x ** 2 - 26.6419 * x ** 3
    elif 5600 <= r < 5701:
        x = r / Constants.R
        v_p = 29.2766 - 23.6027 * x + 5.5242 * x ** 2 - 2.5514 * x ** 3

    # transition zone
    elif 5701 <= r < 5771:
        x = r / Constants.R
        v_p = 19.0957 - 9.8672 * x
    elif 5771 <= r < 5971:
        x = r / Constants.R
        v_p = 39.7027 - 32.6166 * x
    elif 5971 <= r < 6151:
        x = r / Constants.R
        v_p = 20.3926 - 12.2569 * x

    # low velocity zone
    elif 6151 <= r < 6291:
        x = r / Constants.R
        v_p = 4.1875 + 3.9382 * x

    # LID
    elif 6291 <= r < 6346.6:
        x = r / Constants.R
        v_p = 4.1875 + 3.9382 * x

    # crust
    elif 6346.6 <= r < 6356:
        v_p = 6.8
    elif 6356 <= r < 6368:
        v_p = 5.8

    else:
        return

    return v_p


def calculate_p_velocity_from_boundary(boundary):
    """
    Calculate the P-wave velocity from the PREM model at a given boundary.
    :param boundary: string, boundary at which the P-wave velocity is calculated;
    can be either one of the following three: "mantle-core", "inner-core" or "core-mantle";
    velocity is always calculated in the first layer, indicated by its name;
    for example, for "mantle-core", the function returns P-wave velocity in the mantle above the mantle-core boundary.
    :return: v_p: P-wave velocity from the PREM model at a given boundary in km/s
    """
    if boundary == "mantle-core":
        x = Constants.r_core_mantle / Constants.R
        # P-wave velocity above the mantle-core boundary
        v_p = 15.3891 - 5.3181 * x + 5.5242 * x ** 2 - 2.5514 * x ** 3
    elif boundary == "inner-core":
        x = Constants.r_inner_core / Constants.R
        # P-wave velocity above the inner-core boundary
        v_p = 11.0487 - 4.0362 * x + 4.8023 * x ** 2 - 13.5732 * x ** 3
    elif boundary == "core-mantle":
        x = Constants.r_core_mantle / Constants.R
        # P-wave velocity below the core-mantle boundary
        v_p = 11.0487 - 4.0362 * x + 4.8023 * x ** 2 - 13.5732 * x ** 3
    else:
        return
    return v_p


def rotate_to_ned_coordinates(m_uu, m_ss, m_ee, m_us, m_ue, m_se):
    """
    This function takes in 6 moment tensors in Up, South and East coordinates and
    returns the moment tensor matrix in North, East and Down coordinates.
    :param m_uu: float, Up-Up component of the moment tensor
    :param m_ss: float, South-South component of the moment tensor
    :param m_ee: float, East-East component of the moment tensor
    :param m_us: float, Up-South component of the moment tensor
    :param m_ue: float, Up-East component of the moment tensor
    :param m_se: float, South-East component of the moment tensor
    :return: m, array, 3x3 moment tensor matrix in North, East and Down coordinates
    """
    m = np.array([[m_ss, -m_se, m_us],
                  [-m_se, m_ee, -m_ue],
                  [m_us, -m_ue, m_uu]])

    return m


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


def calculate_pearson_correlation(data1, data2):
    """
    This function calculates the Pearson correlation coefficient between data1 and data2, assuming that trend and mean
    are removed.
    :param data1: array, data1
    :param data2: array, data2
    :return: r: float, the Pearson correlation coefficient.
    """
    top = np.sum(data1 * data2)
    bottom = np.sqrt(np.sum(data1 ** 2)) * np.sqrt(np.sum(data2 ** 2))
    r = top / bottom

    return r


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
    idx_start = idx_pred - int(np.around(init_window_len / 2 * sampling_rate))  # TODO: double check whether np.around() is needed here and same for the rest
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


def calculate_amplitude_and_phase_spectrum(data, sampling_rate):
    """
    Calculate the amplitude and phase spectrum of the Fourier transform of the input data
    :param data: array, trace data
    :param sampling_rate: int, sampling rate of the trace
    :return:
    f: array, discrete frequency points
    A_normalized: array, normalized amplitude spectrum
    Phi: array, phase spectrum
    Refer to Stein & Wysession Chapter 6.4 Discrete time series and transforms
    """
    f_N = sampling_rate / 2  # Nyquist frequency
    N = len(data)  # record length
    # define discrete frequency points
    f = np.linspace(0, f_N, int(N / 2 + 1))

    # apply a cosine taper to avoid side ringing
    cos_taper = cosine_taper(N, p=0.1)
    data_tapered = data * cos_taper
    # calculate the discrete Fourier transform
    F = np.fft.rfft(data_tapered)
    # calculate the amplitude spectrum
    A = np.abs(F)  # for a complex number abs takes the norm
    # normalize the amplitude spectrum between 0 and 1
    A_normalized = (A - np.min(A)) / np.max(A - np.min(A))
    # calculate the phase spectrum
    Phi = np.arctan(np.imag(F) / np.real(F))

    return f, A_normalized, Phi


def Gaussian(x, *p):
    """
    # TODO
    :param x:
    :param p:
    :return:
    """
    w, mu, sigma = p
    return w / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def Gaussian2(x, *p):
    """
    # TODO
    :param x:
    :param p:
    :return:
    """
    w1, mu1, sigma1, w2, mu2, sigma2 = p
    return w1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu1) / sigma1) ** 2) + \
           w2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu2) / sigma2) ** 2)


def calculate_source_duration(Mw):
    # TODO: citation
    M0 = 10 ** ((Mw + 10.73) * 1.5)  # dyn-cm
    tau = 2 * 10 ** (-8) * M0 ** (1 / 3)
    return tau


def plot_correlation_values_on_map(ev_lat, ev_lon, st_lats, st_lons, st_corr_vals, st_snrs,
                                   ax, cb_min, cb_max, cb_label, scale=1.0, order='descending'):
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

    # create color map
    color_map = cm.get_cmap('jet')

    # sort by descending correlation values
    if order == 'descending':
        sort_indices = np.argsort(st_corr_vals)[::-1]
    elif order == 'ascending':
        sort_indices = np.argsort(st_corr_vals)
    else:  # no sorting
        sort_indices = np.arange(st_corr_vals)
    st_lats_plot = st_lats[sort_indices]
    st_lons_plot = st_lons[sort_indices]
    st_corr_vals_plot = st_corr_vals[sort_indices]
    st_snrs_plot = st_snrs[sort_indices]

    ax.set_extent([125, 150, 30, 46])
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # plot distance contours
    CS = ax.contour(LON, LAT, Z, levels=np.arange(np.floor(np.min(Z)), np.ceil(np.max(Z)) + 1, 1)[::2],
                    colors='grey', linewidths=0.5)
    ax.clabel(CS, CS.levels, inline=True)

    ax.scatter(st_lons_plot, st_lats_plot, s=st_snrs_plot * scale, color=color_map(st_corr_vals_plot),
               transform=ccrs.PlateCarree())

    # add color bar
    norm = Normalize(cb_min, cb_max)

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map),
                 ax=ax, orientation='horizontal', label=cb_label)
