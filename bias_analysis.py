#!/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


# Utility functions ##############
# Low pass filter for data with constant sampling frequency
def low_pass_filter(data, sampling_freq, cut_freq, order=5):
    order = 4
    sos = signal.butter(order, cut_freq, 'lowpass', fs=sampling_freq, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered


# Return index of element with value nearest to given
# arr should be sorted
def find_nearest(arr, value):
    i = np.searchsorted(arr, value, side='left')
    if i == 0:
        return i
    if i == len(arr):
        return i-1
    if value - arr[i-1] < arr[i] - value:
        return i-1
    return i


# Classes #################
class DataGenerator:
    "Helper class to create phoney observation data."
    def generate_observations(start_HJD, duration, mean_delay=0.02, mag_mean=17, mag_deviation=0.2):
        '''
        Generate phoney observations.
        Returns structured numpy array of observations, each observation is ('HJD', 'mag', 'e_mag')
        'duration' and 'mean_delay' in days
        '''
        night_time_begin = 0.4
        night_time_end = 0.9
        day_time_duration = 1 - (night_time_end - night_time_begin)

        obs_times = []
        # generate moments of observations, negative exponential distribution
        curr_HJD = start_HJD
        while True:
            # random exponential distribution
            curr_HJD += random.expovariate(1/mean_delay)
            # skip daytime
            while (curr_HJD % 1 < night_time_begin) or (curr_HJD % 1 > night_time_end):
                curr_HJD += day_time_duration + random.expovariate(1/mean_delay)
            if curr_HJD > start_HJD + duration:
                break
            obs_times.append(curr_HJD)

        size = len(obs_times)
        data = np.empty(size, ([("HJD", np.float), ("mag", np.float), ("e_mag", np.float)]))
        data["HJD"] = obs_times
        data["mag"] = np.random.normal(mag_mean, mag_deviation, size)
        data["e_mag"] = np.random.uniform(mag_deviation*0.5, mag_deviation*1.5, size)

        return data

    def generate_bias(start_HJD, duration, step, mag_scale=0.1, low_pass_period=0.5):
        '''
        Make slowly changing bias. 'mag_scale' is a maximum deviation, 'low_pass_period' is
        time scale for low pass filter (critical frequency is 1/low_pass_period)
        Returns numpy structured array. Each element is ("HJD", "mag"). Evenly spaced with 'step' separation
        'duration', 'step', 'low_pass_period' in days
        'mag_scale' in magnitude
        '''
        size = int(duration / step)
        data = np.zeros(size, [("HJD", np.float), ("mag", np.float)])
        # prepare array
        data["HJD"] = np.linspace(start_HJD, start_HJD + duration, size, endpoint=False)

        # generate random noise
        random = np.random.normal(0, 1, size)

        # filter out high frequencies
        sampling_freq = 1/step  # in 1/days
        cut_freq = 1/low_pass_period  # in 1/days
        filtered_noise = low_pass_filter(random, sampling_freq, cut_freq)
        # normalize to range [-1,1]
        filtered_noise /= np.max(np.abs(filtered_noise))

        # fill prepared array
        data["mag"] = filtered_noise * mag_scale

        return data

    def add_bias(data, bias):
        '''
        Add bias to data. Finds nearest HJD in bias and adds its mag to data
        Returns as new array shaped same as 'data'
        data is numpy structured array ('HJD', 'mag', 'e_mag')
        bias is numpy structured array ('HJD', 'mag')
        '''
        biased_data = np.array(data)
        for tup in biased_data:
            nearest_i = find_nearest(bias['HJD'], tup[0])
            tup[1] += bias[nearest_i][1]
        return biased_data


class ReduceBias:
    """
    Helper class for reducing bias in measured data
    """
    def find_bias(data, step=0.005, avg_period=0.5):
        '''
        Estimates bias by calculating moving mean.
        Estimates bias uncertanity - probably underestimated, use it rather as a weight for now.
        Assumes data['mag'] is a measurement of a constant physical value with added some low-frequency unknown bias
        Bias is assumed to be average magnitude in range (-avg_period/2, +avg_period/2) days.

        'avg_period' is in days
        'data' is numpy structured array ('HJD', 'mag', 'e_mag'), should be sorted by HJD.
        Returns structured array ('HJD', 'mag', 'e_mag'). Starting just before first data HJD and
        ending just after last data HJD, evenly spaced with step.
        '''
        # prepare array
        new_HJD = np.arange(data['HJD'][0] - avg_period/2, data['HJD'][-1] + avg_period/2, step)
        moving_average = np.zeros(len(new_HJD), dtype=[("HJD", np.float), ("mag", np.float), ("e_mag", np.float)])
        moving_average['HJD'] = new_HJD

        # calculate moving average
        for tup in moving_average:
            HJD = tup[0]
            average_window = data[abs(data["HJD"] - HJD) < avg_period/2]
            if len(average_window) < 2:
                tup[1] = math.nan
                tup[2] = math.nan
                continue

            # weight of each data point is sum of distances to nearest points in window
            # i.e we get a time average as a result
            distances = average_window["HJD"][1:] - average_window["HJD"][:-1]
            weights = np.concatenate(([0], distances)) + np.concatenate((distances, [0]))
            # normalize weights
            weights = weights/np.sum(weights)
            # TODO: add some windowing function here

            # weighted average
            mag = np.sum(average_window["mag"] * weights)
            # uncertanity of a weighted mean
            e_mag = np.sqrt(np.sum(np.square(average_window["e_mag"] * weights)))

            tup[1] = mag
            tup[2] = e_mag

        return moving_average

    def subtract_nearest_bias(data, bias):
        '''
        Find nearest point in 'bias', subtract it from magnitude. Calculate uncertainity.
        Does not modify data, returns an unbiased copy.
        '''
        new_data = np.array(data)
        for tup in new_data:
            HJD = tup[0]
            nearest_i = find_nearest(bias['HJD'], HJD)
            tup[1] -= bias["mag"][nearest_i]
            # uncertainity of a sum
            tup[2] = math.sqrt(tup[2]**2 + bias['e_mag'][nearest_i]**2)

        return new_data


# Script #############

if __name__ == "__main__":
    # plt.ion()  #interactive mode ON

    # Generate data
    start = 245991  # HJD
    duration = 5  # days
    mag_mean = 17
    mag_deviation = 0.2
    data = DataGenerator.generate_observations(start, duration, mag_mean=mag_mean, mag_deviation=mag_deviation)
    bias = DataGenerator.generate_bias(start, duration, step=0.001, mag_scale=mag_deviation*3, low_pass_period=0.5)
    biased_data = DataGenerator.add_bias(data, bias)

    # Reduce bias
    found_bias = ReduceBias.find_bias(biased_data, step=0.005, avg_period=0.25)
    reduced_data = ReduceBias.subtract_nearest_bias(biased_data, found_bias)
    data_average = np.average(data['mag'], weights=data['e_mag'])
    reduced_data['mag'] += data_average

    # Plot
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)

    ax1.errorbar(data['HJD'], data['mag'], data['e_mag'], fmt='o', mfc=None)
    ax1.set_title("Original data")

    ax2.errorbar(biased_data['HJD'], biased_data['mag'], biased_data['e_mag'], fmt='o', mfc=None, label="biased data")
    ax2.plot(bias['HJD'], bias['mag'] + mag_mean, label="added bias")
    ax2.set_title("Original data with added bias")
    ax2.legend()

    ax3.errorbar(biased_data['HJD'], biased_data['mag'], biased_data['e_mag'], fmt='o', mfc=None, label="biased data")
    ax3.plot(found_bias["HJD"], found_bias["mag"], label="deduced bias")
    ax3.legend()

    ax4.errorbar(found_bias["HJD"], found_bias["mag"], found_bias['e_mag'], fmt=',', label="deduced bias")
    ax4.plot(bias["HJD"], bias["mag"] + mag_mean, label="added bias")
    ax4.legend()

    ax5.errorbar(reduced_data["HJD"], reduced_data["mag"], reduced_data['e_mag'], fmt='o')
    ax5.set_xlabel("HJD")
    ax5.set_title("Reduced data")

    plt.show()
