#!/bin/env python3

import bias_analysis

from itertools import islice
import numpy as np
import matplotlib.pyplot as plt

import warnings
# make all warnings exceptions. numpy by default issues warning on floating point exceptions
warnings.simplefilter('error')


def load_ogle_data(file_path):
    """
    Bytes Format Units   Label  Explanations
    --------------------------------------------------------------------------------
    1- 13 F13.5  d       HJD    Heliocentric Julian Date
    15- 20 F6.3   mag     mag    Apparent magnitude
    22- 26 F5.3   mag   e_mag    Uncertainty in mag
    28- 33 A6     ---     Obs    Observatory

    Skip first 19 rows
    """

    with open("dbf1.txt", "r") as f:
        lines = islice(f, 19, None)  # skip first 19 lines
        split_lines = (line.split() for line in lines)
        ogle_tuples = [x for x in split_lines if x[3] == "OGLE"]
        data = np.empty(len(ogle_tuples), ([("HJD", np.float), ("mag", np.float), ("e_mag", np.float)]))
        data['HJD'] = [float(x[0]) for x in ogle_tuples]
        data['mag'] = [float(x[1]) for x in ogle_tuples]
        data['e_mag'] = [float(x[2]) for x in ogle_tuples]
        return np.sort(data, order='HJD')


whole_data = load_ogle_data('dbf1.txt')
# Range 2457484 < HJD < 2457640 is within one season, and have consistently at least several OGLE
# observations every night
data_sample = whole_data[(whole_data['HJD'] > 2457484) & (whole_data['HJD'] < 2457640)]

# The night of the OGLE-2016-BLG-1928 event is HJD 2457557
event = data_sample[(data_sample['HJD'] > 2457557.725) & (data_sample['HJD'] < 2457558)]
not_event = data_sample[(data_sample['HJD'] < 2457557.725) | (data_sample['HJD'] > 2457558)]

# few normal nights 2457545) < HJD < 2457557

# Reduce bias
bias_time_scale = 0.3  # days
data_average = np.average(not_event['mag'], weights=not_event['e_mag'])
found_bias = bias_analysis.ReduceBias.find_bias(not_event, step=0.005, avg_period=bias_time_scale)
reduced_data = bias_analysis.ReduceBias.subtract_nearest_bias(not_event, found_bias)
reduced_data['mag'] += data_average


# Plot reduced data
def make_light_curve_axes(ax):
    ax.set_ylabel("mag")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("HJD")


def plot_light_curve(ax, data, fmt='o', **kwargs):
    ax.errorbar(data['HJD'], data['mag'], data['e_mag'], fmt=fmt, **kwargs)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

make_light_curve_axes(ax1)
plot_light_curve(ax1, data_sample)
ax1.axhline(data_average, label="mean")
plot_light_curve(ax1, found_bias, fmt=',', label="deduced bias")
ax1.legend()
ax1.set_title("OGLE-2016-BLG-1928")
# ax1.set_xlim(2457545.5,2457556.5)

make_light_curve_axes(ax2)
plot_light_curve(ax2, reduced_data)
ax2.set_title("Reduced data")
ax2.axhline(data_average, label="mean")

plt.show()
