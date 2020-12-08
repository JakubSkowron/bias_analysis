#!/bin/env python3

from itertools import islice
import math
import statistics
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

"""
   Bytes Format Units   Label  Explanations
--------------------------------------------------------------------------------
   1- 13 F13.5  d       HJD    Heliocentric Julian Date
  15- 20 F6.3   mag     mag    Apparent magnitude
  22- 26 F5.3   mag   e_mag    Uncertainty in mag
  28- 33 A6     ---     Obs    Observatory
"""

data = []

with open("dbf1.txt", "r") as f:
    lines = islice(f, 19, None)  # skip first 19 lines
    split_line = (line.split() for line in lines)
    data = [(float(HJD), float(mag), float(e_mag), Obs) for HJD, mag, e_mag, Obs in split_line]

# Range HJD > 2457484 and HJD < 2457640 is within one season, and have consistently at least several OGLE
# observations every night
filtered = [(HJD, mag, e_mag) for HJD, mag, e_mag, Obs in data if Obs == "OGLE" and HJD > 2457484 and HJD < 2457640]

# The night of the OGLE-2016-BLG-1928 event
event = np.array([(HJD, mag, e_mag) for HJD, mag, e_mag in filtered if HJD > 2457557.5 and HJD < 2457558])
not_event = np.array([(HJD, mag, e_mag) for HJD, mag, e_mag in filtered if HJD < 2457557.5 or HJD > 2457558])

# few normal nights
few_nights = np.array([(HJD, mag, e_mag) for HJD, mag, e_mag in not_event if HJD > 2457545 and HJD < 2457551])


def plot_light_curve(ax, data, fmt="o", mfc="none", **kwargs):
    ax.set_ylabel("mag")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("HJD")
    HJD = [HJD for HJD, mag, e_mag in data]
    mag = [mag for HJD, mag, e_mag in data]
    e_mag = [e_mag for HJD, mag, e_mag in data]
    ax.errorbar(HJD, mag, e_mag, fmt=fmt, mfc=mfc, **kwargs)
    ax.legend()


fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
ax.set_title("Light curve of OGLE-2016-BLG-1928")
plot_light_curve(ax, not_event, label="not event")
plot_light_curve(ax, event, label="event")

ax = fig.add_subplot(2, 2, 3)
ax.set_title("Few nights")
plot_light_curve(ax, few_nights)

ax = fig.add_subplot(1, 2, 2)
plot_light_curve(ax, event, label="event", color="C1")


def daily_averages():
    values = defaultdict(list)
    for HJD, mag, e_mag in not_event:
        day_begin = math.floor(HJD)
        values[day_begin].append(mag)

    return {HJD: statistics.mean(values) for HJD, values in values.items()}


averages = daily_averages()

modulo = np.array([(HJD % 1, mag - averages[math.floor(HJD)]) for HJD, mag, e_mag in not_event])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("mag")
if not ax.yaxis_inverted():
    ax.invert_yaxis()
ax.set_xlabel("HJD modulo 1")
HJD = [HJD for HJD, mag in modulo]
mag = [mag for HJD, mag in modulo]
fit = np.poly1d(np.polyfit(HJD, mag, 1))
ax.plot(HJD, mag, ',')
ax.plot(HJD, fit(HJD), '-')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("mag")
if not ax.yaxis_inverted():
    ax.invert_yaxis()
ax.set_xlabel("HJD")
HJD, mag = zip(*sorted(averages.items()))
ax.plot(HJD, mag, 'o')


plt.show()
