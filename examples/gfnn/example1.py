#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1
A one layer network driven with a sinusoidal input. Several parameter
sets are provided for experimentation with different types of intrinsic
oscillator dynamics.
"""

from pybrain import IdentityConnection
from pygfnn import AbsIdentityConnection
from pygfnn.tools.plotting.gfnn import *
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Choose network parameters (see pygfnn.tools.shortcuts for more)
    oscParams = gfnn.OSC_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.25, 'max': 4 }
    fs = 40.

    # Make network
    n = gfnn.buildGFNN(200, fs = fs, oscParams = oscParams, freqDist = freqDist,
        outConn = AbsIdentityConnection)

    # Stimulus - 50 seconds of 1Hz sin
    t = np.arange(0, 50, n['h'].dt)
    x = np.sin(2 * np.pi * 1 * t) * 0.25

    # Run the network
    timer = timeit.default_timer
    start = timer()

    for i in range(len(t)):
        out = n.activate(x[i])

    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    print(n.offset, len(n.outputbuffer))
    Z = n['h'].outputbuffer[:n.offset]
    fig1 = ampx(Z, n.dt, freqDist['min'], freqDist['max'])
    fig2 = phasex(Z, n.dt, freqDist['min'], freqDist['max'])
    plt.show()