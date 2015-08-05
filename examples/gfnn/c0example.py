#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
c0example

A one layer network with initial connections
"""

from pygfnn.tools.plotting.gfnn import *
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':
    timer = timeit.default_timer

    # Network parameters
    oscParams = { 'a': 1, 'b1': -1, 'b2': -1000, 'd1': 0, 'd2': 0, 'e': 1 } # Limit cycle
    learnParams = gfnn.LEARN_SUPER_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 2 }

    # Make network
    initN = gfnn.buildGFNN(128, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams)

    # Stimulus - x seconds of silence
    t = np.arange(0, 10, initN['h'].dt)

    # Run the network
    start = timer()
    for i in range(len(t)):
        out = initN.activate(0.)
    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    c0 = initN.recurrentConns[0].c.copy()
    # First plots, showing the c0 learned state
    ampFig1, phaseFig1 = plotConns(c0, freqDist['min'], freqDist['max'])

    # New network parameters
    oscParams = gfnn.OSC_CRITICAL
    learnParams = gfnn.LEARN_CRITICAL

    # Make network with c0
    n = gfnn.buildGFNN(128, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams, c0 = c0)

    # Stimulus - 10 seconds of 1Hz sin
    t = np.arange(0, 10, n['h'].dt)
    x = np.sin(2 * np.pi * 1 * t) * 0.05

    # Run the 2nd network
    start = timer()

    for i in range(len(t)):
        out = n.activate(x[i])

    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    Z = n.outputbuffer[:n.offset]
    ampxFig1 = ampx(Z, n.dt, freqDist['min'], freqDist['max'])

    # First plots, showing the final connection state
    ampFig2, phaseFig2 = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])

    # resetting the network resets the connection matrix to c0 (with random phase), and randomises z0
    n.reset()
    ampFig3, phaseFig3 = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])
    plt.show()
