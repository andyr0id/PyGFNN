#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1P

A one layer network with fixed internal connections
"""

from pygfnn.tools.plotting.gfnn import *
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':
    # Network parameters
    oscParams = { 'a': 1, 'b1': -1, 'b2': -1000, 'd1': 0, 'd2': 0, 'e': 1 } # Limit cycle
    learnParams = gfnn.NOLEARN_ALLFREQ
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 8 }

    # Make network
    n = gfnn.buildGFNN(196, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams)
    n.recurrentConns[0].c0[:] = gfnn.getInitC(n, n, [(1,1), (1,2), (1,3), (1,4), (1,6), (1,8), (2,3), (3,4), (3,8)], thresh=0.01)
    n.reset()

    # First plots, showing initial connection state
    ampFig1, phaseFig1 = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])

    # Stimulus - 50 seconds of 1Hz sin
    t = np.arange(0, 50, n['h'].dt)
    x = np.sin(2 * np.pi * 1 * t) * 0.1

    # Run the network
    timer = timeit.default_timer
    start = timer()

    for i in range(len(t)):
        out = n.activate(x[i])

    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    if learnParams is not None:
        # Second plots, showing final connection state
        ampFig2, phaseFig2 = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])

    Z = n['h'].outputbuffer[:n.offset]
    fig1 = ampx(Z, n.dt, freqDist['min'], freqDist['max'])
    fig2 = phasex(Z, n.dt, freqDist['min'], freqDist['max'])
    plt.show()
