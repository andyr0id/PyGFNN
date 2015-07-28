#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1P

A one layer network with plastic internal connections (and no input)
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
    learnParams = gfnn.LEARN_SUPER_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 2 }

    # Make network
    n = gfnn.buildGFNN(200, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams)

    # Stimulus - x seconds of silence
    t = np.arange(0, 10, n['h'].dt)

    # Space for saving data
    if learnParams is not None:
        w, h = np.shape(n.recurrentConns[0].c)
        c3 = np.zeros((w, h, len(t)), np.complex64)

    # Run the network
    timer = timeit.default_timer
    start = timer()

    for i in range(len(t)):
        out = n.activate(0.)
        if learnParams is not None:
            c3[:, :, i] = n.recurrentConns[0].c

    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    if learnParams is not None:
        sio.savemat('example1P-C3.mat', { 'C3': c3 })
        ampFig, phaseFig = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])
        plt.show()

        # resetting the network resets the connection matrix to c0, and randomises z0
        n.reset()
        ampFig, phaseFig = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])
        plt.show()
