#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
absPhaseExample
A one layer network driven with a sinusoidal input.
"""

from pygfnn import AbsPhaseIdentityConnection
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit

if __name__ == '__main__':
    # Choose network parameters (see pygfnn.tools.shortcuts for more)
    oscParams = gfnn.OSC_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 1.5, 'max': 2.5 }
    fs = 40.
    dur = 10

    # Make network
    dim = 16
    n = gfnn.buildGFNN(dim, fs = fs, oscParams = oscParams, freqDist = freqDist,
        outConn = AbsPhaseIdentityConnection, outDim = dim*2)

    # Stimulus - 50 seconds of 1Hz sin
    t = np.arange(0, dur, n['h'].dt)
    x = np.sin(2 * np.pi * 2 * t) * 0.25

    # Run the network
    timer = timeit.default_timer
    start = timer()

    for i in range(len(t)):
        out = n.activate(x[i])

    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    output = n.outputbuffer[:n.offset] # 0:dim = amp, dim:dim*2 = phase
    for i in xrange(dim):
        osc = output[:fs,i]
        print(osc)
    for i in xrange(dim):
        osc = output[:fs,i+dim]
        print(osc)
