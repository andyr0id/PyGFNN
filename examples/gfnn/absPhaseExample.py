#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
absPhaseExample
A one layer network driven with a sinusoidal input.
"""

from pygfnn import AbsPhaseIdentityConnection
from pygfnn.tools.plotting.gfnn import *
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit
import matplotlib.pyplot as plt

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
    t_labels = list(np.arange(0, dur, 2.5)) + [dur]
    fig1 = plt.figure()
    sub = 1
    ax = ax2 = None
    for i in xrange(dim):
        if sub == 1:
            ax2 = ax = plt.subplot(dim, 1, sub)
        else:
            ax2 = plt.subplot(dim, 1, sub, sharex=ax, sharey=ax)
        osc = output[:,i]
        plt.plot(t, osc)
        plt.setp(ax2.get_xticklabels(), visible=sub==dim)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.ylabel("%.2f" % n['h'].f[i])
        plt.grid(which='both', axis='x')
        sub += 1
    fig2 = plt.figure()
    sub = 1
    for i in xrange(dim):
        if sub == 1:
            ax2 = ax = plt.subplot(dim+1, 1, sub)
        else:
            ax2 = plt.subplot(dim+1, 1, sub, sharex=ax, sharey=ax)
        osc = output[:,i+dim]
        # plt.plot(t, x*2*np.pi, '0.5')
        plt.plot(t, osc)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.ylabel("%.2f" % n['h'].f[i])
        plt.grid(which='both', axis='x')
        sub += 1
    ax2 = plt.subplot(dim+1, 1, sub, sharex=ax)
    plt.plot(t, x)
    plt.xticks(t_labels, [str(x) for x in t_labels])
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.ylabel("stim")
    plt.grid(which='both', axis='x')
    plt.show()