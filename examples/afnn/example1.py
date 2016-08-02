#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1
A one layer network driven with a sinusoidal input. Several parameter
sets are provided for experimentation with different types of intrinsic
oscillator dynamics.
"""

from pybrain import IdentityConnection
from pygfnn import AbsPhaseIdentityConnection
from pygfnn.tools.plotting.gfnn import *
import pygfnn.tools.shortcuts as gfnn

import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib import rc

def plotPhases(t, net, inp, out, adaptive=True):
    fig1 = plt.figure()
    num = net['h'].dim+2
    if adaptive:
        f0 = net['h'].f0
    else:
        f0 = net['h'].f
    ph0s = []
    for j,x in enumerate(inp):
        if j == 0 or (inp[j-1] < 0 and x >= 0):
            ph0s.append(((t[j], t[j]), (0,1)))
    for i in xrange(net['h'].dim):
        sub = i+1
        if sub == 1:
            ax2 = ax = plt.subplot(num, 1, sub)
            plt.title('Phases over time')
        else:
            ax2 = plt.subplot(num, 1, sub, sharex=ax, sharey=ax)
        for j,x in enumerate(ph0s):
            plt.plot(x[0], x[1], '0.5', linestyle='dashed')
        osc = out[:,i+net['h'].dim]
        plt.plot(t, osc)

        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.ylabel("%.2f" % f0[i],
            rotation='horizontal', horizontalalignment='right')
    ax2 = plt.subplot(num, 1, sub+1, sharex=ax)
    for j,x in enumerate(ph0s):
        plt.plot(x[0], x[1], '0.5', linestyle='dashed')
    plt.plot(t, np.mean(out[:,net['h'].dim:], axis=1))
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.ylabel("mean",
            rotation='horizontal', horizontalalignment='right')
    ax2 = plt.subplot(num, 1, sub+2, sharex=ax)
    for j,x in enumerate(ph0s):
        plt.plot(x[0], x[1], '0.5', linestyle='dashed')
    plt.plot(t, inp)
    # plt.xticks(t_labels, [str(x) for x in t_labels])
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.ylabel("stim",
            rotation='horizontal', horizontalalignment='right')
    # plt.grid(which='both', axis='x')
    return fig1

def plotFrequencies(t, net, fs, adaptive=True):
    gfnn = net['h']
    fig1 = plt.figure()
    plt.title('Adaptive Frequencies')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.plot(t, fs, 'b')
    plt.plot([t[0], t[-1]], [f, f], 'r-.')
    ax1 = plt.gca()
    ax1.set_yscale('log')
    if adaptive:
        fmin, fmax = (gfnn.fr_min/(np.pi*2), gfnn.fr_max/(np.pi*2))
    else:
        freqDist = net['h'].freqDist
        fmin, fmax = (freqDist['min'], freqDist['max'])
    ticks, labels = freqs2labels(fmin, fmax)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(labels)
    plt.ylim((fmin, fmax))
    return fig1

if __name__ == '__main__':
    font = {'family': 'serif', 'size': 14}
    rc('font', **font)

    # Choose network parameters (see pygfnn.tools.shortcuts for more)
    oscParams = gfnn.OSC_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 1.0, 'max': 8 }
    fs = 40.

    # Make network - can have a much lower dimensionality than normal GFNNs
    dim = 8
    n = gfnn.buildGFNN(dim, fs = fs, oscParams = oscParams, freqDist = freqDist,
        outConn = AbsPhaseIdentityConnection, outDim = dim*2, adaptive=True)
    # setting the epsilon for the frequency changes (default=1)
    n['h'].e_f = 1.
    n['h'].e_h = 2.

    # for comparison, o normal GFNN
    n2 = gfnn.buildGFNN(dim, fs = fs, oscParams = oscParams, freqDist = freqDist,
        outConn = AbsPhaseIdentityConnection, outDim = dim*2)

    # Stimulus - 40 seconds of sine wave
    end = 40.
    t = np.arange(0, end, n['h'].dt)
    f = 1 + np.random.random() * 3
    f = 2.467
    # f = 2.
    x = np.sin(2 * np.pi * f * t) * 0.25

    # Run the network
    timer = timeit.default_timer
    start = timer()
    fs = []
    for i in range(len(t)):
        out = n.activate(x[i])
        fs.append(n['h'].f.copy())
    fs = np.array(fs)
    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    # Run the comparison network
    timer = timeit.default_timer
    start = timer()
    for i in range(len(t)):
        out = n2.activate(x[i])
    end = timer()
    print('Elapsed time is %f seconds' % (end - start))

    print('Stimulus frequency (Hz): %.3f' % f)
    print('Starting frequencies:')
    print(n['h'].f0)
    print('Mix/max frequencies:')
    print(n['h'].fr_min / (2*np.pi))
    print(n['h'].fr_max / (2*np.pi))
    print('Ending frequencies:')
    print(n['h'].f)
    print('Ending frequency ratios:')
    print(n['h'].f / f)

    output2 = n2.outputbuffer[:n2.offset]
    fig5 = plotPhases(t, n2, x, output2, adaptive=False)
    output = n.outputbuffer[:n.offset]
    fig2 = plotPhases(t, n, x, output)
    Z = n2['h'].outputbuffer[:n2.offset]
    fig4 = ampx(Z, n2.dt, freqDist['min'], freqDist['max'])
    Z = n['h'].outputbuffer[:n.offset]
    fig1 = ampx(Z, n.dt, freqDist['min'], freqDist['max'])
    fig3 = plotFrequencies(t, n, fs)
    plt.show()