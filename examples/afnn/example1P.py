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

def plotPhases(t, net, inp, out):
    fig1 = plt.figure()
    num = net['h'].dim+1
    for i in xrange(net['h'].dim):
        sub = i+1
        if sub == 1:
            ax2 = ax = plt.subplot(num, 1, sub)
            plt.title('Phases over time')
        else:
            ax2 = plt.subplot(num, 1, sub, sharex=ax, sharey=ax)
        for j,x in enumerate(inp):
            if j == 0 or (inp[j-1] < 0 and x >= 0):
                plt.plot((t[j], t[j]), (0,1), '0.5', linestyle='dashed')
        osc = out[:,i+net['h'].dim]
        plt.plot(t, osc)

        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.ylabel("%.2f" % net['h'].f0[i],
            rotation='horizontal', horizontalalignment='right')
    ax2 = plt.subplot(num, 1, sub+1, sharex=ax)
    for i,x in enumerate(inp):
        if i == 0 or (inp[i-1] < 0 and x >= 0):
            plt.plot((t[i], t[i]), (-1,1), '0.5', linestyle='dashed')
    plt.plot(t, inp)
    # plt.xticks(t_labels, [str(x) for x in t_labels])
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.ylabel("stim",
            rotation='horizontal', horizontalalignment='right')
    plt.ylabel("stim",
            rotation='horizontal', horizontalalignment='right')
    # plt.grid(which='both', axis='x')
    return fig1

def plotFrequencies(t, net, fs):
    gfnn = net['h']
    fig1 = plt.figure()
    plt.title('Frequencies over time')
    plt.plot(t, fs)
    plt.plot([t[0], t[-1]], [f, f], '0.5', linestyle='dashed')
    ax1 = plt.gca()
    ax1.set_yscale('log')
    fmin, fmax = (gfnn.fr_min[0]/(np.pi*2), gfnn.fr_max[-1]/(np.pi*2))
    ticks, labels = freqs2labels(fmin, fmax)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(labels)
    plt.ylim((fmin, fmax))
    return fig1

if __name__ == '__main__':
    # Choose network parameters (see pygfnn.tools.shortcuts for more)
    oscParams = gfnn.OSC_CRITICAL
    learnParams = gfnn.LEARN_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 8 }
    fs = 40.

    # Make network - can have a much lower dimentionaliy than normal GFNNs
    dim = 16
    n = gfnn.buildGFNN(dim, fs = fs, oscParams = oscParams, freqDist = freqDist,
        outConn = AbsPhaseIdentityConnection, outDim = dim*2, adaptive=True,
        learnParams = learnParams)
    # setting the eplilon for the frequency changes (deflaut=1)
    n['h'].e_f = .25

    # Stimulus - 40 seconds of cosine wave
    t = np.arange(0, 40, n['h'].dt)
    f = 1 + np.random.random() * 3
    f = 2
    x = np.cos(2 * np.pi * f * t) * 0.1

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
    print('Stimulus frequency (Hz): %.3f' % f)
    print('Starting frequencies:')
    print(n['h'].f0)
    print('Ending frequencies:')
    print(n['h'].f)
    print('Ending frequency ratios:')
    print(n['h'].f / f)

    output = n.outputbuffer[:n.offset]
    Z = n['h'].outputbuffer[:n.offset]
    ampFig, phaseFig = plotConns(n.recurrentConns[0].c, freqDist['min'], freqDist['max'])
    fig3 = ampx(Z, n.dt, freqDist['min'], freqDist['max'])
    fig1 = plotPhases(t, n, x, output)
    fig2 = plotFrequencies(t, n, fs)
    plt.show()