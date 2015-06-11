#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import LSTMLayer, LinearLayer, FullConnection, BiasUnit
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised import RPropMinusTrainer
from pygfnn import GFNN, GFNNLayer, GFNNExtConnection, GFNNIntConnection, RealMeanFieldConnection

import pygfnn.tools.shortcuts as gfnn
import numpy as np
import matplotlib.pyplot as plt
import timeit

"""
example1
A two layer GFNN-LSTM network driven with a series of events. One input, one output.
"""

def buildGFNNLSTM(gfnnDim, lstmDim, **options):
    opt = {
        'oscParams': None,
        'freqDist': None,
        'fs': 40.,
        'learnParams': None,
        'c0': None,
        'interConn': RealMeanFieldConnection,
    }
    for key in options:
        if key not in opt.keys():
            raise NetworkError('buildGFNN unknown option: %s' % key)
        opt[key] = options[key]

    # make GFNN-LSTM - important to use the GFNN class
    n = GFNN('gfnn-lstm', fs = opt['fs'])

    # input layer
    i = LinearLayer(1, name = 'i')
    n.addInputModule(i)

    # output
    o = LinearLayer(1, name = 'o')
    n.addOutputModule(o)

    # bias
    b = BiasUnit(name='b')
    n.addModule(b)

    # hidden
    gfnn = GFNNLayer(gfnnDim, oscParams = opt['oscParams'],
        freqDist = opt['freqDist'], name = 'gfnn')
    n.addModule(gfnn)

    if issubclass(opt['interConn'], RealMeanFieldConnection):
        interDim = 1
    else:
        interDim = gfnnDim
    print(interDim)
    inter = LinearLayer(interDim, name = 'inter')
    n.addModule(inter)

    lstm = LSTMLayer(lstmDim, peepholes = True, name = 'lstm')
    n.addModule(lstm)

    # input -> gfnn
    n.addConnection(GFNNExtConnection(i, gfnn, name = 'igc'))

    # gfnn -> gfnn
    if opt['learnParams'] is not None:
        n.addRecurrentConnection(GFNNIntConnection(gfnn, gfnn,
            learnParams = opt['learnParams'], c0 = opt['c0'],
            name = 'grc'))

    # gfnn -> inter
    n.addConnection(opt['interConn'](gfnn, inter, name = 'gxc'))

    # inter -> lstm
    n.addConnection(FullConnection(inter, lstm, name = 'xlc'))

    # lstm -> lstm
    n.addRecurrentConnection(FullConnection(lstm, lstm, name = 'lrc'))

    # bias -> lstm
    n.addConnection(FullConnection(b, lstm, name = 'blc'))

    # lstm -> output
    n.addConnection(FullConnection(lstm, o, name = 'loc'))

    # bias -> output
    n.addConnection(FullConnection(b, lstm))

    n.sortModules()
    return n

def buildDS(n, num, dur):
    ds = SequentialDataSet(1,1)
    dt = n['gfnn'].dt
    t = np.arange(0, dur, dt)
    length = len(t)
    for i in range(num):
        x = np.zeros(length)
        # tempo between 1 and 2 bps
        bps = 1 + np.random.random()
        p = 1./bps
        lastPulse = np.random.random() * p
        for j in range(length):
            if t[j] > lastPulse and t[j] >= lastPulse + p:
                x[j] = 0.25
                lastPulse = t[j]
            else:
                if j > 0:
                    if x[j-1] < 1e-5:
                        x[j] = 0
                    else:
                        x[j] = x[j-1] * 0.5
        # try to predict the next sample
        target = np.roll(x, -1)

        ds.newSequence()
        for j in range(length):
            ds.addSample(x[j], target[j])
    return ds


if __name__ == '__main__':
    # Choose network parameters (see pygfnn.tools.shortcuts for more)
    oscParams = gfnn.OSC_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 8 }
    gfnnLearnParams = None
    gfnnDim = 50
    lstmDim = 5

    # Build network
    n = buildGFNNLSTM(gfnnDim, lstmDim,
        oscParams = oscParams, freqDist = freqDist, learnParams = gfnnLearnParams)

    # Create a dataset - 10, 40s pulses at various tempos
    ds = buildDS(n, 10, 40)

    # Train (hopfully you'll see errors go down!)
    tr = RPropMinusTrainer(n, dataset=ds, verbose=True)

    timer = timeit.default_timer
    start = timer()
    err = tr.trainEpochs(5)
    end = timer()
    print('Elapsed time is %f seconds' % (end - start))
