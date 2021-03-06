#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1P

A one layer network with plastic internal connections (and no input)
"""

import pygfnn.tools.shortcuts as gfnn

from pygfnn.tools.customxml import NetworkWriter, NetworkReader # Use the pygfnn NetworkReader to read the network
import numpy as np

if __name__ == '__main__':
    # Network parameters
    oscParams = { 'a': 1, 'b1': -1, 'b2': -1000, 'd1': 0, 'd2': 0, 'e': 1 } # Limit cycle
    learnParams = gfnn.LEARN_SUPER_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 2 }
    dim = 5
    size = dim * dim
    c0 = np.zeros(size) + .01 * np.random.randn(size)
    c0 = np.complex64(np.reshape(c0, (dim, dim)))
    theta0 = np.exp(1j * 2 * np.pi * np.random.randn(size))
    c0 *= np.reshape(theta0, (dim, dim))
    # c0 = None

    # Make network
    n1 = gfnn.buildGFNN(dim, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams, c0 = c0)
    print(n1)

    NetworkWriter.writeToFile(n1, 'saveExample.xml')

    n2 = NetworkReader.readFrom('saveExample.xml')
    print(n2)

    print(np.array_equal(n1['h'].f, n2['h'].f))
    print(np.array_equal(n1.recurrentConns[0].c0, n2.recurrentConns[0].c0))

