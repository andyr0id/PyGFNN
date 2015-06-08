#!/usr/bin/env python
__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

"""
example1P

A one layer network with plastic internal connections (and no input)
"""

import pygfnn.tools.shortcuts as gfnn

from pybrain.tools.customxml import NetworkWriter
from pygfnn.tools.customxml import NetworkReader # Use the pygfnn NetworkReader to read the network

if __name__ == '__main__':
    # Network parameters
    oscParams = { 'a': 1, 'b1': -1, 'b2': -1000, 'd1': 0, 'd2': 0, 'e': 1 } # Limit cycle
    learnParams = gfnn.LEARN_SUPER_CRITICAL
    freqDist = { 'fspac': 'log', 'min': 0.5, 'max': 2 }

    # Make network
    n1 = gfnn.buildGFNN(5, oscParams = oscParams, freqDist = freqDist,
        learnParams = learnParams)
    print(n1)

    NetworkWriter.writeToFile(n1, 'saveExample.xml')

    n2 = NetworkReader.readFrom('saveExample.xml')
    print(n2)

    print(n1['h'].f)
    print(n2['h'].f)

