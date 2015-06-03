"""

Build a simple GFNN:

    >>> import numpy as np
    >>> n = buildSimpleGFNN()
    >>> print(n)
    simpleGFNN
       Modules:
        [<LinearLayer 'i'>, <GFNNLayer 'gfnn'>, <LinearLayer 'o'>]
       Connections:
        [<FullConnection 'f1': 'i' -> 'gfnn'>, <RealIdentityConnection 'i1': 'gfnn' -> 'o'>]
       Recurrent Connections:
        []

Set frequencies:

    >>> n['gfnn'].setFreqs({'fspac': 'lin', 'min': .5, 'max': 2})
    >>> n['gfnn'].f
    array([ 0.5  ,  0.875,  1.25 ,  1.625,  2.   ], dtype=float32)
    >>> n['gfnn'].a
    array([ 0. +3.14159274j,  0. +5.49778748j,  0. +7.85398197j,
            0.+10.21017647j,  0.+12.56637096j], dtype=complex64)
    >>> n['gfnn'].b1
    (-1+0j)
    >>> n['gfnn'].b2
    (-1+0j)
    >>> n['gfnn'].e
    (1+0j)

    >>> n['gfnn'].setFreqs({'fspac': 'log', 'min': .5, 'max': 2})
    >>> n['gfnn'].f
    array([ 0.5       ,  0.70710677,  1.        ,  1.41421354,  2.        ], dtype=float32)
    >>> n['gfnn'].a
    array([ 0. +3.14159274j,  0. +4.44288301j,  0. +6.28318548j,
            0. +8.88576603j,  0.+12.56637096j], dtype=complex64)
    >>> n['gfnn'].b1
    array([-0.50000000+0.j, -0.70710677+0.j, -1.00000000+0.j, -1.41421354+0.j,
           -2.00000000+0.j], dtype=complex64)
    >>> n['gfnn'].b2
    array([-0.50000000+0.j, -0.70710677+0.j, -1.00000000+0.j, -1.41421354+0.j,
           -2.00000000+0.j], dtype=complex64)

Run the GFNN:

    >>> n.params[:] = np.ones(len(n.params))
    >>> Y, T = runSimulation(n)
    >>> np.abs(np.mean(Y[-1,:])) < 0.02
    True

"""


__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain import LinearLayer, FullConnection, FullNotSelfConnection, IdentityConnection
from pygfnn import GFNN, GFNNLayer, RealIdentityConnection, RealMeanFieldConnection
from pybrain.tests import runModuleTestSuite

import numpy as np


def buildSimpleGFNN(dim=5, mean=False, conn=False):
    N = GFNN('simpleGFNN')
    i = LinearLayer(1, name = 'i')
    h = GFNNLayer(dim, name = 'gfnn')
    if mean:
        outdim = 1
    else:
        outdim = dim
    o = LinearLayer(outdim, name = 'o')
    N.addOutputModule(o)
    N.addInputModule(i)
    N.addModule(h)
    N.addConnection(FullConnection(i, h, name = 'f1'))
    if conn:
        N.addRecurrentConnection(FullNotSelfConnection(h, h, name = 'r1'))
    if mean:
        N.addConnection(RealMeanFieldConnection(h, o, name = 'm1'))
    else:
        N.addConnection(RealIdentityConnection(h, o, name = 'i1'))
    N.sortModules()
    return N

def runSimulation(n):
    t = np.arange(0, 10, n['gfnn'].dt)
    x = np.sin(2 * np.pi * 2 * t) * 0.1
    x[5*n['gfnn'].fs:-1] *= 0
    for i in range(len(x)):
        o = n.activate(x[i])
    return n.outputbuffer[:n.offset], t

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
    # n = buildSimpleGFNN(200, True)
    # n.params[:] = np.ones(len(n.params))
    # Y, T = runSimulation(n)
    # print(len(T), len(Y))

    # import matplotlib.pyplot as plt
    # fig1 = plt.figure()
    # plt.plot(T, Y)
    # plt.show()
    # print(Y)

