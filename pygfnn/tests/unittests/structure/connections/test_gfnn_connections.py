"""

Build a simple GFNN:

    >>> import numpy as np

Build a meanfield GFNN:

    >>> n = buildSimpleGFNN(mean=True)
    >>> print(n)
    simpleGFNN
       Modules:
        [<LinearLayer 'i'>, <GFNNLayer 'gfnn'>, <LinearLayer 'o'>]
       Connections:
        [<GFNNExtConnection 'f1': 'i' -> 'gfnn'>, <RealMeanFieldConnection 'm1': 'gfnn' -> 'o'>]
       Recurrent Connections:
        []

Run the GFNN:

    >>> n.params[:] = np.ones(len(n.params))
    >>> Y, T = runSimulation(n)
    >>> np.abs(Y[-1,0]) < 0.02
    True

Build a connected GFNN:

    >>> n = buildSimpleGFNN(mean=True, conn=True)
    >>> print(n)
    simpleGFNN
       Modules:
        [<LinearLayer 'i'>, <GFNNLayer 'gfnn'>, <LinearLayer 'o'>]
       Connections:
        [<GFNNExtConnection 'f1': 'i' -> 'gfnn'>, <RealMeanFieldConnection 'm1': 'gfnn' -> 'o'>]
       Recurrent Connections:
        [<GFNNIntConnection 'r1': 'gfnn' -> 'gfnn'>]

Run the GFNN:

    >>> n.params[:] = np.ones(len(n.params))
    >>> Y, T = runSimulation(n)
    >>> np.abs(Y[-1,0]) < 0.02
    True

"""

__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain import LinearLayer, FullConnection, FullNotSelfConnection, IdentityConnection
from pygfnn import GFNN, GFNNLayer, GFNNExtConnection, GFNNIntConnection, RealIdentityConnection, RealMeanFieldConnection
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
    N.addConnection(GFNNExtConnection(i, h, name = 'f1'))
    if conn:
        N.addRecurrentConnection(GFNNIntConnection(h, h, name = 'r1'))
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

