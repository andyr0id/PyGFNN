__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.modules.module import Module
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pygfnn.structure.modules.gfnn import GFNNLayer
from pygfnn.structure.connections.gfnn import GFNNIntConnection

import numpy as np

class GFNN(RecurrentNetwork):

    fs = None
    dt = None

    def __init__(self, name=None, fs=40., *args, **kwargs):
        kwargs['name'] = name
        RecurrentNetwork.__init__(self, *args, **kwargs)
        self.setArgs(fs=np.float32(fs), dt=np.float32(1./fs))

    def addModule(self, m):
        super(RecurrentNetwork, self).addModule(m)
        if isinstance(m, GFNNLayer):
            m.fs = self.fs
            m.dt = self.dt

    def reset(self, randomizeOscs=True, randomizeConnPhase=True):
        """Reset all component modules and the network."""
        Module.reset(self)
        for m in self.modules:
            if isinstance(m, GFNNLayer):
                m.reset(randomizeOscs)
            else:
                m.reset()
        for c in self.recurrentConns:
            if isinstance(c, GFNNIntConnection):
                c.reset(randomizeConnPhase)