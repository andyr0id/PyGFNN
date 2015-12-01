__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import IdentityConnection
from pygfnn.structure.connections.gain import GainConnection

import numpy as np

TWO_PI = np.pi*2
R_TWO_PI = 1/TWO_PI

class AbsPhaseIdentityConnection(GainConnection, IdentityConnection):

    phasegain = R_TWO_PI

    def __init__(self, inmod, outmod, gain=1.0, phasegain=R_TWO_PI, **kwargs):
        if outmod.dim < inmod.dim*2:
            raise ValueError('outmod.dim neds to be >= inmod.dim*2')
        GainConnection.__init__(self, inmod, outmod, gain=gain, **kwargs)
        self.phasegain = phasegain
        self.setArgs(phasegain = self.phasegain)

    def _forwardImplementation(self, inbuf, outbuf):
        length = len(inbuf)
        outbuf[:length] += np.abs(inbuf) * self.gain
        outbuf[length:length*2] += ((np.angle(inbuf) + TWO_PI) % TWO_PI) * self.phasegain