__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import IdentityConnection
from pygfnn.structure.connections.gain import GainConnection

import numpy as np

class AbsPhaseIdentityConnection(GainConnection, IdentityConnection):

    def __init__(self, inmod, outmod, gain=1.0, **kwargs):
    	if outmod.dim < inmod.dim*2:
    		raise ValueError('outmod.dim neds to be >= inmod.dim*2')
        GainConnection.__init__(self, inmod, outmod, gain=gain, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
    	length = len(inbuf)
        outbuf[:length] += np.abs(inbuf) * self.gain
        outbuf[length:length*2] += np.angle(inbuf) * self.gain