__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import IdentityConnection
from pygfnn.structure.connections.gain import GainConnection

import numpy as np

TWO_PI = np.pi*2
R_TWO_PI = 1/TWO_PI

class PhaseIdentityConnection(GainConnection, IdentityConnection):

    def __init__(self, inmod, outmod, gain=R_TWO_PI, **kwargs):
        GainConnection.__init__(self, inmod, outmod, gain=gain, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += ((np.angle(inbuf) + TWO_PI) % TWO_PI) * self.gain