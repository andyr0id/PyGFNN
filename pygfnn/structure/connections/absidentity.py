__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import IdentityConnection
from pygfnn.structure.connections.gain import GainConnection

import numpy as np

class AbsIdentityConnection(GainConnection, IdentityConnection):

    def __init__(self, inmod, outmod, gain=1.0, **kwargs):
        GainConnection.__init__(self, inmod, outmod, gain=gain, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.abs(inbuf) * self.gain