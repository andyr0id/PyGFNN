__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import IdentityConnection

import numpy as np

class RealIdentityConnection(IdentityConnection):

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.real(inbuf)