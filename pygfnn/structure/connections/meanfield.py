__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.connections.linear import LinearConnection

import numpy as np

class MeanFieldConnection(LinearConnection):
    """Connection that just averages all the inputs before forwarding."""

    def __init__(self, *args, **kwargs):
        LinearConnection.__init__(self, *args, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.mean(inbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        #CHECKME: not setting derivatives -- this means the multiplicative weight is never updated!
        inerr += 0