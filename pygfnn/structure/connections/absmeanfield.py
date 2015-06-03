__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pygfnn.structure.connections.meanfield import MeanFieldConnection

import numpy as np

class AbsMeanFieldConnection(MeanFieldConnection):
    """Connection that just averages all the inputs before forwarding."""

    def __init__(self, *args, **kwargs):
        MeanFieldConnection.__init__(self, *args, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.mean(np.abs(inbuf))