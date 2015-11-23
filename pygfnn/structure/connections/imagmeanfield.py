__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pygfnn.structure.connections.meanfield import MeanFieldConnection

import numpy as np

class ImagMeanFieldConnection(MeanFieldConnection):
    """Connection that just averages all the inputs before forwarding."""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.mean(np.imag(inbuf)) * self.gain