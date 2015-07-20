__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pygfnn.structure.connections.gain import GainConnection

import numpy as np

class MeanFieldConnection(GainConnection):
    """Connection that just averages all the inputs before forwarding."""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.mean(inbuf) * self.gain

    def _backwardImplementation(self, outerr, inerr, inbuf):
        #CHECKME: not setting derivatives
        inerr += outerr / self.gain