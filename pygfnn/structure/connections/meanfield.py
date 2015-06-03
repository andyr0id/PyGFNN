__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.connections.connection import Connection

import numpy as np

class MeanFieldConnection(Connection):
    """Connection that just averages all the inputs before forwarding."""

    def __init__(self, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += np.mean(inbuf)