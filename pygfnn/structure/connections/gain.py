__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.connections.connection import Connection

class GainConnection(Connection):
    """Connection that multiplies by a fixed gain factor."""

    gain = 1.0

    def __init__(self, inmod, outmod, gain=1.0, **kwargs):
        Connection.__init__(self, inmod, outmod, **kwargs)
        self.gain = gain
        self.setArgs(gain = self.gain)