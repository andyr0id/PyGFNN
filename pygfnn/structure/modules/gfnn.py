__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure.modules.module import Module
from pygfnn.tools.gfnn import zdot, spontAmp, rk4, zcrk4, limitC

import numpy as np


class GFNNLayer(NeuronLayer):
    """
    Gradient Frequency Neural Network layer
    """

    oscParams = None
    fs = 60.
    dt = 0.01667
    t = 0.
    fspac = None
    f = None
    a = None
    b1 = None
    b2 = None
    e = None
    roote = None
    w = None
    z0 = None
    conns = None
    kSteps = None

    def __init__(self, dim, oscParams=None, freqDist=None, name=None):
        """Create a layer with dim number of units."""
        Module.__init__(self, dim*2, dim, name=name)

        if oscParams is None:
            oscParams = { 'a': 0, 'b1': -1, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }

        if freqDist is None:
            freqDist = {
                'fspac': 'log',
                'min': .5,
                'max': 8
            }

        self.conns = []

        self.setArgs(dim=dim, oscParams=oscParams)

        self.setFreqs(freqDist)

        r0 = np.zeros(dim)
        r = spontAmp(np.real(self.a[0]), np.real(self.b1[0]), np.real(self.b2[0]), self.e)
        r0 = r[-1] + r0
        r0 = r0 + .01 * np.random.randn(np.size(r0))
        phi0 = 2 * np.pi * np.random.randn(np.size(r0))
        z0 = np.complex64(r0 * np.exp(1j * 2 * np.pi * phi0))
        self.z0 = z0
        self.kSteps = np.zeros((4, z0.size), dtype=z0.dtype)
        self.t = np.float32(0.)

    def setFreqs(self, freqDist):
        self.fspac = freqDist['fspac']
        if self.fspac == 'lin':
            f = np.linspace(freqDist['min'], freqDist['max'], self.dim)
        elif self.fspac == 'log':
            f = np.logspace(np.log10(freqDist['min']), np.log10(freqDist['max']), self.dim)
        self.f = np.float32(f)
        self.setOscParams()

    def setOscParams(self, params=None):
        if params is not None:
            self.oscParams = params
        if self.oscParams is not None:
            p = self.oscParams
            if self.fspac == 'lin':
                self.a  = p['a'] + 1j * 2 * np.pi * self.f
                self.b1 = p['b1'] + 1j * p['d1']
                self.b2 = p['b2'] + 1j * p['d2']
                self.w = 1
            elif self.fspac == 'log':
                self.a  = (p['a'] + 1j * 2 * np.pi) * self.f
                self.b1 = (p['b1'] + 1j*p['d1']) * self.f
                self.b2 = (p['b2'] + 1j*p['d2']) * self.f
                self.w = self.f
            self.e = np.complex64(p['e'])
            self.roote = np.sqrt(self.e)

    def getZ(self, prev=1):
        if self.offset == 0:
            return self.z0
        return self.outputbuffer[self.offset-prev]

    def hasLearnableConn(self):
        for c in self.conns:
            if c.learn:
                return True
        return False

    def reset(self):
        """Set all buffers, past and present, to zero."""
        self.offset = 0
        for buffername, l  in self.bufferlist:
            dtype = self._getDtype(buffername)
            buf = getattr(self, buffername)
            buf[:] = np.zeros(l, dtype=dtype)
        self.t = np.float32(0.)

    def _getDtype(self, buffername):
        if buffername == 'outputbuffer' or buffername == 'inputbuffer':
            return np.complex64
        return None


    def _resetBuffers(self, length=1):
        """Reset buffers to a length (in time dimension) of 1."""
        for buffername, dim in self.bufferlist:
            dtype = self._getDtype(buffername)
            setattr(self, buffername, np.zeros((length, dim), dtype=dtype))
        if length==1:
            self.offset = 0

    def _growBuffers(self):
        """Double the size of the modules buffers in its first dimension and
        keep the current values."""
        currentlength = getattr(self, self.bufferlist[0][0]).shape[0]
        # Save the current buffers
        tmp = [getattr(self, n) for n, _ in self.bufferlist]
        self._resetBuffers(currentlength * 2)

        for previous, (buffername, _dim) in zip(tmp, self.bufferlist):
            buffer_ = getattr(self, buffername)
            buffer_[:currentlength] = previous

    def _forwardImplementation(self, inbuf, outbuf):
        extin = inbuf[:self.dim]
        # intin = inbuf[self.dim:self.dim*2]

        if self.hasLearnableConn():
            z, conns = zcrk4(self.t, self.dt, self, extin)
            for i in range(len(self.conns)):
                # self.conns[i].c[:] = limitC(np.reshape(conns[i], self.conns[i].c.shape), self.conns[i].roote)
                self.conns[i].c[:] = conns[i]
        else:
            # simpler integration
            zprev = self.getZ()
            z = rk4(self.t, zprev, self.dt, zdot, self, extin)

        self.t += self.dt
        outbuf[:] = z

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr

if __name__ == "__main__":
    from pybrain.tests import runModuleTestSuite
    import pygfnn.tests.unittests.structure.modules.test_simple_gfnn_network as test
    runModuleTestSuite(test)