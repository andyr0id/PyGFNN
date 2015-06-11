__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import reshape, dot, outer, eye
from pybrain.structure.connections import FullConnection, FullNotSelfConnection
from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer

from pygfnn.tools.gfnn import spontAmp
from pygfnn.structure.modules.gfnn import GFNNLayer

import numpy as np

class GFNNExtConnection(Connection):

    def __init__(self, inmod, outmod, **kwargs):
        # 1st inputs are for external connection
        kwargs['outSliceTo'] = outmod.dim
        Connection.__init__(self, inmod, outmod, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        n = self.outmod
        outbuf += np.sum(inbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        #CHECKME: not setting derivatives -- this means the multiplicative weight is never updated!
        inerr += 0

class GFNNIntConnection(Connection):
    """"""

    learnParams = None
    type = None
    f = None
    learn = False
    w = None
    l = None
    m1 = None
    m2 = None
    e = None
    roote = None
    k = None
    c0 = None
    c = None
    mask = 1
    kSteps = []

    def __init__(self, inmod, outmod, learnParams=None, c0=None, **kwargs):
        # 2nd half for int connections
        kwargs['outSliceFrom'] = outmod.dim
        kwargs['outSliceTo'] = outmod.dim*2
        Connection.__init__(self, inmod, outmod, **kwargs)

        if isinstance(outmod, GFNNLayer):
            outmod.conns.append(self)

        if learnParams is None:
            learnParams = {'learn': True, 'w': 0.05, 'l': 0, 'm1': -1, 'm2': -50, 'e': 4, 'k': 1 } # Critical learning rule

        learnParams['type'] = 'allfreq' # only supported type for now

        self.setArgs(learnParams=learnParams, c0=c0)
        self.setLearnParams(learnParams)
        if c0 is not None:
            self.c0 = np.complex64(c0)
        self.c = np.zeros((self.outdim, self.indim), np.complex64)
        if inmod == outmod:
            # Don't learn self-connections
            # This could be inverted guassian too
            self.mask = 1-eye(self.outdim, self.indim)

        # ParameterContainer.__init__(self, self.indim*self.outdim)
        self.k *= self.mask
        self.kSteps = np.zeros((4, self.outdim, self.indim), np.complex64)

        self.randomize()
        self.reset()

    def setLearnParams(self, learnParams):
        n1 = self.inmod
        n2 = self.outmod

        self.type = learnParams['type']

        f1 = np.tile(n1.f, (n2.outdim, 1))
        f2 = np.tile(n2.f, (n1.outdim, 1)).T
        if self.type == 'allfreq':
            # Full series of resonant monomials
            f = 2 * f1 * f2 / (f1 + f2)

        if n2.fspac == 'log':
            self.f = f
            self.w = learnParams['w'] * n2.f
            self.l = learnParams['l'] * f
            self.m1 = learnParams['m1'] * f
            self.m2 = learnParams['m2'] * f
            self.k = learnParams['k'] * f
        else:
            self.f = np.ones(np.size(f))
            self.w = learnParams['w']
            self.l = learnParams['l']
            self.m1 = learnParams['m1']
            self.m2 = learnParams['m2']
            self.k = learnParams['k']

        self.learn = learnParams['learn']
        self.e = np.complex64(learnParams['e'])
        self.roote = np.sqrt(self.e)

    def randomize(self):
        # FullNotSelfConnection.randomize(self)
        size = np.size(self.f)
        # self._params[:] = np.ones(size)*self.stdParams
        if self.c0 is None:
            a0 = np.zeros(size)
            a = spontAmp(np.real(self.l[0,0]), np.real(self.m1[0,0]), np.real(self.m2[0,0]), self.e)
            a0 += np.min(a)
            a0 = a0 * (1 + .01 * np.random.randn(size))
            theta0 = np.random.randn(size)
            c0 = a0 * np.exp(1j * 2 * np.pi * theta0)
            self.c0 = reshape(c0, (self.outdim, self.indim))

        self.c0 *= self.mask

    def reset(self):
        self.c[:] = self.c0

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf

    def _backwardImplementation(self, outerr, inerr, inbuf):
        #CHECKME: not setting derivatives -- this means the multiplicative weight is never updated!
        inerr += 0

    def _backwardImplementation(self, outerr, inerr, inbuf):
        p = reshape(self.params, (self.outdim, self.indim)) * self.mask
        inerr += dot(p.T, outerr)
        ds = self.derivs
        ds += outer(inbuf, outerr).T.flatten()

if __name__ == "__main__":
    # from pybrain.tests import runModuleTestSuite
    # import pygfnn.tests.unittests.structure.connections.test_gfnn_connections as test
    # runModuleTestSuite(test)
    from pybrain.structure.networks.recurrent import RecurrentNetwork
    from pybrain import LinearLayer, FullConnection, FullNotSelfConnection, IdentityConnection
    from pygfnn import GFNNLayer, RealIdentityConnection, RealMeanFieldConnection
    N = RecurrentNetwork('simpleGFNN')
    i = LinearLayer(1, name = 'i')
    h = GFNNLayer(200, name = 'gfnn')
    o = LinearLayer(200, name = 'o')
    N.addOutputModule(o)
    N.addInputModule(i)
    N.addModule(h)
    N.addConnection(GFNNExtConnection(i, h, name = 'f1'))
    N.addRecurrentConnection(GFNNIntConnection(h, h, name = 'r1'))
    N.addConnection(RealIdentityConnection(h, o, name = 'i1'))
    N.sortModules()