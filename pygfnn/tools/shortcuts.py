__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

from pybrain import LinearLayer, FullConnection, FullNotSelfConnection, IdentityConnection
from pygfnn import GFNN, GFNNLayer, GFNNExtConnection, GFNNIntConnection, RealIdentityConnection, RealMeanFieldConnection

OSC_LINEAR = { 'a': -1, 'b1': 0, 'b2': 0, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_CRITICAL = { 'a': 0, 'b1': -1, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_DETUNE = { 'a': 0, 'b1': -1, 'b2': -1, 'd1': 1, 'd2': 0, 'e': 1 }
OSC_LIMIT_CYCLE = { 'a': 1, 'b1': -1, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_DOUBLE_LIMIT_CYCLE = { 'a': -1, 'b1': 3, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }

LEARN_LINEAR = {'learn': True, 'w': 0.05, 'l': -.1, 'm1': 0, 'm2': 0, 'e': 4, 'k': 1 } # Linear learning rule
LEARN_CRITICAL = {'learn': True, 'w': 0.05, 'l': 0, 'm1': -1, 'm2': -50, 'e': 4, 'k': 1 } # Critical learning rule
LEARN_STRONG_CRITICAL = {'learn': True, 'w': 0.05, 'l': 0, 'm1': -1, 'm2': -50, 'e': 16, 'k': 1 } # Critical, stronger nonlinearity
LEARN_SUPER_CRITICAL = {'learn': True, 'w': 0.05, 'l': .001, 'm1': -1, 'm2': -50, 'e': 16, 'k': 1 } # Supercritical learning rule

class NetworkError(Exception): pass

def buildGFNN(dim, **options):
    opt = {
        'name': 'gfnn',
        'oscParams': None,
        'freqDist': None,
        'fs': 40.,
        'learnParams': None,
        'c0': None,
        'outConn': RealIdentityConnection,
        'outDim': 1,
    }
    for key in options:
        if key not in opt.keys():
            raise NetworkError('buildGFNN unknown option: %s' % key)
        opt[key] = options[key]

    n = GFNN(opt['name'], fs = opt['fs'])
    i = LinearLayer(1, name = 'i')

    h = GFNNLayer(dim,
        oscParams = opt['oscParams'], freqDist = opt['freqDist'], name = 'h')

    if issubclass(opt['outConn'], RealMeanFieldConnection):
        outDim = opt['outDim']
    else:
        outDim = dim
    o = LinearLayer(outDim, name = 'o')

    n.addInputModule(i)
    n.addOutputModule(o)
    n.addModule(h)
    n.addConnection(GFNNExtConnection(i, h, name = 'ic'))

    if opt['learnParams'] is not None:
        n.addRecurrentConnection(GFNNIntConnection(h, h,
            learnParams = opt['learnParams'], c0 = opt['c0'],
            name = 'rc'))


    if issubclass(opt['outConn'], RealMeanFieldConnection):
        oTo = 0
        connPerInter = int(round(float(dim)/outDim))
        for x in range(outDim):
            hFrom = x * connPerInter
            hTo = min(hFrom+connPerInter, dim)
            c = RealMeanFieldConnection(h, o, name='gxc'+`x`,
                inSliceFrom=hFrom, inSliceTo=hTo,
                outSliceFrom=oTo, outSliceTo=oTo+1)
            n.addConnection(c)
            oTo += 1
    else:
        n.addConnection(opt['outConn'](h, o, name = 'oc'))
    n.sortModules()
    return n

if __name__ == '__main__':
    n = buildGFNN(200, **{
        'outConn': RealMeanFieldConnection,
        'outDim': 8
    })
    print(n)