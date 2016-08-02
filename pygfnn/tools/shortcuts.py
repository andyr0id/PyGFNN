__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

import numpy as np
from pybrain import LinearLayer, FullConnection, FullNotSelfConnection, IdentityConnection
from pygfnn import GFNN, GFNNLayer, AFNNLayer, GFNNExtConnection, GFNNIntConnection, RealIdentityConnection, RealMeanFieldConnection, AbsPhaseIdentityConnection

OSC_LINEAR = { 'a': -1, 'b1': 0, 'b2': 0, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_CRITICAL = { 'a': 0, 'b1': -1, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_DETUNE = { 'a': 0, 'b1': -1, 'b2': -1, 'd1': 1, 'd2': 0, 'e': 1 }
OSC_LIMIT_CYCLE = { 'a': 1, 'b1': -1, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }
OSC_DOUBLE_LIMIT_CYCLE = { 'a': -1, 'b1': 3, 'b2': -1, 'd1': 0, 'd2': 0, 'e': 1 }

LEARN_LINEAR = {'learn': True, 'w': 0.05, 'l': -.1, 'm1': 0, 'm2': 0, 'e': 4, 'k': 1 } # Linear learning rule
LEARN_CRITICAL = {'learn': True, 'w': 0.05, 'l': 0, 'm1': -1, 'm2': -50, 'e': 4, 'k': 1 } # Critical learning rule
LEARN_STRONG_CRITICAL = {'learn': True, 'w': 0.05, 'l': 0, 'm1': -1, 'm2': -50, 'e': 16, 'k': 1 } # Critical, stronger nonlinearity
LEARN_SUPER_CRITICAL = {'learn': True, 'w': 0.05, 'l': .001, 'm1': -1, 'm2': -50, 'e': 16, 'k': 1 } # Supercritical learning rule

NOLEARN_ALLFREQ = {'learn': False, 'w': 0.05, 'type': 'allfreq' }

class NetworkError(Exception): pass

def buildGFNN(dim, **options):
    opt = {
        'name': 'gfnn',
        'oscParams': None,
        'freqDist': None,
        'fs': 40.,
        'adaptive': False,
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

    if opt['adaptive']:
        gfnnClass = AFNNLayer
    else:
        gfnnClass = GFNNLayer
    h = gfnnClass(dim,
        oscParams = opt['oscParams'], freqDist = opt['freqDist'], name = 'h')

    if issubclass(opt['outConn'], RealMeanFieldConnection) or issubclass(opt['outConn'], AbsPhaseIdentityConnection):
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

def getInitC(n1, n2, ratios, gfnnLayer='h', thresh=0.01):
    f1 = n1[gfnnLayer].f
    f2 = n2[gfnnLayer].f
    conn = n1.recurrentConns[0]
    w = conn.w
    c0 = conn.c.copy()
    rspace = [None] * (len(ratios)*2)
    for i,r in enumerate(ratios):
        r1 = float(r[0])/r[1]
        r2 = float(r[1])/r[0]
        rspace[i*2] = r1
        rspace[(i*2)+1] = r2
    for i, _f1 in enumerate(f1):
        for j,_f2 in enumerate(f2):
            r = _f2 / _f1
            for tr in rspace:
                diff = abs(tr-r)
                if diff / tr < thresh:
                    c0[i,j] = ((w[i] + w[j]) / 2) + 0j
                    break
    return conn._randomizePhase(c0) * conn.mask

if __name__ == '__main__':
    n = buildGFNN(128, **{
        'learnParams': NOLEARN_ALLFREQ,
        'outConn': RealMeanFieldConnection,
        'outDim': 8
    })
    print(n)
    print(getInitC(n, n, [(1,2), (1,3), (2,3)]))