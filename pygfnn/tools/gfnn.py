__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

import numpy as np

def slope(r, a, b1, b2, e):
    return a + 3 * b1 * np.power(r, 2) + (5 * e * b2 * np.power(r, 4) - 3 * e**2 * b2 * np.power(r, 6)) / (np.power((1 - e * np.power(r, 2)), 2))

def spontAmp(a, b1, b2, e):
    """Finds spontaneous amplitude(s) of fully expanded canonical model"""
    if b2 == 0 and e != 0:
        e = 0
    # Find r* numerically
    r = np.roots([e * (b2 - b1), 0, b1 - e * a, 0, a, 0])
    r = [x for x in r if np.abs(np.imag(x)) < np.spacing(1)]
    r = np.real(np.unique(r)) # only unique real values
    r = [x for x in r if x >= 0] # no negative amplitude
    if b2 != 0:
        r = [x for x in r if x < 1/np.sqrt(e)] # r* below the asymptote only
    # Take only stable r*
    sl1 = slope(r, a, b1, b2, e)
    ind1 = [i for i in range(len(sl1)) if sl1[i] < 0]
    ind2a = [i for i in range(len(sl1)) if sl1[i] == 0]
    sl2b = slope(r-np.spacing(1), a, b1, b2, e)
    ind2b = [i for i in range(len(sl2b)) if sl2b[i] < 0]
    sl2c = slope(r+np.spacing(1), a, b1, b2, e)
    ind2c = [i for i in range(len(sl2b)) if sl2c[i] < 0]
    ind2 = np.intersect1d(ind2a, np.intersect1d(ind2b, ind2c))
    ind = np.concatenate((ind1, ind2)).astype('int')
    r = r[ind]
    if len(ind) > 1:
        r.sort()
        r = r[::-1]
    else:
        r = np.array([r])
    return r

def rk4(t0, y0, h, dydtFn, *args):
    half_h = h/2

    k1 = h * dydtFn(t0, y0, *args)

    t2 = t0 + half_h
    y2 = y0 + (k1/2)
    k2 = h * dydtFn(t2, y2, *args)

    y3 = y0 + (k2/2)
    k3 = h * dydtFn(t2, y3, *args)

    t4 = t0 + h
    y4 = y0 + k3
    k4 = h * dydtFn(t4, y4, *args)

    yi = y0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return yi

def zcrk4(t0, h, m, extin):
    z0 = m.getZ()
    nConns = len(m.conns)
    conns0 = np.zeros((nConns, len(z0), len(z0)), np.complex64)
    for i in range(nConns):
        conns0[i][:] = m.conns[i].c

    z = z0.copy()
    zi = z0.copy()
    conns = conns0.copy()
    wconn = np.zeros((len(z0), len(z0)), np.complex64)

    t = t0
    for kx in range(4):
        m.kSteps[kx][:] = h * zdot(t, z, m, extin, conns)

        for i in range(nConns):
            if m.conns[i].learn:
                wconn[:] = conns[i]
                zi[:] = m.conns[i].inmod.getZ()
                m.conns[i].kSteps[kx][:] = h * cdot(t, wconn, m.conns[i], zi, z)
            else:
                m.conns[i].kSteps[kx][:] = 0

        if kx == 0 or kx == 1:
            z = z0 + m.kSteps[kx]/2

            for i in range(nConns):
                conns[i] = conns0[i] + m.conns[i].kSteps[kx]/2

            if kx == 0:
                t = t0 + h/2
        elif kx == 2:
            z = z0 + m.kSteps[kx]

            for i in range(nConns):
                conns[i] = conns0[i] + m.conns[i].kSteps[kx]

            t = t0 + h

    z[:] = z0 + (m.kSteps[0] + 2*m.kSteps[1] + 2*m.kSteps[2] + m.kSteps[3])/6

    for i in range(nConns):
        conns[i][:] = conns0[i] + (m.conns[i].kSteps[0] + 2*m.conns[i].kSteps[1] + 2*m.conns[i].kSteps[2] + m.conns[i].kSteps[3])/6

    return z, conns

# def nmlExp(x, m, g):
#     a = np.abs(x) + np.spacing(1)
#     return m * np.tanh(g*a) * x / a

def xOverExp(a, x):
    return x / (1 - a*x)

def oneOverExp(a, x):
    return 1. / (1 - a*x)

# def conjExp(a, x):
#     return xOverExp(a, x) * oneOverExp(a, np.conj(x))

def extActivation(w, x, z, roote):
    return w * xOverExp(roote, x) * oneOverExp(roote, np.conj(z))

def intActivation(w, c, zi, zj, roote):
    return w * np.dot(c, xOverExp(roote, zj)) * oneOverExp(roote, np.conj(zi))

def extStimulus(z, m, extin):
    return extActivation(m.w, extin, z, m.roote)

def intStimulus(z, m, conns=None):
    x_int = 0
    for i in range(len(m.conns)):
        conn = m.conns[i]
        if conns is not None:
            c = conns[i]
        else:
            c = conn.c
        zi = conn.inmod.getZ()
        x_int += intActivation(conn.w, c, zi, z, conn.roote)
    return x_int

def zdot(t, z, m, extin, conns=None):
    # stimulus: external and internal
    x = extStimulus(z, m, extin) \
        + intStimulus(z, m, conns)

    # cannonical model
    abz = np.abs(z)
    b1_exp = m.b1 * abz**2
    b2_exp = m.b2 * m.e * abz**4
    ez2_exp = 1 - (m.e * (abz**2))
    dzdt = z * (m.a + b1_exp + (b2_exp / ez2_exp)) + x
    return dzdt

def cdot(t, c, conn, zi, zj):
    x = conn.k * np.outer(xOverExp(conn.roote, zj), xOverExp(conn.roote, zi.conj().T))
    abc = np.abs(c)
    abc_2 = np.power(abc, 2)
    abc_4 = np.power(abc, 4)
    div = np.divide(conn.e * conn.m2 * abc_4, 1 - conn.e * abc_2)
    dcdt = c * (conn.l + conn.m1 * abc_2 + div) + x
    return dcdt

def fdot(t, f, z, m, extin):
    #s = o/h or: sin = im(z) / abs(z)
    abz = np.abs(z)
    abx = np.abs(extin)
    if np.any(abx):
        dfdt = -m.e_f * abx * np.imag(z) * np.power(abz, -2)
    else:
        dfdt = 0.0
    dfdt -= m.e_h * ((f - m.fr0) / m.fr0)
    return m.w * dfdt

def zfrk4(t0, h, m, extin):
    z0 = m.getZ()
    f0 = m.fr

    z = z0.copy()
    f = f0.copy()

    t = t0
    for kx in xrange(4):
        m.kSteps[kx][:] = h * zdot(t, z, m, extin)
        m.fkSteps[kx][:] = h * fdot(t, f, z, m, extin)

        if kx == 0 or kx == 1:
            z[:] = z0 + m.kSteps[kx]/2
            f[:] = f0 + m.fkSteps[kx]/2

            if kx == 0:
                t = t0 + h/2
        elif kx == 2:
            z[:] = z0 + m.kSteps[kx]
            f[:] = f0 + m.fkSteps[kx]

            t = t0 + h

    z[:] = z0 + (m.kSteps[0] + 2*m.kSteps[1] + 2*m.kSteps[2] + m.kSteps[3])/6
    f[:] = f0 + (m.fkSteps[0] + 2*m.fkSteps[1] + 2*m.fkSteps[2] + m.fkSteps[3])/6

    return z, f

def zfcrk4(t0, h, m, extin):
    z0 = m.getZ()
    f0 = m.fr
    nConns = len(m.conns)
    conns0 = np.zeros((nConns, len(z0), len(z0)), np.complex64)
    for i in xrange(nConns):
        conns0[i][:] = m.conns[i].c

    z = z0.copy()
    zi = z0.copy()
    f = f0.copy()
    conns = conns0.copy()
    wconn = np.zeros((len(z0), len(z0)), np.complex64)

    t = t0
    for kx in xrange(4):
        m.kSteps[kx][:] = h * zdot(t, z, m, extin, conns)
        m.fkSteps[kx][:] = h * fdot(t, f, z, m, extin)

        for i in xrange(nConns):
            if m.conns[i].learn:
                wconn[:] = conns[i]
                zi[:] = m.conns[i].inmod.getZ()
                m.conns[i].kSteps[kx][:] = h * cdot(t, wconn, m.conns[i], zi, z)
            else:
                m.conns[i].kSteps[kx][:] = 0

        if kx == 0 or kx == 1:
            z[:] = z0 + m.kSteps[kx]/2
            f[:] = f0 + m.fkSteps[kx]/2

            for i in xrange(nConns):
                conns[i] = conns0[i] + m.conns[i].kSteps[kx]/2

            if kx == 0:
                t = t0 + h/2
        elif kx == 2:
            z[:] = z0 + m.kSteps[kx]
            f[:] = f0 + m.fkSteps[kx]

            for i in xrange(nConns):
                conns[i] = conns0[i] + m.conns[i].kSteps[kx]

            t = t0 + h

    z[:] = z0 + (m.kSteps[0] + 2*m.kSteps[1] + 2*m.kSteps[2] + m.kSteps[3])/6
    f[:] = f0 + (m.fkSteps[0] + 2*m.fkSteps[1] + 2*m.fkSteps[2] + m.fkSteps[3])/6

    for i in xrange(nConns):
        conns[i][:] = conns0[i] + (m.conns[i].kSteps[0] + 2*m.conns[i].kSteps[1] + 2*m.conns[i].kSteps[2] + m.conns[i].kSteps[3])/6

    return z, f, conns

def limitC(c, roote):
    maxAmp = np.real(1./roote)
    abc = np.abs(c)
    if np.max(abc) <= maxAmp:
        return c
    print('limitC start')
    print(np.max(abc))
    ttable = np.greater(abc, maxAmp) * 1
    nttable = 1 - ttable
    scaling = np.divide(maxAmp, (abc * ttable) + nttable)
    scaling = (scaling * ttable) + nttable
    limitedC = c * scaling
    print('limitC end')
    print(np.max(np.abs(limitedC)))
    return limitedC

if __name__ == '__main__':
    spontAmp(0, -1, -1, 1)