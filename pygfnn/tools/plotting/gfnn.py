__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def freqs2labels(startF, endF):
    ticks = [startF]
    while ticks[-1] < endF:
        ticks.append(ticks[-1]*2)
        pass
    labels = [str(x) for x in ticks]
    return ticks, labels

def plotOutput(Y, T):
    fig1 = plt.figure()
    plt.plot(T, Y)
    plt.show()
    return fig1

def ampx(Z, dt, startF, endF, fig1=None):
    if fig1 is not None:
        plt.figure(fig1.number)
    else:
        fig1 = plt.figure()

    startT = 0
    endT = len(Z)*dt
    im1 = plt.imshow(np.abs(Z).T, extent=[startT,endT,startF,endF],
        aspect='auto', interpolation='nearest')
    ax1 = plt.gca()
    ax1.set_yscale('log')
    ticks, labels = freqs2labels(startF, endF)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(labels)

    plt.title('Amplitudes of oscillators over time')
    plt.xlabel('Time (s)');
    plt.ylabel('Oscillator natural frequency (Hz)');

    cb = plt.colorbar()
    cb.set_label('Amplitude')
    return fig1

def plotConns(c, startF, endF, ampFig=None, phaseFig=None):
    if ampFig is not None:
        plt.figure(ampFig.number)
    else:
        ampFig = plt.figure()
    im1 = plt.imshow(np.abs(c), extent=[startF,endF,endF,startF],
        aspect='auto', interpolation='nearest')
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ticks, labels = freqs2labels(startF, endF)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    plt.colorbar()
    plt.title('Amplitudes of connection matrix')

    if phaseFig is not None:
        plt.figure(phaseFig.number)
    else:
        phaseFig = plt.figure()
    im2 = plt.imshow(np.angle(c),
        extent=[startF,endF,endF,startF], aspect='auto', interpolation='nearest')
    ax2 = plt.gca()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    plt.colorbar()
    plt.title('Phases of connection matrix')

    return ampFig, phaseFig