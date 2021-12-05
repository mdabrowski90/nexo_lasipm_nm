# by Mietek Dabrowski ( m.dabrowski@kuleuven.be )
# Date: 11/27/2021
# This script contains Base Signal Processing Functions for the MD-LASIPM FE model and processing of measured data.
# Copyrights M. Dabrowski @ KU Leuven. When using or copying the script or parts of it, please, acknowledge/reference the author and the KU Leuven University.

import numpy as np
import scipy.signal as scipy_sig

import matplotlib.pyplot as plt
import csv

# Time-domain SiPM signal
def gen_sipm_sig(tau, TS=1e-9, NS=1000000):
    # NS - Number of samples
    # TS - Time step between samples
    
    X = [ (x+1)*TS for x in range(-int(np.floor(NS/2)), int(np.floor(NS/2)))]
    sig = []
    for x in X:
        if x < 0:
            sig.append(0)
        else:
            sig.append(np.exp(-x/tau))

    return X, sig

# Generates Triangular weighting function w/ a given peaking time
def gen_triang_wf(tp, TS=1e-9, NS=1000000):
    # tp - Peaking time
    # NS - Number of samples
    # TS - Time step between samples
    
    X = [ (x+1)*TS for x in range(-int(np.floor(NS/2)), int(np.floor(NS/2)))]
    
    tri_ = []
    tri_.append(1)
    
    for i in range(1, int(np.floor(NS/2)) ):
        if i*TS<tp:
            y = (tp - i*TS)/tp
        else:
            y = 0
        tri_.append(y)
    
    tri = []
    for y_ in np.flip(tri_[1:],0):
        tri.append(y_)
    for y_ in tri_:
        tri.append(y_)
    tri.append(0)
    
    return X, tri

# Generate delta pulse
def gen_delta(TS=1e-9, NS=1000000):
    # NS - Number of samples
    # TS - Time step between samples
    X = [ (x+1)*TS for x in range(-int(np.floor(NS/2)), int(np.floor(NS/2)))]
    sig = []

    for x in X:
        if x == 0:
            sig.append(1)
        else:
            sig.append(0)

    return X, sig

# Returns X and Y values
def trig_filter(tp, isig = None, tau=None, f0_f1=[1e7, 5e7], fhpf=1e5, TS=1e-9, NS=1000000, SIPM_SIGNAL=True, POST_AMP=True, HPF=False):
    # tp - Peaking time
    # NS - Number of samples
    # TS - Time step between samples

    #print('#HERE #1')
    if isig is None:
        #print('#HERE #2')
        if SIPM_SIGNAL:
            X, sipm_sig = gen_sipm_sig(tau=tau, TS=TS, NS=NS)
        else:
            X, sipm_sig = gen_delta(TS=TS, NS=NS)

        if POST_AMP:
            sipm_sig = filter_butter_lp( sipm_sig, TS, fcutoff=f0_f1[0] )
            sipm_sig = filter_butter_lp( sipm_sig, TS, fcutoff=f0_f1[1] )
    else:
        NS = len(isig)
        sipm_sig = isig
    
    # Generate Triangular weighting function and scale for unity area
    X, tri = gen_triang_wf(tp, TS=TS, NS=NS)
    tri = np.array(tri)*1/tp
    sig = scipy_sig.fftconvolve(tri, sipm_sig, mode='same') / np.sum(tri)
    
    if HPF:
        sig = filter_butter_hp(sig, TS, fcutoff=fhpf)

    #print('#HERE #3')
    
    return X, sig, sipm_sig, tri

def ss_fft(ydata, dxt, TYPE='normal', HANN=False):
    # TYPE:
    # normal - magnitude of the same (factor of 2 as single-sided)
    # nsd - Noise Spectral Density (single-sided)
    # rms - returns rms-value per bin (NSD integrated over bin-to-bin distance)

    N = len(ydata)
    ydata_ = ydata

    # Applies Hanning window if True
    if HANN:
        ydata_ = np.multiply(ydata, np.hanning(N) ) * 2

    yfft = np.fft.rfft(ydata_)
    xfft = np.fft.rfftfreq(N, dxt)

    dxf = 1/( len(ydata_)*dxt )

    if TYPE.lower() == 'normal':
        yfft = 2*np.abs(yfft)/N
    elif TYPE.lower() == 'nsd':
        yfft = 2*np.abs(yfft) / N * np.sqrt( 1/(2*dxf) )
    elif TYPE.lower() == 'rms':
        yfft = 2*np.abs(yfft)/N * 2**(-1/2)
    else:
        return False

    return xfft, yfft, ydata_
    
def filter_butter_lp(ydata, dt, fcutoff, N=1, FF=False):
    sos = scipy_sig.butter(N, fcutoff, 'lowpass', fs=1/dt, output='sos')
    if not FF:
        return scipy_sig.sosfilt(sos, ydata)
    else:
        return scipy_sig.sosfiltfilt(sos, ydata)
    
def filter_butter_hp(ydata, dt, fcutoff, N=1, FF=False):
    sos = scipy_sig.butter(N, fcutoff, 'highpass', fs=1/dt, output='sos')
    if not FF:
        return scipy_sig.sosfilt(sos, ydata)
    else:
        return scipy_sig.sosfiltfilt(sos, ydata)
    
def extract_je_data(fname):
    csvdata = csv.reader(open(fname, 'r'), delimiter=',')
    # Skip first row
    next(csvdata)
    
    data = []
    data.append([])
    data.append([])
    
    for row in csvdata:
        data[0].append(float(row[1]))
        data[1].append(float(row[2]))
        
    return data

def data_find_nearest_index(data, x=0):
    index = min(range(len(data)), key=lambda i: abs(data[i]-x))
    return index, data[index]

def f_fit_gauss(x, a, mean, std):
    pi = 3.141592653589793238
    ee = 2.71828
    return a * ee**(-1/2* ((x-mean)/std)**2 )

def f_fit_edecay(x, tau, a, b):
    ee = 2.71828
    return a * ee**( -x/tau ) + b

## ################## ##
##     MAIN BODY      ##
## ################## ##

if __name__ == "__main__":
    pass
