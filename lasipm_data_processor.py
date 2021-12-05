# by Mietek Dabrowski ( m.dabrowski@kuleuven.be )
# Date: 11/27/2021
# This script processes computes the ENC on the measured SiPM data and compares it with the MD-LASIPM FE model. 
# Copyrights M. Dabrowski @ KU Leuven. When using or copying the script or parts of it, please, acknowledge/reference the author and the KU Leuven University.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# MD libs
import lasipm_base_sp as lbs
import lasipm_tf_noise_model as lnm

# JE libs
import wavedumpReader

## ################## ##
##     MAIN BODY      ##
## ################## ##

if __name__ == "__main__":

    plt.ion()
    
    PREFIX = '_plt'
    
    # SWITCHES
    PLOT = True
    PLOT_DIST = True
    HPF = False
    REMOVE_DC = False

    FFT_WINDOW = True
    
    # Variables
    FS = 256e6
    TS = 1/FS
    fhpf = 5e5
    
    TIA_G = 1e3*(1+1e3/511)*10*5/3
    LSB = 2/(2**(12)-1)
    LSB_IR = LSB/TIA_G
    
    OVS_TP = 100
    
    # 
    XF = np.logspace(1, 9, 100000)
    
    N = 30551
    #N = 5000
    N_BIN = 250
    
    TP = [ 0.025e-6, 0.05e-6, 0.1e-6, 0.25e-6, 0.5e-6, 0.75e-6, 1e-6]
    N_BINS = [ 200, N_BIN, N_BIN, N_BIN, N_BIN, N_BIN, N_BIN ]
    
    #  Defining peaking times % 
    TP = np.logspace(-8, -6, 11)
    TP = [ np.ceil(tp*1e9/4)*1e-9*4 for tp in TP ]
    TP = np.around(TP, decimals=9)
    
    TP = [ 0.025e-6, 0.1e-6, 0.25e-6, 0.5e-6, 1e-6]
    TP = TP[0:2]
    print(TP)
    
    #exit(0)
    N_BINS = [ N_BIN for tp in TP ]
    
    ## #################################### ##
    ##     READ and SELECT INPUT PULSES     ##
    ## #################################### ##
    
    dataFile = wavedumpReader.DataFile('2s3p_sipmData_5.5V.dat')
    dataFile.file.seek(0) 
    eventSize = 1024;
    
    # Aux variables / Arrays for data processing
    pulses = []
    
    y0 = 0
    y0_list = [y0 for x in range(eventSize)]
    
    for i in range(N):
        try:
            trace = np.fromfile(dataFile.file, dtype=np.dtype('<H'), count=eventSize)*-1
            pk_ = np.amax(trace)
            
            #if -2005 <= np.amin(trace[0:500]) and  np.amax(trace[0:500]) <= -1920 and np.amax(trace[650:]) <= -1920:
            if np.amax(trace[0:500]) <= -1920 and np.amax(trace[650:]) <= -1920:
                
                # Apply 25 ns filter to optimize the SNR for trace sorting
                xtri_, trace_tf_, isig_, tri_ = lbs.trig_filter( tp=0.025e-6, isig=trace, fhpf=fhpf, TS=TS, HPF=False )
                
                pk_ = np.amax(trace_tf_[10:1015])
                pk__ = np.amax(trace_tf_[500:530])
                
                # Below thresholds might have to be adjusted for each file/sipm combination
                if (pk__ >= -1920 and pk__ < -1875) and (pk_ in trace_tf_[500:530]) and np.amax(trace_tf_[580:1015]) <= -1930 and np.amax(trace_tf_[540:560]) <= -1900 and np.amax(trace_tf_[560:580]) <= -1924:
                    if REMOVE_DC:
                        trace = trace-np.mean(trace[0:450])
                    pulses.append(trace)
        
        except Exception as e:
            print(e)
            print(i)
            break
        
    #input('Press [ENTER] to finish.')
    #exit(0)
    #print( np.shape(pulses) )
    
    pulses = np.array(pulses)
    av_pulse = np.average(pulses, axis=0)
    ymean_ = np.mean(av_pulse[0:450])
    av_pulse = av_pulse - ymean_

    # Store nominal averaged pulse in a variable
    av_pulse_ = av_pulse
    x_av_pulse_ = np.array( range(len( av_pulse_ )) )
    
    # Re-assign and extend average pulse
    av_pulse = []
    av_pulse.extend( y0_list )
    av_pulse.extend( av_pulse_ )
    av_pulse.extend( y0_list )

    #TS = 1
    
    fig = plt.figure()
    plt.title(r'$ \bf Transien \ Inputs \ Signals \ to \ Data \ Processor $')
    pulses_ = []
    for trace_ in pulses:
        trace_no_dc_ = np.array(trace_) - ymean_
        trace__ = []
        trace__.extend( y0_list )
        trace__.extend( trace_no_dc_ )
        trace__.extend( y0_list )
        pulses_.append( trace__ )
        
        plt.plot( np.array( range(len(trace_no_dc_)) )*TS, trace_no_dc_, color='orangered', alpha=0.05 )
    
    pulses = pulses_
    
    ## ########################################################## ##
    ##      PLOT AVG PULSE & COMPUTE DECAY TIME & QSPE FACTOR     ##
    ## ########################################################## ##

    qspe_f = np.around(( np.trapz(av_pulse_[516:640]) )*TS, decimals=8)
    
    ix0 = 525
    ix1 = 750
    x_av_fit = np.array( range(len( av_pulse_[ix0:ix1] )) )
    dec_var_fit, cov = curve_fit(lbs.f_fit_edecay, x_av_fit, av_pulse_[ix0:ix1], maxfev = 10000)
    dec_fit_ = [ lbs.f_fit_edecay( x, dec_var_fit[0], dec_var_fit[1], dec_var_fit[2] ) for x in x_av_fit ]
    plt.plot( x_av_pulse_*TS, av_pulse_, color='red', linewidth=0.75 )
    plt.plot( x_av_pulse_[ix0:ix1]*TS, dec_fit_, '--', color='blue', linewidth=0.25 )
    plt.text(2.2e-6, dec_var_fit[1]+dec_var_fit[2] , r'$ \tau = '+str(dec_var_fit[0]*1e9*TS)[0:6]+' [ns] $', fontsize=5)
    plt.text(2.2e-6, dec_var_fit[1]+dec_var_fit[2]-10 , r'$ QSPE_F = '+str(qspe_f)+ ' \ [V*s] $', fontsize=5)
    plt.xlabel(' Time [s] ')
    plt.ylabel(' Amplitude [counts] ')
    fig.savefig('fig'+PREFIX+'_filtered_sigs.png', dpi=600,facecolor=(1, 1, 1))
    plt.xlim([1.75e-6, 2.5e-6])
    fig.savefig('fig'+PREFIX+'_filtered_sigs_zoom.png', dpi=600,facecolor=(1, 1, 1))
    
    #input('Press [ENTER] to finish.')
    #exit(0)
    
    ## ################################ ##
    ##     PEAKING TIME COMPUTATION     ##
    ## ################################ ##
    
    ybase_ = np.mean(av_pulse_[0:450])
    #print(ybase_)
    
    fig = plt.figure()
    plt.title(r'$ \bf Extraction \ of \ peaking \ times $')
    
    TP_EXTR = []
    for j, tp in enumerate(TP):
        xtri_, av_pulse_tf_, isig_, tri_ = lbs.trig_filter( tp=tp, isig=av_pulse, fhpf=fhpf, TS=TS, HPF=HPF )
        
        av_pulse_tf_ = np.array(av_pulse_tf_[eventSize:2*eventSize])
        
        x_av_pulse_interp_ = np.array( [ x/OVS_TP for x in range(OVS_TP*len(x_av_pulse_)) ] )
        y_av_pulse_tf_interp_ = np.interp(x_av_pulse_interp_, x_av_pulse_, av_pulse_tf_)
        
        ymax_ = np.amax( y_av_pulse_tf_interp_ )
        ix0, v_ = lbs.data_find_nearest_index( y_av_pulse_tf_interp_, ymax_)
        ix1, v_ = lbs.data_find_nearest_index( y_av_pulse_tf_interp_[0:540*OVS_TP], ymax_*0.01 )
        
        TP_EXTR.append( np.around((ix0-ix1)*TS/OVS_TP, decimals=10) )
        
        #plt.plot( range(len(av_pulse_tf_)), av_pulse_tf_, color='red', linewidth=0.75 )
        plt.plot( (x_av_pulse_-x_av_pulse_interp_[ix1])*TS, av_pulse_tf_, '-o', color='red', linewidth=0.25, markersize=1.5 )
        plt.plot( (x_av_pulse_interp_-x_av_pulse_interp_[ix1])*TS, y_av_pulse_tf_interp_, '--o', color='blue', linewidth=0.25, markersize=0.75 )
        plt.plot( np.array([0, x_av_pulse_interp_[ix0]-x_av_pulse_interp_[ix1]])*TS, [y_av_pulse_tf_interp_[ix1], y_av_pulse_tf_interp_[ix0]], '-o', color='darkorange', markersize=2 )

    plt.xlabel('Time [s] ')
    plt.ylabel(' Amplitude [counts] ')
    fig.savefig('fig'+PREFIX+'_tp_extr.png', dpi=600,facecolor=(1, 1, 1))
    plt.xlim([-0.2e-6, 1.2e-6])
    fig.savefig('fig'+PREFIX+'_tp_extr_zoom.png', dpi=600,facecolor=(1, 1, 1))
    
    print(TP_EXTR)

    #input('Press [ENTER] to finish.')
    #exit(0)

    ## #################### ##
    ##     PROCESS DATA     ##
    ## #################### ##
    pk = []
    vn = []
    
    pk_bin = []
    vn_bin = []
    
    for j, (tp, n_bins_) in enumerate(zip(TP, N_BINS)):

        print('TP = '+str(tp))
        
        if PLOT:
            fig = plt.figure()
            plt.title(r'$ Transient \ Tp='+str(tp)+'$')
        
        av_trace = []
        pk.append([])
        vn.append([])
        
        for trace in pulses:
            if tp != 0:
                xtri_, trace_tf_, isig_, tri_ = lbs.trig_filter( tp=tp, isig=trace, fhpf=fhpf, TS=TS, HPF=HPF )
            else:
                trace_tf_ = trace
                
            trace_tf_ = trace_tf_[eventSize:2*eventSize]
            av_trace.append(trace_tf_)
            
            pk_ = np.amax( trace_tf_[200:800] )
            pk[-1].append( pk_ )
            vn[-1].append( trace_tf_[250] )
            if PLOT:
                plt.plot( np.array( range(len(trace_tf_)) ), trace_tf_, color='orangered', alpha=0.1 )
        
        pk_bin.append( np.histogram(pk[-1], bins=n_bins_, density=False) )
        vn_bin.append( np.histogram(vn[-1], bins=int(n_bins_/2), density=False) )

        if PLOT:
            av_trace = np.average(np.array(av_trace) , axis=0)
            plt.plot( np.array( range(len(av_trace)) ), av_trace, color='red', linewidth=0.75 )
            plt.xlabel(' Sample # ')
            plt.ylabel(' Amplitude [counts] ')
            fig.savefig('fig'+PREFIX+'_event_Tp='+str(tp)+'.png', dpi=600,facecolor=(1, 1, 1))
    
    spe = []
    std_vn = []
    std_pk = []
    
    if PLOT_DIST:
        fig, ax = plt.subplots( int(len(TP)), sharex=True)
        ax[0].set_title(r'$ \bf Histograms \ for \ different \ TPs $')
        
    for i, (pk_bin_, vn_bin_) in enumerate( zip(pk_bin, vn_bin) ):
        pk_var_fit, cov = curve_fit(lbs.f_fit_gauss, pk_bin_[1][1:], pk_bin_[0], p0=[30, 30, 5], maxfev = 10000)
        pk_fit_ = [ lbs.f_fit_gauss( x, pk_var_fit[0], pk_var_fit[1], pk_var_fit[2] ) for x in  pk_bin_[1][1:]]
        
        vn_var_fit, cov = curve_fit(lbs.f_fit_gauss, vn_bin_[1][1:], vn_bin_[0], p0=[50, 0, 5], maxfev = 10000 )
        vn_fit_ = [ lbs.f_fit_gauss( x, vn_var_fit[0], vn_var_fit[1], vn_var_fit[2] ) for x in  vn_bin_[1][1:]]

        spe.append( pk_var_fit[1] - vn_var_fit[1] )
        std_vn.append( abs(vn_var_fit[2]) )
        std_pk.append( abs(pk_var_fit[2]) )
        
        if PLOT_DIST:
            ax[i].plot( pk_bin_[1][1:], pk_bin_[0] )
            ax[i].plot( vn_bin_[1][1:], vn_bin_[0] )

            ax[i].plot( pk_bin_[1][1:], pk_fit_ )
            ax[i].plot( vn_bin_[1][1:], vn_fit_ )
            
            ax[i].set_xlim([-40, 100])
            
    if PLOT_DIST:
        ax[len(TP)-1].set_xlabel(' Amplitude [counts] ')
        fig.savefig('fig'+PREFIX+'_event_hist.png', dpi=600,facecolor=(1, 1, 1))
        
    #input('Press [ENTER] to finish.')
    #exit(0)
    
    fig, ax = plt.subplots(2)
    ax[0].set_title(r'$ \bf Signal \ Peak \ and \ Noise $')
    ax[0].plot( TP, spe, '-o', markersize=1.5 )
    ax[1].plot( TP, std_vn, '-o', markersize=1.5, label='Baseline resolution' )
    ax[1].plot( TP, std_pk , '-o', markersize=1.5, label='Signal peak resolution' )
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    
    ax[0].set_ylabel(' Signal Peak [counts] ')
    ax[1].set_xlabel(' Filter Peaking Time [s] ')
    ax[1].set_ylabel(' Noise [counts] ')

    ax[1].legend(frameon=False, loc='upper right')
    
    fig.savefig('fig'+PREFIX+'_pk_vn.png', dpi=600,facecolor=(1, 1, 1))
    
    ## #################################### ##
    ##     COMPUTE NOISE FROM THE MODEL     ##
    ## #################################### ##
    
    
    lasipm_nm_3p2s = lnm.LASIPM_NM(TP, 168, 3, 2, cj=8.2e-9, rq=9, ic=400e-6, re=1, rbb=2.5, ib=80e-6, rp=100e3, kf2 = 2*1e-14, f0_f1=[1e7, 1e7], fhpf=fhpf, XF=XF)
    #lasipm_nm_3p2s.set_plot()
    if HPF is True:
        lasipm_nm_3p2s.set_hpf()
    enc_3p2s, mtp_extr, amax_, nsd_arr = lasipm_nm_3p2s.compute_enc()
    
    lasipm_nm_3p2s_hc = lnm.LASIPM_NM(TP, 168, 3, 2, cj=8.5e-9, ic=400e-6, re=1, rbb=2.5, rp=50, kf2 = 5*1e-14, f0_f1=[20e6, 7.5e7], fhpf=fhpf, XF=XF)
    #lasipm_nm_3p2s_hc.set_plot()
    if HPF is True:
        lasipm_nm_3p2s_hc.set_hpf()    
    enc_3p2s_hc, mtp_extr_hc, amax_hc_, nsd_arr_hc = lasipm_nm_3p2s_hc.compute_enc()
    
    ## ################################ ##
    ##     PLOT DATA ENC vs MODEL ENC   ##
    ## ################################ ##
    
    fig = plt.figure()
    plt.title(r'$ \bf ENC \ SPE \ in \ Filter \ Peaking \ Time $')
    plt.plot( TP, np.divide(std_vn, spe), '-o', markersize=1.5 )
    plt.plot( TP, np.divide(std_pk, spe), '-o', markersize=1.5 )
    plt.plot( TP, enc_3p2s[0], '-o', markersize=1.25, label='Noise model #1')
    plt.plot( TP, enc_3p2s_hc[0], '-o', markersize=1.25, label='Noise model #2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(' Filter Peaking Time [s] ')
    plt.ylabel(' ENC in SPE ')
    plt.legend(frameon=False, loc='upper left')
    fig.savefig('fig'+PREFIX+'_enc_ftp.png', dpi=600,facecolor=(1, 1, 1))

    fig = plt.figure()
    plt.title(r'$ \bf ENC \ SPE \ in \ Signal \ Peaking \ Time $')
    plt.plot( TP_EXTR, np.divide(std_vn, spe), '-o', markersize=1.5, label='Baseline Res.' )
    plt.plot( TP_EXTR, np.divide(std_pk, spe), '-o', markersize=1.5, label='Signal Peak Res.' )
    plt.plot( mtp_extr, enc_3p2s[0], '-o', markersize=1.25, label='Noise model #1')
    plt.plot( mtp_extr_hc, enc_3p2s_hc[0], '-o', markersize=1.25, label='Noise model #2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(' Signal Peaking Time [s] ')
    plt.ylabel(' ENC in SPE ')
    plt.legend(frameon=False, loc='upper left')
    fig.savefig('fig'+PREFIX+'_enc_stp.png', dpi=600,facecolor=(1, 1, 1))


    ## #################################### ##
    ##     PLOT NOISE SPECTRAL DENSITIES    ##
    ## #################################### ##
    
    fname = '10pF_chargeInjection_12nF_input_noiseScan_4ms.dat'
    dataFile = wavedumpReader.DataFile(fname)
    dataFile.file.seek(0)

    lasipm_nm_3p2s = lnm.LASIPM_NM(TP, 168, 3, 2, cj=8.2e-9, ic=400e-6, re=1, rbb=2.5, ib=80e-6, rp=100e3, kf2 = 2*1e-14, f0_f1=[1e7, 1e7], fhpf=fhpf, XF=XF)
    #lasipm_nm_3p2s.set_plot()
    if HPF is True:
        lasipm_nm_3p2s.set_hpf()
    enc_3p2s, mtp_extr, amax_, nsd_arr = lasipm_nm_3p2s.compute_enc()
    
    NN = 1050
    NN = 1
    NN_SAMPLES = 100000
    NN_SAMPLES = 1000000
    
    n_pulses = []
    
    for i in range(NN):
        try:
            header = np.fromfile(dataFile.file, dtype='I', count=6)
            if len(header) != 6:
                break
            eventSize = (header[0] - 24) // 2
            
            ntrace = np.fromfile(dataFile.file, dtype=np.dtype('<H'), count=eventSize)
            ntrace = np.array(ntrace - np.mean(ntrace))*LSB_IR
            n_pulses.append( ntrace[0:NN_SAMPLES] )
            
            i+=1

        except:
            print(i)
            break
    
    fig, ax = plt.subplots(2, sharex=False)
    ax[0].set_title(r'$ \bf Signal \ vs \ Model \ NSD $')
    for n_trace_ in n_pulses:
        x_n_trace_ = np.array(range( len(n_trace_) ))*TS
        ax[0].plot( x_n_trace_, n_trace_)
        
        xfft_, yfft_, v_ = lbs.ss_fft( n_trace_, TS, TYPE='nsd', HANN=FFT_WINDOW)
        yfft_flt_ = lbs.filter_butter_lp(yfft_, TS, N=1, fcutoff=4e6, FF=True)
        ax[1].plot( xfft_, yfft_flt_ )
            
        xfft_, yfft_, v_ = lbs.ss_fft(n_trace_, TS, TYPE='nsd', HANN=FFT_WINDOW)
        for (tp, nsd_nm_) in zip(TP, nsd_arr):
            xtri_, trace_tf_, isig_, tri_ = lbs.trig_filter( tp=tp, isig=n_trace_, fhpf=fhpf, TS=TS, HPF=HPF )
            ax[0].plot( x_n_trace_, trace_tf_ )
            
            xfft_tf_, yfft_tf_, v_ = lbs.ss_fft(trace_tf_, TS, TYPE='nsd', HANN=FFT_WINDOW)
            yfft_tf_flt_ = lbs.filter_butter_lp(yfft_tf_, TS, N=1, fcutoff=4e6, FF=True)
            
            ax[1].plot( xfft_tf_, yfft_tf_flt_, linewidth=0.05 )
            ax[1].plot( XF,  nsd_nm_, '--', linewidth=1)
    
    ax[1].set_xlim([1e3, 1e8])
    ax[1].set_ylim([1e-13, 1e-9])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[0].set_xlabel(' Time [s] ')
    ax[0].set_ylabel(' Amplitude [A] ')
    ax[1].set_xlabel(' Frequency [Hz] ')
    ax[1].set_ylabel(' IR NSD [A/sqrt(Hz)] ')
    
    fig.savefig('fig'+PREFIX+'_nsd.png', dpi=600,facecolor=(1, 1, 1))
    
    dataFile.close()
    del dataFile
    
    input('Press [ENTER] to finish.')
    exit(0)
