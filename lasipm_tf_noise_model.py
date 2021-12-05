# by Mietek Dabrowski ( m.dabrowski@kuleuven.be )
# Date: 11/27/2021
# Large-Area SiPM Front-end noise model - MD-LASIPM FE Model
# This script computes ENC and other noise-related merits for triangular filter for various SiPM arrangements.
# Copyrights M. Dabrowski @ KU Leuven. When using or copying the script or parts of it, please, acknowledge/reference the author and the KU Leuven University.

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# MD Libs
import lasipm_base_sp as lbs

# Large-Area SiPM Noise Model
class LASIPM_NM:

    # CONSTANTS - used for computation of ballistic deficit
    NS = 1000000     # Number of samples
    TS = 1e-9      # Time step between samples (the smaller the number, the more accurate the BD)
    
    # PHYSICS CONSTANTS
    k = 1.38e-23 # Boltzmann constant
    q = 1.6e-19  # Unit charge

    def __init__(self, TP, T, P, S, qspe=360e-15, cj=8.7e-9, rq=12, cq=330e-12, ic=393e-6, re=22, rbb=8, ib=2*200e-6, rp=380.8, kf2 = 0, f0_f1=[1e7, 1e8], fhpf=1e5, XF = np.linspace(1, 10e9, 100000)):
        self.TP = TP        # An array w/ peaking times for which the ENC is to be evalueated

        self.P = P          # Number of SiPMs in parallel
        self.S = S          # Number of SiPMs in series
        
        self.T = T          # Measurement temperature
        self.vt = self.k*T/self.q

        self.qspe = qspe/S  # Single-photo-electron charge

        self.cj = cj*(P/S)  # Junction capacitance
        self.cq = cq*(P/S)  # Quenching capacitance
        self.rq = rq*(S/P)  # Quenching resistance

        self.re = re        # Resistance at the emmiter node of the input transistor
        self.rbb = rbb      # Input-transistor base resistance

        # Compute noise power spectral densities for parallel and series noise sources
        # Parallel noise. Includes: The shot noise from transistors and 
        self.ip_dn2 = 2*self.q*ib + 4*self.k*self.T/rp
        self.vs_dn2 = 2*self.k*self.T*self.vt/ic + 4*self.k*self.T*self.re + 4*self.k*self.T*self.rbb
        self.vsrq_dn2 = 4*self.k*self.T*self.rq
        
        self.ip_kf2 = kf2   # 1/(f**2) noise
        
        # Post-amplifier pole frequencies
        self.f0_f1 = f0_f1
        self.fhpf = fhpf
        
        # Define integration region in F
        self.XF = XF

        # Switches
        self.INCL_SIPM_ZIN = True
        self.FILTER_SIGNAL = True
        self.SIPM_SIGNAL = True
        self.POST_AMP = True
        self.HPF = False

        # Plot switches
        self.PLOT_COMPUTE_ENC = False
        
    def abs_sq_sig_tf(self, f, tp):
        return np.sin(np.pi*f*tp)**4/( (f*np.pi)**4 * tp**4 ) * self.abs_sq_post_amp_tf(f) * self.abs_sq_hpf_tf(f)
    
    def abs_sq_sn_tf(self, f):
        if self.INCL_SIPM_ZIN:
            return (2*np.pi*f*self.cj)**2 * 1/(1+ (2*np.pi*f*self.rq*(self.cq+self.cj))**2) * (1+(2*np.pi*f*self.cq*self.rq)**2)
        else:
            return (2*np.pi*f*self.cj)**2
    
    def abs_sq_snrq_tf(self, f):
        if self.INCL_SIPM_ZIN:
            return (2*np.pi*f*self.cj)**2 * 1/(1+ (2*np.pi*f*self.rq*(self.cq+self.cj))**2)
        else:
            return (2*np.pi*f*self.cj)**2
    
    def abs_sq_fn2_tf(self, f):
        return 1/(f**2)
    
    def abs_sq_post_amp_tf(self, f):
        if self.POST_AMP:
            return 1 / (1+ (f/self.f0_f1[0])**2) / (1+ (f/self.f0_f1[1])**2)
        else:
            return 1
    
    def abs_sq_hpf_tf(self, f):
        if self.HPF:
            return (f/self.fhpf)**2 / (1+ (f/self.fhpf)**2)
        else:
            return 1
    
    def get_pn_in_f(self, tp=1e-6, no_filter=False):
        # No triang filter, with post-amplifier
        if no_filter:
            return [ np.sqrt( self.ip_dn2 * self.abs_sq_post_amp_tf(f) ) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ np.sqrt( self.ip_dn2 * self.abs_sq_sig_tf(f, tp) ) for f in self.XF ]
    
    def get_sn_in_f(self, tp=1e-6, no_filter=False):
        # No triang filter, with post-amplifier
        if no_filter:
            return [ np.sqrt( self.vs_dn2 * self.abs_sq_sn_tf(f) * self.abs_sq_post_amp_tf(f) ) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ np.sqrt( self.vs_dn2 *  self.abs_sq_sn_tf(f) * self.abs_sq_sig_tf(f, tp) ) for f in self.XF ]
    
    def get_sn_rq_in_f(self, tp=1e-6, no_filter=False):
        # No triang filter, with post-amplifier
        if no_filter:
            return [ np.sqrt( self.vsrq_dn2 *  self.abs_sq_snrq_tf(f) * self.abs_sq_post_amp_tf(f) ) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ np.sqrt( self.vsrq_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_snrq_tf(f) ) for f in self.XF ]
    
    def get_pn_sn_in_f(self, tp=1e-6, no_filter=False, f0_f1=[1e7, 1e8]):
        sn_in_f = self.get_sn_in_f(tp, no_filter, f0_f1)
        pn_in_f = self.get_pn_in_f(tp, no_filter, f0_f1)
        return [ np.sqrt(sn_in_f[i]**2 + pn_in_f[i]**2 ) for i in range(0,len(sn_in_f))]
    
    def compute_enc(self):
        enc_pn_spe = []
        enc_sn_spe = []
        enc_snrq_spe = []
        enc_fn2_spe = []
        enc_spe = []
        
        amax    = []
        qn_pn   = []
        qn_sn   = []
        qn_snrq = []
        qn_fn2 = []

        ypt_dn = []
        tp_extr = []

        if self.PLOT_COMPUTE_ENC:
            fig, ax = subplots(4, sharex=False)
        
        for tp in self.TP:
            
            print(str(self.P)+'p'+str(self.S)+'s'+' : Computing ENC for ALL NOISE SOURCES for TP: '+str(tp))
            
            ypt_pn_dn2 = []  # Y-point
            ypt_sn_dn2 = []
            ypt_snrq_dn2 = []
            ypt_fn2_dn2 = []
            
            for f in self.XF:
                ypt_pn_dn2.append( self.ip_dn2 * self.abs_sq_sig_tf(f, tp) )
                ypt_sn_dn2.append( self.vs_dn2 *  self.abs_sq_sn_tf(f) * self.abs_sq_sig_tf(f, tp) )
                ypt_snrq_dn2.append( self.vsrq_dn2 *  self.abs_sq_snrq_tf(f) * self.abs_sq_sig_tf(f, tp) )
                ypt_fn2_dn2.append( self.ip_kf2 * self.abs_sq_fn2_tf(f) * self.abs_sq_sig_tf(f, tp) )

            ypt_dn2_ = np.sqrt( [sum(y) for y in zip(ypt_pn_dn2, ypt_sn_dn2, ypt_snrq_dn2, ypt_fn2_dn2)] )
            ypt_dn.append( ypt_dn2_ )
            
            qn_pn_ = np.sqrt( np.trapz(ypt_pn_dn2, self.XF) )
            qn_sn_ = np.sqrt( np.trapz(ypt_sn_dn2, self.XF) )
            qn_snrq_ = np.sqrt( np.trapz(ypt_snrq_dn2, self.XF) )
            qn_fn2_ = np.sqrt( np.trapz(ypt_fn2_dn2, self.XF) )
            
            qn_pn.append( qn_pn_ )
            qn_sn.append( qn_sn_ )
            qn_snrq.append( qn_snrq_ )
            qn_fn2.append( qn_fn2_ )
            
            enc_pn_spe_   = qn_pn_ / self.qspe
            enc_sn_spe_   = qn_sn_ / self.qspe
            enc_snrq_spe_ = qn_snrq_ / self.qspe
            enc_fn2_spe_ = qn_fn2_ / self.qspe
            
            amax_, x_, sig_, sipm_sig_, tri_ = self.filter_signal(tp)
            isipm_sig = np.trapz(sipm_sig_, x=x_)
            amax_ = amax_/isipm_sig
            
            amax.append( amax_ )

            tp_extr_, ix_ = self.extract_tp(x_, sig_)
            tp_extr.append( tp_extr_ )
            
            if self.PLOT_COMPUTE_ENC:
                xfft_, yfft_, v_ = lbs.ss_fft(sig_, x_[1]-x_[0], TYPE='normal', HANN=False)
                ax[0].plot( x_, tri_ )
                ax[1].plot( x_, sipm_sig_ )
                ax[2].plot( x_, sig_ )
                ax[2].plot( [x_[ix_[0]], x_[ix_[1]]], [sig_[ix_[0]], sig_[ix_[1]]], '-o', color='darkorange', markersize=2 )
                ax[3].plot( xff_t, yfft_ )

            if self.FILTER_SIGNAL:
                enc_pn_spe_   = enc_pn_spe_ / amax_
                enc_sn_spe_   = enc_sn_spe_ / amax_
                enc_snrq_spe_ = enc_snrq_spe_ / amax_
                enc_fn2_spe_ = enc_fn2_spe_ / amax_
                
            # Partial ENCs
            enc_pn_spe.append( enc_pn_spe_ )
            enc_sn_spe.append( enc_sn_spe_ )
            enc_snrq_spe.append( enc_snrq_spe_ )
            enc_fn2_spe.append( enc_fn2_spe_ )
            
            # Total ENC
            enc_spe.append( (enc_pn_spe_**2 + enc_sn_spe_**2 + enc_snrq_spe_**2 + enc_fn2_spe_**2)**(1/2) )
        
        if self.PLOT_COMPUTE_ENC:
            ax[3].set_xscale('log')
            ax[3].set_yscale('log')

        print('\n')
        return [enc_spe, enc_pn_spe, enc_sn_spe, enc_snrq_spe, enc_fn2_spe], tp_extr, amax, ypt_dn

    def extract_tp(self, xdata, ydata, th=0.01):
        ymax_ = np.amax(ydata)
        ix0, v_ = lbs.data_find_nearest_index(ydata, x=ymax_)
        ix1, v_ = lbs.data_find_nearest_index(ydata[0:int(len(ydata)/2)], x=ymax_*th)
        return xdata[ix0]-xdata[ix1], [ix0, ix1]
        
    # Back-of-envelope ENC calculation
    def formula_enc(self, aw=1, ap=1/3):
    
        enc_pn = []
        enc_sn = []
        enc_snrq = []
        enc = []
    
        for tp in self.TP:
        
            enc_pn_ = np.sqrt( ap * tp * self.ip_dn2 ) / self.qspe
            enc_sn_ = np.sqrt( self.cj**2 * aw * 1/tp * self.vs_dn2 ) / self.qspe
            enc_snrq_ = np.sqrt( self.cj**2 * aw * 1/tp * self.vsrq_dn2 ) / self.qspe

            enc_pn.append(enc_pn_)
            enc_sn.append(enc_sn_)
            enc_snrq.append(enc_snrq_)
            enc.append( np.sqrt(enc_pn_**2 + enc_sn_**2 + enc_snrq_**2) )

        return enc, enc_pn, enc_sn, enc_snrq
    
    def filter_signal(self, tp):
        x, sig, sipm_sig, tri = lbs.trig_filter(tp = tp, isig = None, tau = self.cj*self.rq,
                                                f0_f1 = self.f0_f1, fhpf = self.fhpf, TS = self.TS, NS = self.NS,
                                                SIPM_SIGNAL = self.SIPM_SIGNAL, POST_AMP = self.POST_AMP, HPF = self.HPF)
        
        return np.amax(sig), x, sig, sipm_sig, tri
    
    def set_incl_sipm_zin(self):
        self.INCL_SIPM_ZIN = True
    
    def unset_incl_sipm_zin(self):
        self.INCL_SIPM_ZIN = False
    
    def set_filter_signal(self):
        self.FILTER_SIGNAL = True
    
    def unset_filter_signal(self):
        self.FILTER_SIGNAL = False
    
    def set_sipm_signal(self):
        self.SIPM_SIGNAL = True
    
    def unset_sipm_signal(self):
        self.SIPM_SIGNAL = False
    
    def set_post_amp(self):
        self.POST_AMP = True
    
    def unset_post_amp(self):
        self.POST_AMP = False
    
    def set_hpf(self):
        self.HPF = True
    
    def unset_hpf(self):
        self.HPF = False

    def set_plot(self):
        self.PLOT_COMPUTE_ENC = True
    
    def unset_plot(self):
        self.PLOT_COMPUTE_ENC = False
    
## ################## ##
##     MAIN BODY      ##
## ################## ##

if __name__ == "__main__":
    # For examples of how to implement the model, please, refer to lasipm_data_processor.py
    pass
