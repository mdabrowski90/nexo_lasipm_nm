# by M. Dabrowski 08/14/2020
# Large-Area SiPM triangular filter noise model.
# This script computes ENC in peaking time for various SiPM arrangements.
# Developed within the nEXO project. For usage outside of nEXO, please contact the author at mdabrowski@bnl.gov

import numpy as np
import matplotlib.pyplot as plt

import csv

# Large-Area SiPM Noise Model
class LASIPM_NM:

    # CONSTANTS
    NS = 100000
    TS = 2.5e-10
    
    # PHYSICS CONSTANTS
    k = 1.38e-23 # Boltzmann constant
    q = 1.6e-19  # Unit charge

    def __init__(self, TP, T, P, S, qspe=360e-15, cj=8.7e-9, rq=12, cq=330e-12, ic=393e-6, re=22, rbb=8, ib=2*200e-6, rp=380.8):
        self.TP = TP

        self.P = P
        self.S = S
        
        self.T = T
        self.vt = self.k*T/self.q

        self.qspe = qspe/S

        self.cj = cj*(P/S)
        self.cq = cq*(P/S)
        self.rq = rq*(S/P)

        # Emitter resistance
        self.re = re

        # Input transistor base resistance
        self.rbb = rbb

        # Compute noise power spectral densities for parallel and series noise sources
        # Parallel noise. Includes: The shot noise from transistors and 
        self.ip_dn2 = 2*self.q*ib + 4*self.k*self.T/rp
        self.vs_dn2 = 2*self.k*self.T*self.vt/ic + 4*self.k*self.T*self.re + 4*self.k*self.T*self.rbb
        self.vsrq_dn2 = 4*self.k*self.T*self.rq

        # Switches
        self.INCL_SIPM_ZIN = True
        self.BALLISTIC_DEFICIT = True

    def abs_sq_sig_tf(self, f, tp):
        return np.sin(np.pi*f*tp)**4/( (f*np.pi)**4 * tp**2 )

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
    
    def compute_enc(self):
        # Specification of limits for integration in frequency
        xf = np.logspace(0, 12, 12*1000)

        enc_pn_spe = []
        enc_sn_spe = []
        enc_snrq_spe = []
        enc_spe = []
        
        for tp in self.TP:
            
            print(str(self.P)+'p'+str(self.S)+'s'+' : Computing ENC for ALL NOISE SOURCES for TP: '+str(tp))
            
            ypt_pn = []  # Y-point
            ypt_sn = []
            ypt_snrq = []
            
            for f in xf:
                ypt_pn.append( self.ip_dn2 * self.abs_sq_sig_tf(f, tp) )
                ypt_sn.append( self.vs_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_sn_tf(f) )
                ypt_snrq.append( self.vsrq_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_snrq_tf(f) )
                
            enc_pn_spe_   = np.sqrt( np.trapz(ypt_pn, xf) ) / self.qspe
            enc_sn_spe_   = np.sqrt( np.trapz(ypt_sn, xf) ) / self.qspe
            enc_snrq_spe_ = np.sqrt( np.trapz(ypt_snrq, xf) ) / self.qspe
            
            if self.BALLISTIC_DEFICIT:
                enc_pn_spe_   = enc_pn_spe_ / self.ballistic_deficit(tp)
                enc_sn_spe_   = enc_sn_spe_ / self.ballistic_deficit(tp)
                enc_snrq_spe_ = enc_snrq_spe_ / self.ballistic_deficit(tp)

            # Partial ENCs
            enc_pn_spe.append( enc_pn_spe_ )
            enc_sn_spe.append( enc_sn_spe_ )
            enc_snrq_spe.append( enc_snrq_spe_ )
            
            # Total ENC
            enc_spe.append(np.sqrt(enc_pn_spe_**2 + enc_sn_spe_**2 + enc_snrq_spe_**2))

        print('\n')
        return enc_spe, enc_pn_spe, enc_sn_spe, enc_snrq_spe

    
    def formula_enc(self, aw=1, ap=1/3):
    
        enc_pn = []
        enc_sn = []
        enc_snrq = []
        enc = []
    
        for tp in self.TP:
        
            enc_pn_ = np.sqrt( ap * tp * self.ip_dn2 ) / self.qspe
            enc_sn_ = np.sqrt( self.cj**2 * aw * 1/tp * self.vs_dn2 ) / self.qspe
            enc_snrq_ = np.sqrt( self.cj**2 * aw * 1/tp * 4*self.k*self.T*self.rq ) / self.qspe

            enc_pn.append(enc_pn_)
            enc_sn.append(enc_sn_)
            enc_snrq.append(enc_snrq_)
            enc.append( np.sqrt(enc_pn_**2 + enc_sn_**2 + enc_snrq_**2) )

        return enc_pn, enc_sn, enc_snrq, enc
    
    # Dummy time-domain SiPM signal
    def gen_sipm_sig(self):
        
        X = [ (x+1)*self.TS for x in range(-int(np.floor(self.NS/2)), int(np.floor(self.NS/2)))]

        tau = self.cj*self.rq
        
        sig = []
    
        for x in X:
            if x < 0:
                sig.append(0)
            else:
                sig.append(np.exp(-x/tau))
                
        return X, sig

    # Generates Triangular weighting function w/ a certain peaking time
    def gen_triang_wf(self, tp):

        X = [ (x+1)*self.TS for x in range(-int(np.floor(self.NS/2)), int(np.floor(self.NS/2)))]
        
        tri_ = []
        tri_.append(1)
    
        for i in range(1, int(np.floor(self.NS/2)) ):
            if i*self.TS<tp:
                y = (tp - i*self.TS)/tp
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

    # Returns X and Y values
    def trig_filter(self,tp):
        # NS - number of samples
        # TS - time step between samples
        # tp - filter peaking time
        X, sipm_sig = self.gen_sipm_sig()
        X, tri = self.gen_triang_wf(tp)
        return X, np.convolve(tri, sipm_sig, mode='same')/np.sum(sipm_sig), tri

    def ballistic_deficit(self, tp):
        x, sig, tri = self.trig_filter(tp)
        return np.amax(sig)
    
    def set_incl_sipm_zin(self):
        self.INCL_SIPM_ZIN = True

    def unset_incl_sipm_zin(self):
        self.INCL_SIPM_ZIN = False

    def set_ballistic_deficit(self):
        self.BALLISTIC_DEFICIT = True

    def unset_ballistic_deficit(self):
        self.BALLISTIC_DEFICIT = False

def extract_je_data(fname):
    csvdata = csv.reader(open(fname, 'r'), delimiter=',')
    # Skip first row
    #csvdata = next(csvdata)
    next(csvdata)
    
    data = []
    data.append([])
    data.append([])
    
    for row in csvdata:
        data[0].append(float(row[1]))
        data[1].append(float(row[2]))
        
    return data
        
## ################## ##
##     MAIN BODY      ##
## ################## ##

if __name__ == "__main__":

    je_fnames = ['je_data/200730_enc-gaussFilter-noAP-2s.csv',
                 'je_data/200730_enc-gaussFilter-noAP-4s.csv',
                 'je_data/200730_enc-gaussFilter-noAP-6s.csv']

    je_data = []
    for fname in je_fnames:
        je_data.append(extract_je_data(fname))    

    # Step 0: CREATE ARRAY WITH PEAKING TIMES (TP)
    TP = np.logspace(-8, -5, 16)

    # STEP1: CREATE OBJECT WITH CORRECT SiPM ARRANGEMENT
    lasipm_nm_2s = LASIPM_NM(TP, 168, 1, 2)
    lasipm_nm_2p2s = LASIPM_NM(TP, 168, 2, 2)
    lasipm_nm_3p2s = LASIPM_NM(TP, 168, 3, 2)

    # STEP2: RUN OBJECT'S METHOD: compute_enc(), WHICH RETURNS ENC ARRAY
    enc_2s = lasipm_nm_2s.compute_enc()
    enc_2p2s = lasipm_nm_2p2s.compute_enc()
    enc_3p2s = lasipm_nm_3p2s.compute_enc()

    # STEP3: PLOT DATA
    plt.ion()
    plt.figure()
    plt.title('ENC in Tp for 2s')
    plt.plot(je_data[0][0], je_data[0][1], '-o', markersize=2, label='Measured data - Gaussian Filter')
    plt.plot(TP, enc_2s[0], '-o', markersize=1.25, label='Noise model - Triangular Filter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(frameon=False, loc='upper right')
    plt.draw()

    plt.figure()
    plt.title('ENC in Tp for 2p2s')
    plt.plot(je_data[1][0], je_data[1][1], '-o', markersize=2, label='Measured data - Gaussian Filter')
    plt.plot(TP, enc_2p2s[0], '-o', markersize=1.25, label='Noise model - Triangular Filter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(frameon=False, loc='upper right')
    plt.draw()

    plt.figure()
    plt.title('ENC in Tp for 3p2s')
    plt.plot(je_data[2][0], je_data[2][1], '-o', markersize=2, label='Measured data - Gaussian Filter')
    plt.plot(TP, enc_3p2s[0], '-o', markersize=1.25, label='Noise model - Triangular Filter')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(frameon=False, loc='upper right')
    plt.draw()
    
    input('Press [ENTER] to finish.')
