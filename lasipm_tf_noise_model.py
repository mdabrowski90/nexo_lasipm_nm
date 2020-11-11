# by M. Dabrowski 08/14/2020
# Large-Area SiPM triangular filter noise model.
# This script computes ENC in peaking time for various SiPM arrangements.
# Developed within the nEXO project. For usage outside of nEXO, please contact the author at mdabrowski@bnl.gov

import numpy as np
import matplotlib.pyplot as plt

import csv

from mdplt import MDPLT

# Large-Area SiPM Noise Model
class LASIPM_NM:

    # CONSTANTS - used for computation of ballistic deficit
    NS = 100000     # Number of samples
    TS = 2.5e-10    # Time step between samples (the smaller the number, the more accurate the BD)
    
    # PHYSICS CONSTANTS
    k = 1.38e-23 # Boltzmann constant
    q = 1.6e-19  # Unit charge

    def __init__(self, TP, T, P, S, qspe=360e-15, cj=8.7e-9, rq=12, cq=330e-12, ic=393e-6, re=22, rbb=8, ib=2*200e-6, rp=380.8, XF = np.linspace(1, 10e9, 100000)):
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

        # Define integration region in F
        self.XF = XF

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

    def get_pn_in_f(self, tp=1e-6, no_filter=False, f0_f1=[1e7, 1e8]):
        # f0_f1 - Post Amplifier's poles - to remove it's influence set the poles to very high frequency
        post_a_f = lambda f: 1/np.sqrt((1+ (f/f0_f1[0])**2))/np.sqrt((1+ (f/f0_f1[1])**2))
        if no_filter:
            return [ np.sqrt( self.ip_dn2 )*post_a_f(f) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ 1/tp* np.sqrt( self.ip_dn2 * self.abs_sq_sig_tf(f, tp) ) *post_a_f(f)  for f in self.XF ]

    def get_sn_in_f(self, tp=1e-6, no_filter=False, f0_f1=[1e7, 1e8]):
        # f0_f1 - Post Amplifier's poles - to remove it's influence set the poles to very high frequency
        post_a_f = lambda f: 1/np.sqrt((1+ (f/f0_f1[0])**2))/np.sqrt((1+ (f/f0_f1[1])**2))
        if no_filter:
            return [ np.sqrt( self.vs_dn2 * self.abs_sq_sn_tf(f) )* post_a_f(f) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ 1/tp* np.sqrt( self.vs_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_sn_tf(f) * post_a_f(f) ) for f in self.XF ]

    def get_sn_rq_in_f(self, tp=1e-6, no_filter=False, f0_f1=[1e7, 1e8]):
        # f0_f1 - Post Amplifier's poles - to remove it's influence set the poles to very high frequency
        post_a_f = lambda f: 1/np.sqrt((1+ (f/f0_f1[0])**2))/np.sqrt((1+ (f/f0_f1[1])**2))
        if no_filter:
            return [ np.sqrt( self.vsrq_dn2 *  self.abs_sq_snrq_tf(f) ) * post_a_f(f) for f in self.XF ]
        else:
            # For normalization purposes, the signal must be scaled by 1/tp
            return [ 1/tp* np.sqrt( self.vsrq_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_snrq_tf(f) ) * post_a_f(f) for f in self.XF ]

    def get_pn_sn_in_f(self, tp=1e-6, no_filter=False, f0_f1=[1e7, 1e8]):
        sn_in_f = self.get_sn_in_f(tp, no_filter, f0_f1)
        pn_in_f = self.get_pn_in_f(tp, no_filter, f0_f1)
        return [ np.sqrt(sn_in_f[i]**2 + pn_in_f[i]**2 ) for i in range(0,len(sn_in_f))]
            
    def compute_enc(self):
        enc_pn_spe = []
        enc_sn_spe = []
        enc_snrq_spe = []
        enc_spe = []
        
        for tp in self.TP:
            
            print(str(self.P)+'p'+str(self.S)+'s'+' : Computing ENC for ALL NOISE SOURCES for TP: '+str(tp))
            
            ypt_pn = []  # Y-point
            ypt_sn = []
            ypt_snrq = []
            
            for f in self.XF:
                ypt_pn.append( self.ip_dn2 * self.abs_sq_sig_tf(f, tp) )
                ypt_sn.append( self.vs_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_sn_tf(f) )
                ypt_snrq.append( self.vsrq_dn2 * self.abs_sq_sig_tf(f, tp) *  self.abs_sq_snrq_tf(f) )
                
            enc_pn_spe_   = np.sqrt( np.trapz(ypt_pn, self.XF) ) / self.qspe
            enc_sn_spe_   = np.sqrt( np.trapz(ypt_sn, self.XF) ) / self.qspe
            enc_snrq_spe_ = np.sqrt( np.trapz(ypt_snrq, self.XF) ) / self.qspe
            
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

    # Back-of-envelope ENC calculation
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

    # Generates Triangular weighting function w/ a given peaking time
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

    je_fnames = ['je_data/200820_enc-triangFilter-2s.csv',
                 'je_data/200820_enc-triangFilter-4s.csv',
                 'je_data/200820_enc-triangFilter-6s.csv']

    je_data = []
    for fname in je_fnames:
        je_data.append(extract_je_data(fname))    

    # Step 0: CREATE ARRAY WITH PEAKING TIMES (TP)
    TP = np.logspace(-8, -5, 16)

    # (optional) CREATE ARRAY WITH FREQUENCY SPECTRUM OVER WHICH THE NOISE IS INTEGRTATED (default one can be found in __init__() of LASIPM_NM )
    XF = np.linspace(1, 1e9, 100000)

    # STEP1: CREATE OBJECT WITH CORRECT SiPM ARRANGEMENT
    lasipm_nm_2s = LASIPM_NM(TP, 168, 1, 2, rp=50, XF=XF)
    lasipm_nm_2p2s = LASIPM_NM(TP, 168, 2, 2, rp=50, XF=XF)
    lasipm_nm_3p2s = LASIPM_NM(TP, 168, 3, 2, rp=50, XF=XF)
    lasipm_nm_6p = LASIPM_NM(TP, 168, 6, 1, rp=50, XF=XF)

    # STEP2: RUN OBJECT'S METHOD: compute_enc(), WHICH RETURNS ENC ARRAY
    enc_2s = lasipm_nm_2s.compute_enc()
    enc_2p2s = lasipm_nm_2p2s.compute_enc()
    enc_3p2s = lasipm_nm_3p2s.compute_enc()

    # PLOTTING NOISE SPECTRUMS

    # To observe how spectrum should look like with only a capacitance at the input (instead of a real SiPM):
    # (1) Create an object with S=1, P = 1 and cj = (desired capacitance value) and cq = 1 fF
    lasipm_nm_4p7 = LASIPM_NM(TP, 168, 1, 1, cj=4.7e-9, cq=1e-15, rp=50, XF=XF) # Example with 4.7 nF
    lasipm_nm_12 = LASIPM_NM(TP, 168, 1, 1, cj=12e-9, cq=1e-15, rp=50, XF=XF) # Example with 12 nF
    
    # (2) Turn off the SiPM option
    lasipm_nm_4p7.INCL_SIPM_ZIN = False
    lasipm_nm_12.INCL_SIPM_ZIN = False
    
    # (3) Plot noise spectrums using get_pn_in_f(), get_sn_in_f() and get_pn_sn_in_f() functions
    # (optional) To apply triangular filtering: set no_filter=False while providing an appropriate peaking time
    # To modify poles of the post-amplifier set f0_f1=[pole0, pole1]
    plt.ion()
    plt.figure()
    plt.plot(XF, lasipm_nm_4p7.get_pn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Parallel noise spectrum - 4.7 nF')
    plt.plot(XF, lasipm_nm_4p7.get_sn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Series noise spectrum - 4.7 nF')
    plt.plot(XF, lasipm_nm_4p7.get_pn_sn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Parallel+Series noise spectrum - 4.7 nF')
    plt.plot(XF, lasipm_nm_12.get_pn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Parallel noise spectrum - 12 nF')
    plt.plot(XF, lasipm_nm_12.get_sn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Series noise spectrum - 12 nF')
    plt.plot(XF, lasipm_nm_12.get_pn_sn_in_f(no_filter=True, f0_f1=[5e6, 2e7]), label='Parallel+Series noise spectrum - 12 nF')
    plt.legend(frameon=False, loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.draw()

    input('Press [ENTER] to finish.')
    exit(0)


