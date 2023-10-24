import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
import time
from pycbc.waveform import get_td_waveform
from pycbc.waveform import utils
from scipy.stats import norm
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.cm as cm

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)


'''
Class handling noise generation

Option:

->Ttot    : noise samples duration, in seconds (Default is 1)
->fe      : sampling frequencies, in Hz (Default is 2048)
->kindPSD : noise type: 'flat' or 'analytic' (Default is g. flat). Noise is stationary in both cases (see below)
->fmin    : minimal frequency for noise definition (Def = 20Hz)
->fmax    : maximal frequency for noise definition (Def = 1500Hz)
->nsamp   : for each frequency the PSD is considered as a gaussian centered on PSD
            of width PSD/sqrt(nsamp), number of samples used to estimate the PSD

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution
            still it's not based on real data, ie not including glitches for example.
            So that's gaussian colored noise
            Reference used for analytic noise is cited below

Noise is produced over a given frequency range.

Indeed there is no reason to produce noise well outside
detector acceptance

'''

class GenNoise:

    def __init__(self,Ttot=1,fe=2048,kindPSD='flat',nsamp=160,fmin=20,fmax=1500,verbose=False):

        if not((isinstance(Ttot,int) or isinstance(Ttot,float) or isinstance(Ttot,list)) and (isinstance(fe,int) or isinstance(fe,float) or isinstance(fe,list))):
            raise TypeError("Ttot et fe doivent être des ints, des floats, ou des list")
        if not(isinstance(kindPSD,str)):
            raise TypeError("kindPSD doit être de type str")
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées pour kindPSD sont 'flat' et 'analytic'")

        # Deal with the fact that we can sample the frame with different frequencies
        if isinstance(Ttot,list):
            if not isinstance(fe,list):
                raise TypeError("Ttot et fe doivent être du même type")
            elif not len(Ttot)==len(fe):
                raise ValueError("Les list Ttot et fe doivent faire la même taille")
            else:
                self.__listTtot=Ttot           # List of chunk lengths
                self.__listfe=fe               # List of corresponding sampling freqs
                self.__Ttot=sum(Ttot)          # Total sample length
                self.__fe=max(fe)              # Max sampling freq
                self.__nTtot=len(Ttot)         # Total number of subsamples
        else:
            self.__Ttot=Ttot                   # Total sample length
            self.__fe=fe                       # Sampling freq
            self.__nTtot=1
        
        # We will generate a sample with the total length and the max sampling freq, and resample
        # only at the end
        
        self.__verb=verbose
        self.__fmin=fmin
        self.__fmax=fmax
        self.__N=int(self.__Ttot*self.__fe)    # The total number of time steps produced
        self.__delta_t=1/self.__fe             # Time step
        self.__delta_f=self.__fe/self.__N      # Frequency step
        self.__kindPSD=kindPSD                 # PSD type
        self.__nsamp=nsamp                     # Noise width per freq step.

        # N being defined we can generate all the necessary vectors
        
        self.__norm=npy.sqrt(self.__N)

        self.__T=npy.arange(self.__N)*self.__delta_t  # Time values
        
        # Frequencies (FFT-friendly)
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        
        self.__PSD=npy.ones(self.__N, dtype=npy.float64) # Setting the PSD to one means infinite noise
        
        # Then we produce the PSD which will be use to generate the noise in freq range.
        self._genPSD()

        self.__Nf=npy.zeros(self.__N,dtype=complex)          # Noise FFT
        self.__Nf2=npy.zeros(self.__N,dtype=complex)         # Noise FFT (whitened if required)
        self.__Nfr=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT real part
        self.__Nfi=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT imaginary part

            
        if self.__verb:
            print("____________________________")
            print("Noise generation")
            print("____________________________")

        
    '''
    NOISE 1/8
    
    Noise generation for analytic option, account for shot, thermal, quantum and seismic noises
    This is the one sided PSD, as defined in part IV.A of:
    https://arxiv.org/pdf/gr-qc/9301003.pdf
    '''
    def Sh(self,f):
        
        ##Shot noise (Eq 4.1)
        hbar=1.05457182e-34 #m2kg/s
        lamda=5139e-10 #m
        etaI0=60 #W
        Asq=2e-5
        L=4e3 #m
        fc=100 #Hz
        
        Sshot=(hbar*lamda/etaI0)*(Asq/L)*fc*(1+(f/fc)**2)
        
        ##Thermal Noise (Eq 4.2 to 4.4)
        kb=1.380649e-23 #J/K
        T=300 #K
        f0=1 #Hz
        m=1000 #kg
        Q0=1e9
        Lsq=L**2
        fint=5e3 #Hz
        Qint=1e5
        Spend=kb*T*f0/(2*(npy.pi**3)*m*Q0*Lsq*((f**2-f0**2)**2+(f*f0/Q0)**2))
        Sint=2*kb*T*fint/((npy.pi**3)*m*Qint*Lsq*((f**2-fint**2)**2+(f*fint/Qint)**2))
    
        Sthermal=4*Spend+Sint
        
        #Seismic Noise (Eq 4.6)
        S0prime=1e-20 #Hz**23
        f0=1 #Hz
        with npy.errstate(divide='ignore'):
            Sseismic=npy.where((f!=f0) & (f!=0),S0prime*npy.power(f,-4)/(f**2-f0**2)**10,(1e-11)**2)
            
        #Quantum noise (Eq 4.8)
        with npy.errstate(divide='ignore'):
            Squant=npy.where((f!=0),8*hbar/(m*Lsq*(2*npy.pi*f)**2),(1e-11)**2)
        
        return (Squant+Sshot+Sthermal+Sseismic)
    
    '''
    NOISE 2/8
    
    Produce the PSD, so in the noise power in frequency domain
    We don't normalize it, so be carefuf if the signal is not whitened
    '''

    def _genPSD(self):
    
        # Frequency range for the PSD
        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        
        # Generate the function
        if self.__kindPSD=='flat':
            sigma=2e-23
            self.__PSD[ifmin:ifmax]=sigma**2
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2] # Double sided
        elif self.__kindPSD=='analytic':
            self.__PSD[ifmin:ifmax]=self.Sh(abs(self.__F[ifmin:ifmax]))
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]

            
    '''
    NOISE 3/8
    
    PSD type change
    '''

    def changePSD(self,kindPSD):
        del self.__PSD
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées sont 'flat' et 'analytic'")
        self.__kindPSD=kindPSD
        self._genPSD()


    '''
    NOISE 4/8
    
    Create the noise in freq domain from the PSD

    We start from a PSD which provides us the power of the noise at a given frequency
    In order to create a noise ralisation, we need first to generate a random noise realisation of the noise
    in the frequency domain

    For each frequency we choose a random value of the power centered on the PSD value, we consider
    that power distribution is gaussian with a width equal to power/4 (rule of thumb, could be improved)

    Then when the power is chosen we choose a random starting phase Phi0 in order to make sure that the
    frequency component is fully randomized.

    Nf is filled like that:

    a[0] should contain the zero frequency term,
    a[1:n//2] should contain the positive-frequency terms,
    a[n//2+1:] should contain the negative-frequency terms,
    
    PSD and Nf(f)**2 are centered around PSD(f)

    '''
 
    def _genNfFromPSD(self):
    
        # The power at a given frequency is taken around the corresponding PSD value
        # We produce over the full frequency range
        self.__Nfr[0:self.__N//2+1]=npy.random.normal(npy.sqrt(self.__PSD[0:self.__N//2+1]),npy.sqrt(self.__PSD[0:self.__N//2+1]/self.__nsamp))
        self.__Nfi[0:self.__N//2+1]=self.__Nfr[0:self.__N//2+1]

        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        
        # The initial phase is randomized
        # randn provides a nuber following a normal law centered on 0 with sigma=1, so increase a
        # bit to make sure you cover all angular values (thus the factor 100)
        
        phi0=100*npy.random.randn(len(self.__Nfr))
        self.__Nfr*=npy.cos(phi0)
        self.__Nfi*=npy.sin(phi0)
        
        # Brutal filter
        self.__Nfr[ifmax:]=0.
        self.__Nfi[ifmax:]=0.
        self.__Nfr[:ifmin]=0.
        self.__Nfi[:ifmin]=0.
 
        # Then we can define the components
        self.__Nf[0:self.__N//2+1].real=self.__Nfr[0:self.__N//2+1]
        self.__Nf[0:self.__N//2+1].imag=self.__Nfi[0:self.__N//2+1]
        self.__Nf[-1:-self.__N//2:-1]=npy.conjugate(self.__Nf[1:self.__N//2])


    '''
    NOISE 5/8
    
    Get noise signal in time domain from signal in frequency domain (inverse FFT)

    https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

    If whitening option is set to true signal is normalized. The whitened noise should be a gaussian centered on 0 and of width 1.

    '''

    def _genNtFromNf(self,whitening):

        self.__Nt=[]                       # Noise in time domain
        
        for j in range(self.__nTtot):      # one array per chunk
            self.__Nt.append([])
        
        # Inverse FFT over the total length
        self.__Nt[0] = npy.fft.ifft(self.__Nf[:]/npy.sqrt(self.__PSD),norm='ortho').real if whitening else npy.fft.ifft(self.__Nf[:],norm='ortho').real
        
        self.__Nf2=npy.fft.fft(self.__Nt[0],norm='ortho') # Control
        

    '''
    NOISE 6/8
    
    Signal resampling
    '''
       
    def _resample(self):
        Ntref=self.__Nt[0] # The temporal raalization at max sampling
                    
        if self.__verb:
            print("Signal Nt has frequency",self.__fe,"and duration",self.__Ttot,"second(s)")

        for j in range(self.__nTtot):
            if self.__verb:
                print("Chunk",j,"has frequency",self.__listfe[j],"and covers",self.__listTtot[j],"second(s)")
            
            #Pick up the data chunk
            ndatapts=int(self.__listTtot[j]*self.__fe)
            nttt=len(Ntref)
            Nt=Ntref[-ndatapts:]
            Ntref=Ntref[:nttt-ndatapts]
            decimate=int(self.__fe/self.__listfe[j])
            self.__Nt[j]=Nt[::int(decimate)]
    
    
    '''
    NOISE 7/8
    
    The full procedure to produce a noise sample once the noise object has been instantiated
    '''
    
    def getNewSample(self,whitening=True):
        if not(isinstance(whitening,bool)):
            raise TypeError("Un booléen est attendu")
        self._genNfFromPSD()                    # Noise realisation in frequency domain
        self._genNtFromNf(whitening=whitening)  # Noise realisation in time domain
        if self.__nTtot > 1:                    # If requested, resample the data
            self._resample()
        return self.__Nt.copy()
   
   
    '''
    NOISE 8/8
    
    Plot macros
    '''
    
    # The main plot (noise in time domain)
    def plotNoise(self):
    
        listT=[] # Time of the samples accounting for the diffrent freqs
        if self.__nTtot > 1:
            maxT=self.__Ttot
            for j in range(self.__nTtot):
                delta_t=1/self.__listfe[j]
                N=int(self.__listTtot[j]*self.__listfe[j])
                listT.append(npy.arange((maxT-self.__listTtot[j])/delta_t,maxT/delta_t)*delta_t)
                maxT-=self.__listTtot[j]
        else:
            listT.append(self.__T)
        
        for j in range(self.__nTtot):
            plt.plot(listT[j], self.__Nt[j],'-',label=f"noise at {self.__listfe[j]} Hz")
        plt.title('Time domain noise realisation')
        plt.xlabel('t (s)')
        plt.ylabel('noise (no unit)')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        
    # The 1D projection (useful to check that noise has been correctly whitened
    def plotNoise1D(self,band):
        _, bins, _ = plt.hist(self.__Nt[band],bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__Nt[band])
        print("Look at frequency band:",self.__listfe[band],"Hz")
        print(f"Width: {sigma}")
        print(f"Mean: {mu}")
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line)
        plt.title(f'Time domain noise 1D projection for band at {self.__listfe[band]}Hz')

    # Frequency domain
    def plotTF(self,fmin=None,fmax=None):
    
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
             
    # Frequency domain whithened
    def plotTF2(self,fmin=None,fmax=None):
    
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain (normalized to PSD)')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
 
    # PSD
    def plotPSD(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__PSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.title('PSD')
        plt.xlabel('f (Hz)')
        plt.ylabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

        
    def getNf(self):
        return self.__Nf
        
    @property
    def kindPSD(self):
        return self.__kindPSD
        
    @property
    def PSD(self):
        return self.__PSD
        
    @property
    def length(self):
        return self.__N
        
    @property
    def T(self):
        return self.__T
    
    
#########################################################################################################################################################################################
'''
Class handling signal generation (templates)

Option:

->Tsample      : sampling durations, in s (Default is 1)
->fe           : sampling frequencies, in Hz (Default is 2048)
->kindTemplate : generator type: 'EM' or 'EOB' (Default is EM)
->fDmin        : the minimal sensitivity of the detector (Default is 15Hz)
->fDmax        : the maximal sensitivity of the detector (Default is 1000Hz)



'''

class GenTemplate:
    def __init__(self,Tsample=1,fe=2048,fDmin=15,fDmax=1000,kindTemplate='EM',verbose=False):
        if not (isinstance(fe,int) or isinstance(fe,float) or isinstance(fe,list)) and not (isinstance(fDmin,int) or isinstance(fDmin,float)):
            raise TypeError("fe et fDmin doivent être des ints ou des floats (fe peut aussi être une list)")
    
        if not(isinstance(kindTemplate,str)):
            raise TypeError("kindTemplate doit être de type str")
        if kindTemplate!='EM' and kindTemplate!='EOB':
            raise ValueError("Les seules valeurs autorisées pour kindTemplate sont 'EM' et 'EOB'")
        
        self.__type=1
        if kindTemplate!='EM':
            self.__type=0
        
        # We play the same trick than for the noise here
        # We will generate one big sample at max frequency and resample only afterwards
        # As this is for ML we just have to do this once
        
        if isinstance(fe,list):
            if not isinstance(Tsample,list):
                raise TypeError("Tsample et fe doivent être du même type")
            elif not len(Tsample)==len(fe):
                raise ValueError("Les list Ttot et fe doivent faire la même taille")
            else:
                self.__listTsample=Tsample
                self.__listfe=fe
                self.__Tsample=sum(Tsample)
                self.__fe=max(fe)
                self.__nTsample=len(self.__listTsample)
        else:
            self.__Tsample=Tsample
            self.__fe=fe
            self.__nTsample=1
        
        print(self.__fe)
        
        # Then we just instantiate some very basic params

        self.__fDmind=fDmin            # The minimum detectable frequency
        self.__fDmin=0.85*fDmin        # Minimum frequency with a margin (need it to smooth the template FFT)
        self.__fDmaxd=fDmax            # The maximum detectable frequency
        self.__delta_t=1/self.__fe     # Time step
        self.__Tdepass=0.1*self.__type # For the EM mode we add a small period after the chirp (to smooth FFT).
                                       # This is included in EOB by construction
        self.__m1=10*Msol              # m1,m2 in solar masses
        self.__m2=10*Msol              #
        self.__D=1*MPC                 # D in Mpc
        self.__Phic=0                  # Initial phase
        
        self.__kindTemplate=kindTemplate              # Template type

        # Parameters related to template SNR sharing
        self._tint=npy.zeros(self.__nTsample)
        self._fint=npy.zeros(self.__nTsample)
        self._currentSnr=npy.zeros(self.__nTsample)
        self._rawSnr=npy.zeros(self.__nTsample)
        self._evolSnrTime = []
        self._evolSnr = []
        self._evolSnrFreq = []
        self.__verb=verbose
        
 
    '''
    Template 1/10
    
    EM approximation calculations
    
    Here the time is computed with the equation (1.68) of the following document:

    http://sviret.web.cern.ch/sviret/docs/Chirps.pdf
    
    f0 is the frequency at which the detector start to be sensitive to the signal, we define t0=0
    The output is tc-t0
    Value is computed with the newtonian approximation, maybe we should change for EOB
    '''
  
    def phi(self,t):
        return -npy.power(2.*(self.__tc-t)/(5.*self.__tSC),5./8.)
                        
    def h(self,t):
        A=npy.power(2./5.,-1./4.)*self.__rSC/(32*self.__D)
        return A*npy.power((self.__tc-t)/self.__tSC,-1./4.)*npy.cos(2*(self.phi(t)+self.__Phic))
  
    def getTchirp(self,f0):
        return npy.power(125./128.,1./3.)*npy.power(self.__tSC,-5./3.)*npy.power(2*npy.pi*f0,-8./3.)/16.#pas terrible pour calculer tisco car w(t) etabli dans le cadre Newtonien
        
    def get_t(self,f0):#f frequence de l'onde grav omega/2pi --> si on souhaite frequence max dans le detecteur fmax ca correspond a fGW=fmax/2
        return self.__tc-self.getTchirp(f0)
        
    def get_f(self,delta_t):
        return npy.power(125./(128.*(16**3)),1./8.)*npy.power(self.__tSC,-5./8.)*npy.power(delta_t,-3./8.)/(2*npy.pi)
    
    
    '''
    Template 2/10
    
    Method updating the templates properties, should be called right after the initialization
    
    Info on template types:
    
    EOB uses the option SEOBNRv4, which includes the merger and ringdown stages
    EM is a simple EM equivalent model
    '''
    
    def majParams(self,m1,m2,D=None,Phic=None):
    
        # Start by updating the main params
        
        self.__m1=self.__m1 if m1 is None else m1*Msol
        self.__m2=self.__m2 if m2 is None else m2*Msol
        self.__D=self.__D if D is None else D*MPC
        self.__Phic=self.__Phic if Phic is None else Phic
        self.__M=self.__m1+self.__m2                        # Total mass
        self.__eta=self.__m1*self.__m2/(self.__M**2)        # Reduced mass
        self.__MC=npy.power(self.__eta,3./5.)*self.__M      # Chirp mass
        self.__rSC=2.*G*self.__MC/(c**2)                    # Schwarzchild radius of chirp
        self.__tSC=self.__rSC/c                             # Schwarzchild time of chirp
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M) # Innermost closest approach freq
        
        
        # Below we compute some relevant values for the EM approach
        
        # Chirp duration between fDmin and coalescence
        # We increase the vector size if larger than Tsample
      
        self.__Tchirp=max(self.getTchirp(self.__fDmin/2),self.__Tsample)
        
        # Duration in the detector acceptance (<Tchirp)
        self.__Tchirpd=self.getTchirp(self.__fDmind/2)
    
        # The length difference is not in the det acceptance
        # and can therefore be used to compute the blackman window
            
        self.__Tblack=self.__Tchirp-self.__Tchirpd  # End of Blackman window
        self.__Tblack_start=0                       # Start of Blackman window
    
        # Values for EOB templates are different here
        # We use SEOBnr approximant via pyCBC
        # https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
        #
        
        if self.__type==0:
            # The signal starting at frequency fDmin (~Tchirp)
            hp,hq = get_td_waveform(approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=self.__fDmin)

            # The signal starting at frequency fDmind (~Tchirpd)
            hpd,hqd = get_td_waveform(approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=self.__fDmind)

            # Remove 0's at the end
            c1=0
            for c1 in range(len(hp)-1,-1,-1): # Don't consider 0 at the end
                if abs(hp.numpy()[c1])>1e-35:
                    break
            hp_tab=hp.numpy()[:c1]
            
            c1=0
            for c1 in range(len(hpd)-1,-1,-1): # Don't consider 0 at the end
                if abs(hpd.numpy()[c1])>1e-35:
                    break
            hp_tabd=hpd.numpy()[:c1]

            # Here we have the correct values
            self.__Tchirp=max(len(hp_tab)*self.__delta_t,self.__Tsample)
            self.__Tchirpd=len(hp_tabd)*self.__delta_t
    
            # Blackman window is defined differently here
            # Because there is some signal after the merger for those templates
            
            self.__Tblack=self.__Tchirp-self.__Tchirpd
            self.__Tblack_start=self.__Tchirp-len(hp_tab)*self.__delta_t
            
        # The total length of signal to produce (and total num of samples)
        self.__Ttot=self.__Tchirp+self.__Tdepass
        N=int(self.__Ttot*self.__fe)
        self.__N=N+N%2

        self.__TFnorm=npy.sqrt(self.__N)
        self.__Ttot=float(self.__N)/float(self.__fe) # Total time produced
        self.__TchirpAndTdepass=self.__Ttot
        self.__delta_f=self.__fe/self.__N
        self.__tc=self.__Ttot-self.__Tdepass         # Where we put the end of template generated
        
        # Vectors
        
        self.__T=npy.arange(self.__N)*self.__delta_t
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        self.__St=npy.zeros(self.__N)
        self.__Stinit=npy.zeros(self.__N)
        self.__Sf=npy.zeros(self.__N,dtype=complex)
        self.__Sfn=npy.zeros(self.__N,dtype=complex)
        self.__Filtf=npy.zeros(self.__N,dtype=complex)
        self.__Filt=npy.zeros(self.__N)
        self.__norm=1.
        self.__Stfreqs=npy.arange(len(self.__Sf))*self.__delta_f
        
        if self.__verb:
            print("____________________________")
            print("Template generation")
            print("____________________________")
            print("")
            print(f"We will produce the signal for a CBC with masses ({m1},{m2})")
            print(f"Template type is {self.__kindTemplate}")
            print(f"Total length of signal produced is {self.__Ttot:.1f}s")
            print(f"Chirp duration in the det. acceptance is {(self.__Tchirp-self.__Tblack):.2f}s")
            print(f"Coalescence is at t={(self.__Ttot-self.__Tdepass):.2f}s")
            print(f"So the signal will enter detector acceptance at t={self.__Tblack:.2f}s")
            
            
    '''
    Template 3/10
    
    Produce the full template signal in the time-domain
    '''

    def _genStFromParams(self):

        itmin=0
        itmax=int(self.__tc/self.__delta_t)

        if self.__kindTemplate=='EM':
        
            # Simple approach, generate signal between 0 and tc, and then add some zeroes
            self.__St[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            self.__Stinit[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            
        elif self.__kindTemplate=='EOB':
                
            # The signal starting at frequency fDmin
            hp,hq = get_td_waveform(approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=self.__fDmin)
            f = utils.frequency_from_polarizations(hp, hq)

            c1=0
            for c1 in range(len(hp)-1,-1,-1): # Don't consider 0 at the end
                if abs(hp.numpy()[c1])>1e-35:
                    break
            hp_tab=hp.numpy()[:c1]
            freqs=f.numpy()[:c1]

            
            #itmin=int((self.__Tblack_start)/self.__delta_t)
            itmin=len(self.__St)-len(hp_tab)
            #print(itmin,len(hp_tab),len(self.__St))
            self.__St[:]= npy.concatenate((npy.zeros(itmin),hp_tab))
            self.__Stfreqs[:]= npy.concatenate((npy.zeros(itmin),freqs))
            self.__Stinit[:]= npy.concatenate((npy.zeros(itmin),hp_tab))

        else:
            raise ValueError("Valeur pour kindTemplate non prise en charge")
        
    '''
    Template 4/10
    
    Express the signal in frequency-domain
    
    We add screening at the beginning and at the end (for EM only)
    in order to avoid discontinuity
    
    Screening is obtained with a Blackman window:
    https://numpy.org/doc/stable/reference/generated/numpy.blackman.html
    '''
    
    def _genSfFromSt(self):
        S=npy.zeros(self.__N)
        S[:]=self.__St[:]
 
        # Blackman window at low freq
        # Compute the bins where the signal will be screend to avoid discontinuity
        # Should be between fDmin and fDmind
        #
        
        iwmax=int(self.__Tblack/self.__delta_t)
        iwmin=int(self.__Tblack_start/self.__delta_t)

        if self.__verb:
            print("Producing frequency spectrum from temporal signal")
            print(f"To avoid artifacts, blackman window will be applied to the signal start between times {(iwmin*self.__delta_t):.2f} and {(iwmax*self.__delta_t):.2f}")
 
        S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[:iwmax-iwmin]
         
        # Blackman window at low freq (EM only)
        #
        
        if self.__kindTemplate=='EM':
            weight=10
            iwmin=int(self.__tc/self.__delta_t)-weight
            iwmax=int(self.__tc/self.__delta_t)+weight
            S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[iwmin-iwmax:]
        
        # Finally compute the FFT of the screened signal
        # We use binning dependent params here, Sf value defined this way does
        # depend on frequency, as the norm factor is 1/sqrt(N) only
        #
        
        self.__Sf[:]=npy.fft.fft(S,norm='ortho')
        
        # Normalized to be coherent w/PSD def (the option ortho is doing a special normalization which is
        # fine only if you do the invert tranform afterward)
        self.__Sfn[:]=self.__Sf[:]/(npy.sqrt(self.__N)*self.__delta_f)
        del S
    
    '''
    Template 5/10
    
    Compute rhoOpt, which is the output we would get when filtering the template with noise only
    In other words this corresponds to the filtered power we should get in absence of signal
    
    The frequency range here is in principle the detector one
    But one could also use full frequency range (actually this is a good question ;-) )
    
    Often called the optimal SNR or SNRmax, which is kind of misleading
    
    '''
      
    def rhoOpt(self,Noise):     
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmind/self.__delta_f)
        
        if self.__verb:
            print(f"Calculations will be done in frequency range ({self.__fDmind},{min(self.__fDmaxd,self.__fe/2)})")
        
        # Evolution of the time w.r.t of the chirp frequency,
        # between fmind and fmaxd, ie the detector acceptance
        # This is w.r.t to Tchirp=Ttot, so numbers can be negative at start
        # if Ttot is lower than time in the detector acceptance
            
        freqs=npy.arange(len(self.__Sf))*self.__delta_f
        self._evolSnrTime=self.get_t(freqs[ifmin:ifmax]/2)
        self._evolSnrFreq=freqs[ifmin:ifmax]

        ifmaxe=-1
        ifmine=10000
            
        if self.__kindTemplate=='EOB': # Special EOB case treatment
            freqsEOB=self.__Stfreqs
            times=self.__T
            tmptimes=[]
            fmax=-1
            fmin=10000
            idx=0
            for f in freqsEOB:
                if f>fmax:
                    fmax=f
                    ifmaxe=idx
                if f<fmin:
                    fmin=f
                    ifmine=idx
                idx+=1

            self._evolSnrTime=self.__T[ifmine:ifmaxe]
            self._evolSnrFreq=freqsEOB[ifmine:ifmaxe]
                
        
        # <x**2(t)> calculation
        # Sfn and PSD have the same normalisation
        #
        # !!! PSD=Sn/2 !!!
        
        ropt=npy.sqrt(2*npy.trapz(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])/(Noise.PSD[ifmin:ifmax]),freqs[ifmin:ifmax]).real)
        
        #
        # Definition of the SNRmax used here is available in Eqn 26 of the foll. paper:
        #
        # https://arxiv.org/pdf/gr-qc/9402014.pdf
        #
        # !!! This value is not depending on Topt or fe !!!
        

        if self.__verb:
            print(f'MF output value when template is filtered by noise (No angular or antenna effects, D=1Mpc) over the total period is equal to {ropt:.2f}',self.__delta_f)
        self.__norm=ropt
        

        # Here we compute the rhoOpt share per frequency bin (put the right norm for PSD here)
        self._evolSnr_f = (2/self.__norm*self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])/(Noise.PSD[ifmin:ifmax]/self.__delta_f)).real
        
        
        # Get the maximum reachable SNR
        self._evolSnr=npy.cumsum(self._evolSnr_f)  # SNR evolution vs time/freq
        snrMaxTest=self._evolSnr[len(self._evolSnr)-1]

        if self.__verb:
            print(f'Max SNR which can be collected= {snrMaxTest:.2f}')
            print('')
            print('Now compute the SNR repartition among chunks of data')

        self._evolSnr = self._evolSnr/snrMaxTest   # SNR evolution normalized
        
        # Compute the SNR sharing along time for the sample produced

        idx_samp=[]
        # tstart is the time when our sample starts in the template produced
        tstart=self.__Ttot-self.__Tsample
        
        # When do we enter into the different data chunk ?
        for j in range(self.__nTsample-1,-1,-1):
            tend=tstart+self.__listTsample[j]
            k=0
            found=False
            if (self._evolSnrTime[0]>tend):
                idx_samp.append(-1)
                tstart=tend
            else:
                for i in self._evolSnrTime:
                    if i>=tstart and found==False:
                        idx_samp.append(k)
                        found=True
                    if i>=tend:
                        tstart=tend
                        break
                    k+=1
        idx_samp.reverse()

        
        # Ok now we have everything in hand to compute the times
        # So for n bands each portion will contain 100/n% of rhoOpt
        # tint will contain the time of each section, going back from Tc=0
        # so the range [Tc-tint[n-1],Tc] will contain 100/n % of rhoopt, and so on
            
                
        ifint=ifmin+1
        rint=1./self.__nTsample
                
        vals=npy.arange(self.__nTsample+1)*rint
        self._Snr_vs_freq=self._evolSnr
        self._Snr_vs_freq_base=self._Snr_vs_freq
        theSum=0.
        
        if self.__kindTemplate=='EOB':
            Snr_EOB=[]
            for j in range(len(self._evolSnrTime)):
            
                if self._evolSnrFreq[j]<self.__fDmind:
                    Snr_EOB.append(0)
                    continue
                idxfreq=int((self._evolSnrFreq[j]-self.__fDmind)/self.__delta_f)
                if idxfreq!=0:
                    Snr_EOB.append(self._Snr_vs_freq[idxfreq])
                else:
                    Snr_EOB.append(0)
            self._Snr_vs_freq=npy.asarray(Snr_EOB)

        for j in range(self.__nTsample):
        
            if idx_samp[j]==-1:
                if self.__verb:
                    print("We collect no SNR in chunk",j,f"({self.__listfe[j]}Hz)")
            else:
                if (j==0):
                    self._rawSnr[j]=100*(1-self._Snr_vs_freq[idx_samp[j]])
                else:
                    self._rawSnr[j]=100*(self._Snr_vs_freq[idx_samp[j-1]]-self._Snr_vs_freq[idx_samp[j]])
        
                if self.__verb:
                    if (j==0):
                        print("We collect",100*(1-self._Snr_vs_freq[idx_samp[j]]),"% of SNR in chunk",j,f"({self.__listfe[j]}Hz)")
                    else:
                        print("We collect",100*(self._Snr_vs_freq[idx_samp[j-1]]-self._Snr_vs_freq[idx_samp[j]]),"% of SNR in chunk",j,f"({self.__listfe[j]}Hz)")
                        
        
        for j in range(self.__nTsample):
            idx=0
                        
            for k in self._evolSnrFreq:
                idxfreq=int((k-self.__fDmind)/self.__delta_f)
                snrprop=1.1
                if (idxfreq>=0):
                    snrprop=self._Snr_vs_freq_base[idxfreq]
            
                if snrprop<=vals[j]:
                    self._tint[j]=self._evolSnrTime[idx]
                    self._fint[j]=k
                idx+=1
            
        
        totalSnr = npy.sum(self._rawSnr)
        if self.__verb:
            print(f"With the current samples one collected  : {totalSnr}% of the possible SNR")
            print("")
            print(f"Optimal sharing (tstart/fstart of the chunk) with this number of bands would be the following for this template: \n--> Timings : {self._tint}, \n--> Frequencies : {self._fint}")

            
        # Renormalize to 100% (SV: not sure this is really necessary, or as a cross check afterwards)
        
        if totalSnr>0:
            self._currentSnr = self._rawSnr/totalSnr
        if self.__verb:
            print(f"Voici les pourcentages renormalisés en SNR des chunks choisis : {self._currentSnr}")
        
        return self.__norm

    '''
    Template 6/10
    
    Then we normalize the signal, first with PSD (like the noise)
    Then with rhoOpt (to get SNR=1)
    With this method to get a signal at a given SNR one just have to rescale it by a factor SNR
    
    It's important to have the same normalization everywhere otherwise it's a mess
    '''

    def _whitening(self,kindPSD,Tsample,norm):
        if kindPSD is None:
            print('No PSD given, one cannot normalize!!!')
            self.__St[:]=npy.fft.ifft(self.__Sf,norm='ortho').real
            return self.__St
        
        # We create a noise instance to perform the whitening
        
        Noise=GenNoise(Ttot=self.__Ttot,fe=self.__fe, kindPSD=kindPSD,fmin=self.__fDmin,fmax=self.__fDmaxd)
        Noise.getNewSample()
        self.Noise=Noise


        # Important point, noise and signal are produced over the same length, it prevent binning pbs
        # We just produce the PSD here, to do the weighting
        rho=self.rhoOpt(Noise=Noise)         # Get SNRopt
        Sf=npy.zeros(self.__N,dtype=complex)
        Sf=self.__Sf/npy.sqrt(Noise.PSD*self.__N*self.__delta_f)     # Whitening of the signal
        # Need to take care of the fact that PSD has not the normalization of Sf.
    
        self.__Sfn=Sf                        # Signal whitened
      
        # The withened and normalized signal in time domain
        #
        
        self.__St[:]=npy.fft.ifft(Sf,norm='ortho').real/(rho if norm else 1)
        
        # We also add to the template object the corresponding matched filter
        # The filter is defined only on a limited frequency range (the detector one)
        # Whatever range you choose you should pick the same than for rhoOpt here
        
        # Otherwise it's 0 (band-pass filter)
        
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)
                
        self.__Filtf=npy.conjugate(Sf) # The filter function is whitened
        self.__Filtf[:ifmin]=0
        self.__Filtf[ifmax:]=0
        self.__Filtf[-1:-self.__N//2:-1]=npy.conjugate(self.__Filtf[1:self.__N//2])

    '''
    TEMPLATE 7/10
    
    Signal resampling
    '''
    
    def _resample(self,signal):
    
        S=[]
        cutlimit=[]
        vmax=len(signal)

        for j in range(self.__nTsample):  # Loop over subsamples
            decimate=int(self.__fe/self.__listfe[j])
            N=int(self.__listTsample[j]*self.__fe)  # Length of the subsample
            T=npy.arange(N)*self.__delta_t
            cutlimit.append(vmax-len(T))
            if cutlimit[j] > len(signal):
                raise ValueError("La valeur de début de cut ne doit pas être en dehors du vecteur cut")
            chunk=signal[cutlimit[j]:vmax]
            S.append(chunk[::int(decimate)])
            vmax=cutlimit[j]
        return S
    
    '''
    TEMPLATE 8/10
    
    The main template generation macro
    '''
    
    def getNewSample(self,kindPSD='flat',Tsample=1,tc=None,norm=True):
        
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
    
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!=None:
            raise ValueError("Les seules valeurs autorisées sont None, 'flat', et 'analytic'")
        
        if not(isinstance(norm,bool)):
            raise TypeError("Un booléen est attendu")
        
        # The signal is produced along time, no normalization yet
        self._genStFromParams()

        # Go in the frequency domain
        self._genSfFromSt()

        # Whiten the signal and normalize it to SNR=1
        self._whitening(kindPSD,Tsample,norm)

        # Resample it if needed
        tc= self.__Ttot-self.__Tdepass if tc==None else min(tc,self.__Ttot-self.__Tdepass)

        # Here is the number of points of the required frame
        N=int(Tsample*self.__fe)
        S=npy.zeros(N)

        itc=int(tc/self.__delta_t)

        if tc<=self.__Ttot:
            S[:itc]=self.__St[-itc:] # There will be 0s at the start
        else:
            S[itc-self.__N:itc]=self.__St[:] # There will be 0s at the end

        if self.__nTsample > 1:
            return self._resample(S)
        else :
            listS=[]
            listS.append(S)
            return listS
          

    '''
    TEMPLATE 9/10
    
    Produce a new template with a different value of tc (useful for training sample prod)
    '''
    
    def getSameSample(self,Tsample=1,tc=None):
        
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
    
        tc= self.__Ttot-self.__Tdepass if tc==None else min(tc,self.__Ttot-self.__Tdepass)
        N=int(Tsample*self.__fe)
        S=npy.zeros(N)

        itc=int(tc/self.__delta_t)
        if tc<=self.__Ttot:
            if len(S)!=itc:
                itc = len(S)
            S[:itc]=self.__St[-itc:]
        else:
            S[itc-self.__N:itc]=self.__St[:]
        if self.__nTsample > 1:
            return self._resample(S)
        else :
            listS=[]
            listS.append(S)
            return listS
        
        
    '''
    TEMPLATE 10/10
    
    Plot macros
    '''
    
    # The template in time domain (with a different color for the samples
    
    def plot(self,Tsample=1,tc=0.95,SNR=1):
        
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
        
        listT=[]
        if self.__nTsample > 1:
            maxT=self.__Tsample
            for j in range(self.__nTsample):
                delta_t=1/self.__listfe[j]
                N=int(self.__listTsample[j]*self.__listfe[j])
                listT.append(npy.arange((maxT-self.__listTsample[j])/delta_t,maxT/delta_t)*delta_t)
                maxT-=self.__listTsample[j]
        else:
            N=int(Tsample*self.__fe)
            listT.append(npy.arange(N)*self.__delta_t)
        
        for j in range(self.__nTsample):
            plt.plot(npy.resize(listT[j],len(self.getSameSample(Tsample=Tsample,tc=tc)[j])), self.getSameSample(Tsample=Tsample,tc=tc)[j]*SNR,'-',label='h(t)')

        plt.title('Template dans le domaine temporel de masses ('+str(self.__m1/Msol)+', '+str(self.__m2/Msol)+') Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (No Unit)')
        plt.grid(True, which="both", ls="-")
        plt.legend()

    # 1-D signal projection
    def plotSignal1D(self):
        _, bins, _ = plt.hist(self.__St,bins=100, density=1)
        #_, bins, _ = plt.hist(npy.abs(self.__Sfn),bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__St)
        print("Largeur de la distribution temporelle (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        #plt.yscale('log')
        plt.plot(bins, best_fit_line)
        plt.title('Bruit temporel normalisé')

    # The Filter
    def plotFilt(self,Tsample=1,SNR=1):
    
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
        
        N=int(Tsample*self.__fe)
        T=npy.arange(N)*self.__delta_t
        
        plt.plot(T, self.__Filt,'-',label='filt(t)')
        plt.title('Template filtré dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('rho(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    # Fourier transform of whitened signal and noise. Signal normalized to S/N=1
    def plotTF(self):
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.Noise.PSD[ifmin:ifmax]/self.__delta_f),'-',label='Sn(f)')
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sf[ifmin:ifmax])/self.__norm,'.')
        plt.title('Template dans le domaine frequentiel')
        plt.xlabel('f (Hz)')
        plt.ylabel('h(f) (No Unit)')
        plt.yscale('log')
          
    # Normalized FFT of signal and SNR proportion evolution
    def plotTFn(self):
                         
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmind/self.__delta_f)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sfn[ifmin:ifmax]),'.')
        ax2.plot(self.__F[ifmin:ifmax],self._evolSnr,'.',color='red')
        ax1.set_yscale('log')
        plt.grid(True, which="both", ls="-")

    # Evolution of the SNR vs time, divided by samples
    def plotSNRevol(self):

        props=self._rawSnr/100.
        
        fig3, ax3 = plt.subplots()
        fig3.suptitle('SNR accumulation vs template time')
        ax3.plot(self._evolSnrTime,self._Snr_vs_freq)
        ax3.set_xlabel('t')
        ax3.set_ylabel('SNR proportion collected',rotation=270)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_label_coords(1.1,0.5)
        ax3.set_ylim([0., 1.])
        ymin, ymax = ax3.get_ylim()
        xmin, xmax = ax3.get_xlim()
        totprop=1.
        
        # Select the color map named rainbow
        cmap = cm.get_cmap(name='gist_rainbow')
        tstart=self.__Ttot
            
        for j in range(self.__nTsample):
            propchunk=props[j]
            tstart-=self.__listTsample[j]
            ax3.axvspan(tstart, self.__Ttot, (totprop-propchunk)*ymax, totprop*ymax, facecolor=cmap(int(256/(2*self.__nTsample))*j),alpha=0.5)
            plt.axvline(x = tstart, color = 'b', linestyle='dashed')
            totprop-=propchunk
    
    @property
    def length(self):
        return self.__N
        
    def duration(self):
        return self.__Tchirp
        
    def filtre(self):
        return self.__Filtf
    
    def norma(self):
        return self.__norm
  
#################################################################################################################################################################################################
'''
Class handling dataset production (either for training or testing)

Options:

--> mint    :
--> step    :
--> NbB     :
--> tcint   :
--> choice  :
--> kindBank:

'''


class GenDataSet:
    """Classe générant des signaux des sets de training 50% dsignaux+bruits 50% de bruits"""
    def __init__(self,mint=(10,50),NbB=1,tcint=(0.75,0.95),kindPSD='flat',kindTemplate='EM',Ttot=1,fe=2048,kindBank='linear',paramFile=None,step=0.1,choice='train'):
        
        self.__choice=choice
        if paramFile is None:
            self.__listTtot=Ttot
            self.__listfe=fe
            self.__nTtot=1
            self.__Ttot=Ttot
            self.__fe=fe
            self.__step=step
            self.__kindPSD=kindPSD
            self.__mmin=min(mint)
            self.__mmax=max(mint)
            self.__tcmin=self.__Ttot*max(min(tcint),0.5)
            self.__tcmax=self.__Ttot*min(max(tcint),1.)
            self.__NbB=NbB
            self.__kindTemplate=kindTemplate
            if (kindBank=='linear' or kindBank=='optimal'):
                self.__kindBank=kindBank
            else:
                raise ValueError("Les valeurs pour la banque sont 'optimal' ou 'linear'")
        elif os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")

        start_time = time.time()
        print("Starting dataset generation")

        # Noise stream generator
        self.__NGenerator=GenNoise(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD)
        # Template stream generator
        self.__TGenerator=GenTemplate(Tsample=self.__listTtot,fe=self.__listfe,kindTemplate=self.__kindTemplate)
    
        self.__listDelta_t=[] # pour le plot des portions de SNR
        self.__listSNRevolTime=[]
        self.__listSNRevol=[]
        self.__listfnames=[]
        self.__listSNRchunksAuto=[[] for x in range(self.__nTtot)]
        self.__listSNRchunksSpecial=[[] for x in range(self.__nTtot)]
        self.__listSNRchunksBalance=[[] for x in range(self.__nTtot)]
                
        print("1 After init --- %s seconds ---" % (time.time() - start_time))
        
        self._genGrille()   # Binary objects mass matrix
        
        print("2 After grille --- %s seconds ---" % (time.time() - start_time))
        
        self._genSigSet()   # The signals
        
        print("3 After signal --- %s seconds ---" % (time.time() - start_time))
        
        self._genNoiseSet() # The noises (one realization per template)
        
        print("4 After noise --- %s seconds ---" % (time.time() - start_time))
        
        for j in range(self.__nTtot):
            print("Chunk",j)
            print("Signal set shape:",self.__Sig[j].shape)
            print("Noise set shape:",self.__Noise[j].shape)
        
        self.__Labels=npy.concatenate((npy.ones(self.__Ntemplate*self.__NbB,dtype=int),npy.zeros(self.__Ntemplate*self.__NbB,dtype=int))) # 1 <-> signal , 0 <-> noise

        # At the end we plot a map of the different SNRs
        self.plotSNRmap()

        plt.show()
    
    '''
    DATASET 1/
    
    Parser of the parameters file
    '''
    
    def _readParamFile(self,paramFile):
        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='Ttot' or lignes[1][0]!='fe' or lignes[2][0]!='kindPSD' or lignes[3][0]!='mint' or lignes[4][0]!='tcint' or lignes[5][0]!='NbB' or lignes[6][0]!='kindTemplate' or lignes[7][0]!='kindBank' or lignes[8][0]!='step' or len(lignes)!=9:
            raise Exception("Dataset param file error")
        if not len(lignes[0])==len(lignes[1]):
            raise Exception("Ttot and fe vectors don't have the same size")
        
        nValue=len(lignes[0]) # taille de la liste des valeurs de Ttot +1 (légende)
        self.__nTtot=nValue-1
        self.__listTtot=[]
        self.__listfe=[]
        if nValue > 2:  # Multi band
            for x in range(1,nValue):
                self.__listTtot.append(float(lignes[0][x]))
                self.__listfe.append(float(lignes[1][x]))
            self.__Ttot=sum(self.__listTtot)
            self.__fe=max(self.__listfe)
        else: # Single band
            self.__Ttot=float(lignes[0][1])
            self.__listTtot.append(self.__Ttot)
            self.__fe=float(lignes[1][1])
            self.__listfe.append(self.__fe)
            
        self.__kindPSD=lignes[2][1]
        self.__mmin=min(float(lignes[3][1]),float(lignes[3][2]))
        self.__mmax=max(float(lignes[3][1]),float(lignes[3][2]))
        self.__tcmin=self.__Ttot*max(min(float(lignes[4][1]),float(lignes[4][2])),0.5)
        self.__tcmax=self.__Ttot*min(max(float(lignes[4][1]),float(lignes[4][2])),1.)
        self.__NbB=int(lignes[5][1])
        self.__kindTemplate=lignes[6][1]
        self.__step=float(lignes[8][1])
        kindBank=lignes[7][1]
        if (kindBank=='linear' or kindBank=='optimal'):
            self.__kindBank=kindBank
        else:
            raise ValueError("Les valeurs pour la banque sont 'optimal' ou 'linear'")
            
    '''
    DATASET 2/
    
    Produce a data grid with the mass coordinates
    '''
    
    def _genGrille(self):
        if self.__kindBank=='linear':
            n = self.__step
            N = len(npy.arange(self.__mmin, self.__mmax, n))
            self.__Ntemplate=int((N*(N+1))/2)
            self.__GrilleMasses=npy.ones((self.__Ntemplate,2))
            self.__GrilleMasses.T[0:2]=self.__mmin
            c=0
            
            # Fill the grid (half-grid in fact)
            # Each new line starts at the diagonal, then reach the end
            # mmin,mmax
            # mmin+step,mmax
            #...
            
            for i in range(N):
                for j in range(i,N):
                
                    self.__GrilleMasses[c][0]=self.__mmin+i*self.__step
                    self.__GrilleMasses[c][1]=self.__mmin+j*self.__step
                    c+=1
                    
        else: # The optimized bank
            with open(os.path.dirname(__file__)+'/params/outputVT_sorted.dat') as mon_fichier:
                mon_fichier_reader = csv.reader(mon_fichier, delimiter=' ')
                M = npy.array([l for l in mon_fichier_reader],dtype=float)

            self.__GrilleMasses=((M[(M.T[0]>=self.__mmin) & (M.T[1]>=self.__mmin) & (M.T[0]<=self.__mmax) & (M.T[1]<=self.__mmax)]).T[:2]).T
            self.__Ntemplate=len(self.__GrilleMasses)
        
    '''
    DATASET 3/
    
    Produce the templates
    '''
        
    def _genSigSet(self):
        self.__Sig=[]
        c=0
        
        # First we produce the object with the correct size
        # The size is 2*Ntemplate*NbB but we put template only in the first half
        # The rest is filled with 0, it's important for GetDataSet
        
        for j in range(self.__nTtot): # Loop over samples
            dim=int(self.__listTtot[j]*self.__listfe[j])
            self.__Sig.append(npy.zeros((self.__Ntemplate*2*self.__NbB,dim)))
            
            # The following lines are for the SNR repartition
            self.__listSNRchunksAuto[j].append(npy.full((self.__Ntemplate*2*self.__NbB),(1.0/self.__nTtot)))
            self.__listSNRchunksBalance[j].append(npy.full((self.__Ntemplate*2*self.__NbB),(1.0/self.__nTtot)))
            if j==0:
                self.__listSNRchunksSpecial[j].append(npy.full((self.__Ntemplate*2*self.__NbB),(1.0)))
            else:
                self.__listSNRchunksSpecial[j].append(npy.full((self.__Ntemplate*2*self.__NbB),(0.0)))

        self.__listSNRchunksAuto=npy.reshape(self.__listSNRchunksAuto, (self.__nTtot, self.__Ntemplate*2*self.__NbB))
        self.__listSNRchunksSpecial=npy.reshape(self.__listSNRchunksSpecial, (self.__nTtot, self.__Ntemplate*2*self.__NbB))
        self.__listSNRchunksBalance=npy.reshape(self.__listSNRchunksBalance, (self.__nTtot, self.__Ntemplate*2*self.__NbB))

        # Now fill the object
        for i in range(0,self.__Ntemplate):
            if c%1000==0:
                print("Producing sample ",c,"over",self.__Ntemplate*self.__NbB)
            self.__TGenerator.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1])
            temp=self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,
                                                             Tsample=self.__Ttot,
                                                             tc=npy.random.uniform(self.__tcmin,self.__tcmax))

            self.__listSNRevol=npy.append(self.__listSNRevol,self.__TGenerator._rawSnr)

            # Fill the corresponding data
            for j in range(self.__nTtot):
                self.__Sig[j][c]=temp[j]
                self.__listSNRchunksAuto[j][c]=self.__TGenerator._currentSnr[j]
            c+=1
            
            # Fill the NbB-1 additional samples (with a different tc)
            for k in range(1,self.__NbB):
                temp=self.__TGenerator.getSameSample(Tsample=self.__Ttot,tc=npy.random.uniform(self.__tcmin,self.__tcmax))
                for j in range(self.__nTtot):
                    self.__Sig[j][c]=temp[j]
                    self.__listSNRchunksAuto[j][c]=self.__TGenerator._currentSnr[j]
                c+=1   
        self.__listSNRchunksAuto=npy.transpose(self.__listSNRchunksAuto)
        self.__listSNRchunksSpecial=npy.transpose(self.__listSNRchunksSpecial)
        self.__listSNRchunksBalance=npy.transpose(self.__listSNRchunksBalance)
             
             
    '''
    DATASET 4/
    
    Produce the noises here we fill everything
    '''
          
    def _genNoiseSet(self):
        self.__Noise=[]
        for j in range(self.__nTtot):
            dim=int(self.__listTtot[j]*self.__listfe[j])
            self.__Noise.append(npy.zeros((self.__Ntemplate*2*self.__NbB,dim)))

        for i in range(0,self.__Ntemplate*self.__NbB*2):
            if i%1000==0:
                print("Producing sample ",i,"over",self.__Ntemplate*self.__NbB*2)
            temp=self.__NGenerator.getNewSample()
            for j in range(self.__nTtot):
                self.__Noise[j][i]=temp[j]

    '''
    DATASET 5/
    
    Get a dataset from the noise and signal samples
    '''

    def getDataSet(self,SNRopt=1,weight='auto'):
        nbands=self.__nTtot
        dset=[]

        if weight=='auto':
            list_weights=self.__listSNRchunksAuto
        if weight=='balance':
            list_weights=self.__listSNRchunksBalance
        if weight=='special':
            list_weights=self.__listSNRchunksSpecial
            
        print("Getting a training set of",self.Nsample,"events based on",nbands,"frequency bands")

        # If the SNR is within a range, we define a vector containing the random SNR values
        if (isinstance(SNRopt,tuple)):
            randSNR=npy.random.uniform((min(SNRopt)), (max(SNRopt)), size=self.Nsample)

        for i in range(nbands):
            if (isinstance(SNRopt,tuple)):
                dset.append((self.__Sig[i].T*randSNR).T+self.__Noise[i]) # Sig=0 in the second half, so this is just noise...
            else:
                dset.append(self.__Sig[i]*(SNRopt)+self.__Noise[i])

        # Dataset has form ([Nsample][N1],[Nsample][N2],...)
        # Reorganize it as [Nsample][N1+N2]
        finaldset=[]

        for i in range(self.Nsample):
            tempset=[]
            for j in range(nbands):
                sect=npy.asarray(dset[j][i])
                tempset.append(sect)
            sec=npy.concatenate(tempset)
            finaldset.append(sec)
        fdset=npy.asarray(finaldset)
                
        return fdset, list_weights
    
    
    '''
    DATASET 6/
    
    Special macro
    '''
    
    def specialGetDataSet(self,SNRopt=1):
        nbands=self.__nTtot
        dset=[]
        listSNRchunks = [[0.5] for x in range(nbands)]
        middle=int((((npy.shape(self.__Sig[0])[0])-1)*0.25)+1)
            
        print("Getting a special test set based on",nbands,"frequency bands")

        for i in range(nbands):
            dset.append(self.__Sig[i][middle]*(SNRopt)+self.__Noise[i][middle])

        finaldset=[]
        tempset=[]
        for j in range(nbands):
            sect=npy.asarray(dset[j])
            tempset.append(sect)
        sec=npy.concatenate(tempset)
        finaldset.append(sec)

        fdset=npy.asarray(finaldset)
        #print(fdset.shape)
        return fdset, listSNRchunks
    
    '''
    DATASET 7/
    
    Plots
    '''
    
    @property
    def specialLabels(self):
        middle=int((((npy.shape(self.__Sig[0])[0])-1)*0.25)+1)
        npy.set_printoptions(threshold=npy.inf) 
        print(f"Le label de l'échantillon choisi pour le test est {self.__Labels[middle]}")
        return self.__Labels[middle]
        
    def plot(self,i,SNRopt=1):
        plt.plot(self.__NGenerator.T,self.getDataSet(SNRopt)[0][i],'.')
        
    def plotSNRmap(self):
        mstep = self.__step # pas des masses à faire choisir plus tard
        mlow=float(self.__mmin)
        mhigh=float(self.__mmax)
        
        Nbmasses=int((mhigh-mlow)/mstep)
        residual=(mhigh-mlow)/mstep-Nbmasses
        if residual>0:
            Nbmasses=Nbmasses+1

        X, Y = npy.meshgrid(npy.linspace(mlow-mstep/2., mhigh+mstep/2., Nbmasses+1), npy.linspace(mlow-mstep/2., mhigh+mstep/2., Nbmasses+1))
        
        for k in range(self.__nTtot):
            Z=npy.zeros((Nbmasses,Nbmasses))
            c=0
            for i in range(Nbmasses):
                if c==len(self.__listSNRevol/self.__nTtot):
                    break
                for j in range(i,Nbmasses):
                    #print(i,j,self.__listSNRevol[self.__nTtot*c+k])
                    Z[i][j] = self.__listSNRevol[self.__nTtot*c+k]
                    if (i!=j):
                        Z[j][i] = self.__listSNRevol[self.__nTtot*c+k]
                    c+=1
                    if c==len(self.__listSNRevol/self.__nTtot):
                        break
            plt.figure('MapDistribution_SNR : chunck N°'+str(k))
            plt.pcolormesh(X,Y,Z)
            plt.xlabel('m1')
            plt.ylabel('m2')
            plt.colorbar()
            plt.title(label='Proportion of the total SNR collected in the plan (m1,m2) : chunk N°'+str(k))
            plt.show()

    
    def saveGenerator(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")

        # Save the sample in an efficient way
        for k in range(self.__nTtot):
            fname=dossier+self.__choice+'-'+str(self.__nTtot)+'chunk'+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+self.__kindBank+'-'+str(k)+'-'+str(self.__Ttot)+'s'+'-data'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the object without the samples
        self.__Sig=[]
        self.__Noise=[]
        fichier=dossier+self.__choice+'-'+str(self.__nTtot)+'chunk'+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+self.__kindBank+'-'+str(self.__Ttot)+'s'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
                
                
    @classmethod
    def readGenerator(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        
        print("We deal with a dataset containing",obj.__nTtot,"frequency bands")
        for i in range(obj.__nTtot):
            print("Band",i,"data is contained in file",obj.__listfnames[i]+'.npz')
            data=npy.load(str(obj.__listfnames[i])+'.npz')
            obj.__Sig.append(data['arr_1'])
            obj.__Noise.append(data['arr_0'])
        data=[] # Release space
        f.close()
        return obj
    
    @property
    def Ntemplate(self):
        return self.__Ntemplate
    @property
    def Nsample(self):
        return self.__Ntemplate*2*self.__NbB
    @property
    def Labels(self):
        return self.__Labels
    @property
    def mInt(self):
        return self.__mmin,self.__mmax
    @property
    def mStep(self):
        return self.__step
    @property
    def kindPSD(self):
        return self.__kindPSD
    @property
    def kindTemplate(self):
        return self.__kindTemplate
    @property
    def kindBank(self):
        return self.__kindBank
############################################################################################################################################################################################
def parse_cmd_line():
    import argparse
    """Parseur pour la commande gendata"""
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="commande à choisir",choices=['noise', 'template', 'set', 'sequence', 'all'])
    parser.add_argument("--kindPSD","-kp",help="Type de PSD à choisir",choices=['flat', 'analytic'],default='analytic')
    parser.add_argument("--kindTemplate","-kt",help="Type de PSD à choisir",choices=['EM', 'EOB'],default='EM')
    parser.add_argument("--time","-t",help="Durée du signal",type=float,nargs='+',default=[1., 9.])
    parser.add_argument("-fe",help="Fréquence du signal",type=float,nargs='+',default=[2048, 512])
    parser.add_argument("-fmin",help="Fréquence minimale visible par le détecteur",type=float,default=20)
    parser.add_argument("-fmax",help="Fréquence maximale visible par le détecteur",type=float,default=1000)
    parser.add_argument("-snr",help="SNR (pour creation de sequence)",type=float,default=7.5)
    parser.add_argument("-m1",help="Masse du premier objet",type=float,default=20)
    parser.add_argument("-m2",help="Masse du deuxième objet",type=float,default=20)
    parser.add_argument("--set","-s",help="Choix du type de set par défaut à générer",choices=['train','test'],default=None)
    parser.add_argument("-step",help="Pas considéré pour la génération des paires de masses",type=float,default=0.1)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de set",default=None)
    parser.add_argument("--verbose","-v",help="verbose mode",type=bool,default=False)
    
    args = parser.parse_args()
    return args

'''
Main part of gendata.py



'''
def main():
    import gendata as gd
    
    args = parse_cmd_line()
    

    if args.cmd=='noise': # Simple Noise generation
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax,verbose=args.verbose)
        NGenerator.getNewSample(whitening=True)
    
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()
    
        if args.verbose:
            plt.figure(figsize=(10,5))
            NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
            NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
            plt.figure(figsize=(10,5))
            NGenerator.plotTF2(fmin=args.fmin,fmax=args.fmax)
        
            for j in range(len(args.time)):
                plt.figure(figsize=(10,5))
                NGenerator.plotNoise1D(j)

        plt.show()
        
    elif args.cmd=='template': # Simple Template generation
        TGenerator=gd.GenTemplate(Tsample=args.time,fe=args.fe,kindTemplate=args.kindTemplate,verbose=args.verbose)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=True)
        
        
        plt.figure(figsize=(10,5))
        TGenerator.plotTF()
            
        plt.figure(figsize=(10,5))
        TGenerator.plotTFn()
        
        print(f"TGenerator.duration() = {TGenerator.duration()}")
        #plt.figure(figsize=(10,5))
        #TGenerator.plot(Tsample=args.time,SNR=7.5)
        plt.figure(figsize=(10,5))
        TGenerator.plot(Tsample=TGenerator.duration(),tc=TGenerator.duration(),SNR=args.snr/npy.sqrt(1.))
        
        plt.figure(figsize=(10,5))
        TGenerator.plotSignal1D()
        
        plt.figure(figsize=(10,5))
        TGenerator.plotSNRevol()
        
        plt.show()
        
    elif args.cmd=='all':
        TGenerator=gd.GenTemplate(Tsample=args.time,fe=args.fe,kindTemplate=args.kindTemplate,verbose=args.verbose)
        TGenerator.majParams(args.m1,args.m2)
        
        if (len(args.time)>1):
            randt=npy.random.uniform(TGenerator.duration(),sum(args.time))
        else :
            randt=npy.random.uniform(TGenerator.duration(),args.time)
            
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=TGenerator.duration(),tc=randt,norm=True)
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax,verbose=args.verbose)
        NGenerator.getNewSample()
                
        plt.figure(figsize=(10,5))
        NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
        NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()
        if (len(args.time)>1):
            TGenerator.plot(Tsample=TGenerator.duration(),tc=TGenerator.duration(),SNR=args.snr/npy.sqrt(1.))
        else:
            TGenerator.plot(Tsample=args.time,tc=TGenerator.duration(),SNR=args.snr/npy.sqrt(1.))
            
        #plt.legend()
        plt.show()
        
        
    else: # Logically there is only set remaining, so dataset
        if args.set=='train':
            chemin='./params/default_trainGen_params.csv'
            set='train'
            Generator=gd.GenDataSet(paramFile=chemin,choice=set)
        elif args.set=='test':
            chemin='./params/default_testGen_params.csv'
            set='test'
            Generator=gd.GenDataSet(paramFile=chemin,choice=set)
        else:
            Generator=gd.GenDataSet(paramFile=args.paramfile)
    
        chemin = './generators/'
        Generator.saveGenerator(chemin)

############################################################################################################################################################################################
if __name__ == "__main__":
    main()
