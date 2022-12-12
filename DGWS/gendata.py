import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
from pycbc.waveform import get_td_waveform
from scipy.stats import norm
from scipy import signal
from scipy.interpolate import interp1d

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

->Ttot    : noise sample duration, in seconds (Default is 1)
->fe      : sampling frequency, in Hz (Default is 2048)
->kindPSD : noise type: 'flat' or 'analytic' (Default is flat)
->fmin    : minimal frequency for noise definition
->fmax    : maximal frequency for noise definition
->nsamp   : for each frequency the PSD is considered as a gaussian centered on PSD
            of width PSD/sqrt(nsamp), number of samples used to estimate the PSD

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution
            still it's not based on real data, ie not including glitches for example.
            Reference used for analytic noise is cited below

Noise is produced over a given frequency range.

Indeed there is no reason to produce noise well outside
detector acceptance

'''

class GenNoise:
    """Classe générant du bruit"""
    def __init__(self,Ttot=1,fe=2048,kindPSD='flat',nsamp=1600,fmin=20,fmax=1500):
        """constructeur: Choix du temps total Ttot et de sa fréquence d'échantillonnage fe et du type de bruit kindPSD"""
        if not((isinstance(Ttot,int) or isinstance(Ttot,float)) and (isinstance(fe,int) or isinstance(fe,float))):
            raise TypeError("Ttot et fe doivent être des ints ou des floats")
        if not(isinstance(kindPSD,str)):
            raise TypeError("kindPSD doit être de type str")
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées pour kindPSD sont 'flat' et 'analytic'")
            
        self.__Ttot=Ttot                    # temps total du signal
        self.__fe=fe                        # frequence d'echantillonage
        self.__fmin=fmin                    #
        self.__fmax=fmax                    #
        self.__N=int(self.__Ttot*self.__fe) # nombre de points temporel
        self.__delta_t=1/self.__fe          # pas de temps
        self.__delta_f=self.__fe/self.__N   # pas de fréquence
        self.__kindPSD=kindPSD              # type de PSD
        self.__nsamp=nsamp                  # largeur du bruit

       
        # Vecteurs de base
        self.__T=npy.arange(self.__N)*self.__delta_t   #vecteur des temps
        
        #vecteur des frequences [0->fe/2,-fe/2->-1] pour le module iFFT
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f

        self._genPSD() #Génération de la PSD du bruit constant dans le cas d'un bruit blanc
        
        self.__Nt=npy.zeros(self.__N)               #vecteur d'une realisation du bruit
        self.__Nf=npy.zeros(self.__N,dtype=complex) #TF de la realisation du bruit
        self.__Nf2=npy.zeros(self.__N,dtype=complex) #TF de la realisation du bruit
        self.__Nfr=npy.zeros(self.__N, dtype=npy.float64)              # TF real part
        self.__Nfi=npy.zeros(self.__N, dtype=npy.float64)              # TF imaginary part
        
    '''
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
    Produce the PSD, so in the noise power in frequency domain
    We normalize it with the size of the FFT, in order to get coherent value wrt the signal
    '''

    def _genPSD(self):
    
                    
        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
    
    
        self.__PSD=npy.ones(self.__N, dtype=npy.float64) # Setting the PSD to one means infinite noise
        if self.__kindPSD=='flat':
            sigma=2e-23*npy.sqrt(self.__N)
            self.__PSD[ifmin:ifmax]=sigma**2
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2] # Double sided
        elif self.__kindPSD=='analytic':
            self.__PSD[ifmin:ifmax]=self.Sh(abs(self.__F[ifmin:ifmax]))*self.__N
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]
            
    '''
    PSD type change
    '''

    def changePSD(self,kindPSD):
        del self.__PSD
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées sont 'flat' et 'analytic'")
        self.__kindPSD=kindPSD
        self._genPSD()

    '''
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
        for i in range(len(self.__Nfr)):
            if (i<ifmax and i>ifmin):
                phi0 = float(npy.random.randint(1000))/1000.*8*math.atan(1.);
                self.__Nfr[i]*=math.cos(phi0) # real part
                self.__Nfi[i]*=math.sin(phi0) # imaginary part
            else:
                self.__Nfr[i]=0 # real part
                self.__Nfi[i]=0 # imaginary part
            
        # Then we can define the components
        self.__Nf[0:self.__N//2+1].real=self.__Nfr[0:self.__N//2+1]
        self.__Nf[0:self.__N//2+1].imag=self.__Nfi[0:self.__N//2+1]
        self.__Nf[-1:-self.__N//2:-1]=npy.conjugate(self.__Nf[1:self.__N//2])
        #print(self.__Nf)

    '''
    Get noise signal in time domain from signal in frequency domain (inverse FFT)

    https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

    If whitening option is set to true signal is normalized. The whitened noise should be a gaussian centered on 0 and of width 1.

    '''

    def _genNtFromNf(self,whitening):
        # We whiten that with the average PSD
        self.__Nt[:]=npy.fft.ifft(self.__Nf/npy.sqrt(self.__PSD),norm='ortho').real if whitening else 1/npy.sqrt(self.__N)*npy.fft.ifft(self.__Nf,norm='ortho').real
        self.__Nf2[:]=npy.fft.fft(self.__Nt,norm='ortho') # Control
        #print(self.__Nt)
    '''
    The full procedure to produce a noise sample
    '''
        
    def getNewSample(self,whitening=True):
        if not(isinstance(whitening,bool)):
            raise TypeError("Un booléen est attendu")
        self._genNfFromPSD()
        self._genNtFromNf(whitening=whitening)
        return self.__Nt.copy()
   
    '''
    Plot macros
    '''
    
    def plotNoise(self):
        plt.plot(self.__T, self.__Nt,'-',label='n(t)')
        plt.title('Réalisation d\'un bruit dans le domaine temporel')
        plt.xlabel('t (s)')
        plt.ylabel('n(t) (no unit)')
        plt.grid(True, which="both", ls="-")

    def plotNoise1D(self):
        _, bins, _ = plt.hist(self.__Nt,bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__Nt)
        print("Largeur de la distribution temporelle (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

        plt.plot(bins, best_fit_line)
        plt.title('Bruit temporel normalisé')

    def plotTF(self,fmin=None,fmax=None):
    
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf[ifmin:ifmax])/npy.sqrt(self.__N),'.',label='n_tilde(f)')
        plt.title('Réalisation d\'un bruit dans le domaine fréquentiel')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/Hz)')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
                
    def plotTF2(self,fmin=None,fmax=None):
    
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Signal de contrôle (FFT du bruit temporel)')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/Hz)')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
        
    def plotPSD(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__PSD[ifmin:ifmax])/npy.sqrt(self.__N),'-',label='Sn(f)')
        plt.title('PSD analytique du bruit')
        plt.xlabel('f (Hz)')
        plt.ylabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    def plotPSD1D(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)
        plt.hist(npy.abs(self.__Nf[ifmin:ifmax]),bins=100, range=[-8e-23, 8e-23])
        plt.title('PSD analytique du bruit')
        plt.xlabel('Sn(f)^(1/2) (1/sqrt(Hz))')

        
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

->fe           : sampling frequency, in Hz (Default is 2048)
->kindTemplate : generator type: 'EM' or 'EOB' (Default is 2048)
->fDmin        : the minimal sensitivity of the detector (frequency)
->fDmax        : the maximal sensitivity of the detector (frequency)



'''

class GenTemplate:
    """Classe générant des signaux d'OG"""
    def __init__(self,fe=2048,fDmin=10,fDmax=4000,kindTemplate='EM'):
        """constructeur: Choix de la fréquence d'échantillonnage fe et la fréquence minimale du détecteur fDmin"""
        if not(isinstance(fe,int) or isinstance(fe,float)) and not(isinstance(fDmin,int) or isinstance(fDmin,float)):
            raise TypeError("fe et fDmin doivent être des ints ou des floats")
    
        if not(isinstance(kindTemplate,str)):
            raise TypeError("kindTemplate doit être de type str")
        if kindTemplate!='EM' and kindTemplate!='EOB':
            raise ValueError("Les seules valeurs autorisées pour kindTemplate sont 'EM' et 'EOB'")
        
        
        self.__fDmin=0.9*fDmin        # Frequence min pour le signal reco
        self.__fDmind=fDmin           # Frequence min pour le detecteur
        self.__fDmaxd=fDmax           # Frequence max pour le detecteur
        self.__fe=fe                  # Frequence d'echantillonage en Hz

        self.__delta_t=1/self.__fe    # Période
        self.__Tdepass=0.1            # Temps rajouté à la fin du signal pour éviter les pics de TF en mode EM
        self.__m1=10*Msol             # m1,m2 donnee en masse solaire
        self.__m2=10*Msol             #
        self.__D=1*MPC                # D en Mpc
        self.__Phic=0                 # Phase initiale, en radian
        self.__M=self.__m1+self.__m2  # Masse totale du systeme
        self.__eta=self.__m1*self.__m2/(self.__M**2)   # Masse réduite
        self.__MC=npy.power(self.__eta,3./5.)*self.__M # Chirp mass
        self.__rSC=2.*G*self.__MC/(c**2)               # Rayon Schwarzchild du chirp
        self.__tSC=self.__rSC/c                        # Temps Schwarzchild du chirp
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M) # Fréquence de la dernière orbite stable
        self.__Tchirp=self.getTchirp(self.__fDmin/2)        # Durée du chirp entre fmin et la coalescence
        
        # Temps entre fDmin et fDmind (approximation EM)
        # Ce temps est utiliser pour la fenetre de Blackman en entrée
        
        self.__Tblack=self.getTchirp(self.__fDmin/2)-self.getTchirp(self.__fDmind/2)
        self.__kindTemplate=kindTemplate                    # EM ou EOB
    
        #rajout de Tdepass pour eviter les oscillation dans la TF
        Ttot=self.__Tchirp+self.__Tdepass
        N=int(Ttot*self.__fe) # Nombre de points echantillonnés
        
        # On régle le nombre de points afin qu'il soit pair (pour la TF)
        # Une puissance de 2 serait encore mieux mais bon pour l'instant on reste avec ça
        self.__N=N+N%2
        self.__delta_f=self.__fe/self.__N          # Step en fréquence
        self.__Ttot=float(self.__N)/float(self.__fe)
        self.__TchirpAndTdepass=self.__Ttot
        self.__tc=self.__Ttot-self.__Tdepass       # Where we put the chirp
        
        self.__T=npy.arange(self.__N)*self.__delta_t  # Tableau contenant le temps de chaque mesure
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f # Toutes les fréquences pour la TF (double sided)
        self.__St=npy.zeros(self.__N)                # Template dans le domaine temporel
        self.__Sf=npy.zeros(self.__N,dtype=complex)  # Template dans le domaine fréquenciel

        self.__Filt=npy.zeros(self.__N)                 # Filtre adapté dans le domaine temporel
        self.__Filtf=npy.zeros(self.__N,dtype=complex)  # Filtre adapté le domaine fréquenciel
              

        
    '''
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
    Method updating the templates properties, should be called right after the initialization
    '''
    
    def majParams(self,m1,m2,D=None,Phic=None):
        self.__m1=self.__m1 if m1 is None else m1*Msol
        self.__m2=self.__m2 if m2 is None else m2*Msol
        self.__D=self.__D if D is None else D*MPC
        self.__Phic=self.__Phic if Phic is None else Phic
        self.__M=self.__m1+self.__m2
        self.__eta=self.__m1*self.__m2/(self.__m1+self.__m2)**2
        self.__MC=npy.power(self.__eta,3./5.)*self.__M
        self.__rSC=2.*G*self.__MC/c**2
        self.__tSC=self.__rSC/c
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M)
        self.__Tchirp=self.getTchirp(self.__fDmin/2)
        self.__Tblack=self.getTchirp(self.__fDmin/2)-self.getTchirp(self.__fDmind/2)
        Ttot=self.__Tchirp+self.__Tdepass
        N=int(Ttot*self.__fe) # Nombre de points echantillonnés
        self.__N=N+N%2
        self.__Ttot=float(self.__N)/float(self.__fe)
        self.__TchirpAndTdepass=self.__Ttot
        self.__delta_f=self.__fe/self.__N
        self.__tc=self.__Tchirp
        
        del self.__T
        self.__T=npy.arange(self.__N)*self.__delta_t
        del self.__F
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        del self.__St
        self.__St=npy.zeros(self.__N)

        self.__Stinit=npy.zeros(self.__N)
        del self.__Sf
        self.__Sf=npy.zeros(self.__N,dtype=complex)

        self.__Sfn=npy.zeros(self.__N,dtype=complex)
        del self.__Filtf
        self.__Filtf=npy.zeros(self.__N,dtype=complex)
        del self.__Filt
        self.__Filt=npy.zeros(self.__N)
        self.__norm=1.

        
    '''
    Produce the time serie of the template
    
    Here there are 2 option, EM or EOB
    
    EOB uses the option SEOBNRv4, which includes the merger and ringdown stages (means that one can remove TDepass here
    
    '''

    def _genStFromParams(self):

        itmin=0
        itmax=int(self.__tc/self.__delta_t)
        print('Generating a template of type',self.__kindTemplate,'with masses (m1/m2)=(',self.__m1/Msol,'/',self.__m2/Msol,")")
        
        if self.__kindTemplate=='EM':
            self.__St[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            self.__Stinit[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            print("Signal duration generated",self.__Ttot,self.getTchirp(self.__fDmin/2)+self.__Tdepass)
            print("Enters into detector frequency range at t=",self.__Tblack)
            print("Coalescence occurs at t=",self.__Tchirp)
        elif self.__kindTemplate=='EOB':
        
            #
            # Use pyCBC here:
            # https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
            #
        
            # The signal starting at frequency fDmin
            hp,hq = get_td_waveform(approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=self.__fDmin)

            # The signal starting at frequency fDmind
            hpd,hqd = get_td_waveform(approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=self.__fDmind)

            c=0
            for c in range(len(hp)-1,-1,-1): # Don't consider 0 at the end
                if abs(hp.numpy()[c])>1e-35:
                    break
            hp_tab=hp.numpy()[:c]
            
            # Ok this is nasty, in the future we should update self.__N to the right value
            # To be done definitely

            if (len(hp_tab)>self.__N):
                hp_tab=hp_tab[len(hp_tab)-self.__N:]

            c2=0
            
            for c2 in range(len(hpd)-1,-1,-1): # Don't consider 0 at the end
                if abs(hpd.numpy()[c2])>1e-35:
                    break
            hp_tabd=hpd.numpy()[:c2]
            if (len(hp_tabd)>self.__N):
                hp_tabd=hp_tabd[len(hp_tabd)-self.__N:]


            if hp.sample_times.numpy()[c]>=0:
                self.__TchirpAndTdepass=len(hp_tab)*self.__delta_t
                self.__TchirpAndTdepass2=len(hp_tabd)*self.__delta_t
                self.__Tblack=self.__TchirpAndTdepass-self.__TchirpAndTdepass2
                print("Chirp times:",self.__Ttot,self.__TchirpAndTdepass2,self.__TchirpAndTdepass)
                self.__St[:]=npy.concatenate((hp_tab,npy.zeros(self.__N-len(hp_tab))))
                self.__Stinit[:]=npy.concatenate((hp_tab,npy.zeros(self.__N-len(hp_tab))))
            else:
                raise Exception("Erreur le temps de coalescence n'est pas pris en compte dans le template")
        else:
            raise ValueError("Valeur pour kindTemplate non prise en charge")
        
    '''
    Create the fourier transform of the signal
    We add screening at the beginning and at the end (for EM only at the end)
    in order to avoid discontinuity
    '''
    
    def _genSfFromSt(self):
        S=npy.zeros(self.__N)
        S[:]=self.__St[:]
 
        # Fenetre de Blackmann à basse fréquence
        # Compute the bins where the signal will be screend to avoid discontinuity
        # Should be between fDmin and fDmind
        #
        
        iwmin=0
        c=0
        if abs(S[0])<1e-35:
            for c in range(len(S)):
                if (abs(S[c])>=1e-35):
                    iwmin=c
                    break
        #print(c)
        iwmax=int(self.__Tblack/self.__delta_t)
        print("Blackman window will be applied to the signal start between times",iwmin*self.__delta_t,"and",iwmax*self.__delta_t)
 
        S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[:iwmax-iwmin]
         
        #Fenetre de Blackmann à haute fréquence (EM only)
        #
    
        if self.__kindTemplate=='EM':
            time=0.01*npy.sqrt(2048/self.__fe)
            weight=int(time/self.__delta_t)
            iwmin=int(self.__Tchirp/self.__delta_t)-weight
            iwmax=int(self.__Tchirp/self.__delta_t)+weight
            S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[iwmin-iwmax:]
            
        # Compute the FFT of the signal
        # We use binning dependent params here, Sf value defined this way does
        # depend on frequency, as the norm factor is 1/sqrt(N) only
        #
        
        self.__Sf[:]=npy.fft.fft(S,norm='ortho') # Params dep du binning
 
        del S
    
    '''
    Compute rhoOpt, which is the output we would get when filtering the template with noise only
    In other words this corresponds to the filtered power we should get in absence of signal
    
    The frequency range here is in principle the detector one
    But one could also used full frequency range (actually this is a good question ;-) )
    
    Often called the optimal SNR or SNRmax
    
    '''
      
    def rhoOpt(self,Noise):
                
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)

        # <x**2(t)> calculation
        # Sf and PSD have the same normalisations
        #
        ropt=npy.sqrt(2*(self.__Sf[ifmin:ifmax]*npy.conjugate(self.__Sf[ifmin:ifmax])/Noise.PSD[ifmin:ifmax]).sum().real)
        
        #
        # Definition of the SNRmax used here is available in Eqn 26 of the foll. paper:
        #
        # https://arxiv.org/pdf/gr-qc/9402014.pdf
        #

        print('SNRmax (No angular, antenna effects, D=1Mpc) over the total period is equal to',ropt)
        
        self.__norm=ropt
        return self.__norm

    '''
    Then we normalize the signal, first with PSD (like the noise)
    Then with rhoOpt (to get SNR=1)
    With this method to get a signal at a given SNR one just have to rescale it by a factor SNR
    '''

    def _whitening(self,kindPSD,Tsample,norm):
        if kindPSD is None:
            print('No PSD given, one cannot normalize!!!')
            self.__St[:]=npy.fft.ifft(self.__Sf,norm='ortho').real
            return self.__St
        
        # We create a noise instance to do the whitening
        
        Noise=GenNoise(Ttot=self.__Ttot,fe=self.__fe, kindPSD=kindPSD,fmin=self.__fDmin,fmax=self.__fDmaxd)
        Noise.getNewSample()
        # Important point, the noise seq the same length at signal here, it prevent binning pbs
        # We just produce the PSD here, to do the weighting
        rho=self.rhoOpt(Noise=Noise)         # Get SNRopt
        Sf=npy.zeros(self.__N,dtype=complex)
        Sf=self.__Sf/npy.sqrt(Noise.PSD)     # Whitening of the signal in the frequncy domain
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
    Generate a signal over a period Tsample
    
    tc is the time at which the signal is ending, it is set to 0.95 Tsample by default
    '''
    
    def getNewSample(self,kindPSD='flat',Tsample=1,tc=None,norm=True):
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!=None:
            raise ValueError("Les seules valeurs autorisées sont None, 'flat', et 'analytic'")
        
        if not(isinstance(norm,bool)):
            raise TypeError("Un booléen est attendu")
            
        self._genStFromParams()
        self._genSfFromSt()
        self._whitening(kindPSD,Tsample,norm)
        
        tc= 0.95*Tsample if tc==None else min(tc,0.95*Tsample)
        N=int(Tsample*self.__fe)
        S=npy.zeros(N)
        
        itc=int(tc/self.__delta_t)
        print('Size of sample',N)
        if tc<=self.__Ttot:
            S[:itc]=self.__St[-itc:] # There will be 0s at the start
        else:
            S[itc-self.__N:itc]=self.__St[:] # There will be 0s at the end
        
        return S
        
    def getSameSample(self,Tsample=1,tc=None):
        tc= 0.95*Tsample if tc==None else min(tc,0.95*Tsample)
        N=int(Tsample*self.__fe)
        S=npy.zeros(N)
        
        itc=int(tc/self.__delta_t)
        if tc<=self.__Ttot:
            S[:itc]=self.__St[-itc:]
        else:
            S[itc-self.__N:itc]=self.__St[:]
        return S
        
    def getInitSample(self,Tsample=1,tc=None):
        tc= 0.95*Tsample if tc==None else min(tc,0.95*Tsample)
        N=int(Tsample*self.__fe)
        S=npy.zeros(N)
        
        itc=int(tc/self.__delta_t)
        if tc<=self.__Ttot:
            S[:itc]=self.__Stinit[-itc:]
        else:
            S[itc-self.__N:itc]=self.__Stinit[:]
        return S
        
    
    def plot(self,Tsample=1,tc=0.95,SNR=1):
        N=int(Tsample*self.__fe)
        T=npy.arange(N)*self.__delta_t
        
        plt.plot(T, self.getSameSample(Tsample=Tsample,tc=tc)*SNR,'-',label='h(t)')
        #plt.plot(T, self.getInitSample(Tsample=Tsample,tc=tc)*SNR,'-',label='h(t)')
        plt.title('Template dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    def plotSignal1D(self):
        _, bins, _ = plt.hist(self.__St,bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__St)
        print("Largeur de la distribution temporelle (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

        plt.plot(bins, best_fit_line)
        plt.title('Bruit temporel normalisé')


    def plotFilt(self,Tsample=1,SNR=1):
        N=int(Tsample*self.__fe)
        T=npy.arange(N)*self.__delta_t
        
        plt.plot(T, self.__Filt,'-',label='filt(t)')
        plt.title('Template filtré dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('rho(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    def plotTF(self):
                        
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sf[ifmin:ifmax]),'.')
    
    def plotTFn(self):
                                
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sfn[ifmin:ifmax]),'.')
        
    
    @property
    def length(self):
        return self.__N
        
    def duration(self):
        return self.__TchirpAndTdepass

        
    def filtre(self):
        return self.__Filtf
    
    def norma(self):
        return self.__norm
    
#########################################################################################################################################################################################
'''
Class handling random sequence generation (signal+noise)

Option:

->fe           : sampling frequency, in Hz (Default is 2048)
->kindTemplate : generator type: 'EM' or 'EOB' (Default is 2048)
->Ttot         : signal duration, in seconds (Default is 1)
->fe           : sampling frequency, in Hz (Default is 2048)
->kindPSD      : noise type: 'flat' or 'analytic' (Default is flat)
->nsamp        : for each frequency the PSD is considered as a gaussian centered on PSD
                 of width PSD/sqrt(nsamp), number of samples used to estimate the PSD
            
flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution, still it's not based on data


'''

class GenSequence:
    """Classe générant des signaux d'OG"""
    def __init__(self,fe=2048,fDmin=20,kindTemplate='EM',time=10.,SNR=7.5,m1=10,m2=10):
        """constructeur: Choix de la fréquence d'échantillonnage fe et la fréqeunce minimale du détecteur fDmin"""
        if not(isinstance(fe,int) or isinstance(fe,float)) and not(isinstance(fDmin,int) or isinstance(fDmin,float)):
            raise TypeError("fe et fDmin doivent être des ints ou des floats")
    
        if not(isinstance(kindTemplate,str)):
            raise TypeError("kindTemplate doit être de type str")
        if kindTemplate!='EM' and kindTemplate!='EOB':
            raise ValueError("Les seules valeurs autorisées pour kindTemplate sont 'EM' et 'EOB'")
        
        self.__dbg=0
        self.__fDmin=0.9*fDmin        # Frequence min pour le signal reco
        self.__fDmind=fDmin           # Frequence min pour le signal reco
        self.__fe=fe                  # Frequence d'echantillonage en Hz
        self.__delta_t=1/self.__fe    # Période
        self.__Tdepass=0.1            # Temps rajouté à la fin du signal pour éviter les pics de TF
        self.__m1=m1*Msol             # m1,m2 donnee en masse solaire
        self.__m2=m2*Msol             #
        self.__D=1*MPC                # D en Mpc
        self.__Phic=0                 # Phase initiale, en radian
        self.__M=self.__m1+self.__m2  # Masse totale du systeme
        self.__eta=self.__m1*self.__m2/(self.__M**2)   # Masse réduite
        self.__MC=npy.power(self.__eta,3./5.)*self.__M # Chrip mass
        self.__rSC=2.*G*self.__MC/(c**2)               # Rayon Schwarzchild du chirp
        self.__tSC=self.__rSC/c                        # Temps Schwarzchild du chirp
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M) # Fréquence de la dernière orbite stable
        self.__kindTemplate=kindTemplate                    # EM ou EOB
    
        
        self.__TGenerator=GenTemplate(fe=self.__fe,kindTemplate=self.__kindTemplate)
        self.__SNR=SNR
        self.__TGenerator.majParams(self.__m1/Msol,self.__m2/Msol)
        
        self.__N=int(time*self.__fe) # Nombre de points echantillonnés
        self.__time=time
        if self.__dbg==1:
            self.__N=int(self.__TGenerator.duration()*self.__fe)
            self.__time=self.__TGenerator.duration()
        
        self.__delta_f=self.__fe/self.__N   # Step en fréquence
        
        self.__T=npy.arange(self.__N)*self.__delta_t  # Tableau contenant le temps de chaque mesure
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f # Toutes les fréquences pour la TF (double sided)
        self.__St=npy.zeros(self.__N)                # Sequence dans le domaine temporel
        self.__Sf=npy.zeros(self.__N,dtype=complex)  # Sequence dans le domaine fréquentiel
        self.__Filt=npy.zeros(self.__N)                 # Filtrage domaine temporel
        self.__Filtf=npy.zeros(self.__N,dtype=complex)  # Template dans le domaine fréquenciel
        

        self.__cTime=npy.random.uniform(self.__TGenerator.duration(),time) # Where is the chirp injected
        
        if self.__dbg==1:
            self.__cTime=self.__TGenerator.duration()
        
        # The signal is normalized to the noise power, so one just have to multiply it by SNR to get a signal of the corresponding intensity
        self.__sig=SNR*self.__TGenerator.getNewSample(kindPSD='analytic',tc=self.__cTime,Tsample=self.__time,norm=True)
        
        self.__NGenerator=GenNoise(self.__time,fe=self.__fe,kindPSD='analytic')
        self.__background=self.__NGenerator.getNewSample()

       
        print('We produced a random sequence of',self.__time,'seconds of noise')
        print('A signal of duration',self.__TGenerator.duration(),'is ending at t=',self.__cTime)
        print('Length of the Fourier transform for the scan will be',2*npy.ceil(self.__TGenerator.duration()),'seconds')
        self.__FFTstep=npy.ceil(self.__TGenerator.duration())
        self.__Nstep=int(self.__time/self.__FFTstep)-1
        self.__npts=int(2*self.__FFTstep/self.__delta_t)
        
        if self.__dbg==1:
            self.__FFTstep=self.__TGenerator.duration()
            self.__npts=int(self.__FFTstep/self.__delta_t)
            self.__Nstep=1
                    
        print('We will scan the sequence with',self.__Nstep,'FFTs')
        
        
        self.__St=self.__sig+self.__background
        
        
    def matchedFilter(self):
    
        self.__Filt=npy.zeros(self.__N) # We provide the filter value over the whole sequence
        npts=self.__npts
        print(self.__npts)
        for i in range(self.__Nstep):
            imin=int(i/2*npts)
            imax=int((i/2+1)*npts)
            #print('Analyse data in the range',imin*self.__delta_t,'--',imax*self.__delta_t)
            Sfo=npy.zeros(npts,dtype=complex)
            Sfo=npy.fft.fft(self.__St[imin:imax],norm='ortho') # FFT of the signal chunk
            Filter=self.__TGenerator.filtre() # Get the template filter
            #print('Avant filtrage du signal',len(Sfo),len(Filter))

            # Logically Sfo is larger than filter, but the sampling rate being the same there is a f bin difference
            # So we resample Filter to have the same granularity

            n = len(Sfo)
            f = interp1d(npy.linspace(0, 1, len(Filter)), Filter, 'linear') # Rescale the filter to align it with the FFT
            Sf_scaled=f(npy.linspace(0, 1, n)) # Here we account for the normalisation
            
            if (self.__dbg==1):
                Sf_scaled=Filter
                    
            df=self.__fe/npts
                    
            ifmax=int(min(1500,self.__fe/2)/df)
            ifmin=int(self.__fDmind/df)

            dP=(self.__TGenerator.norma())

            self.__Filt[imin:imax]=1/dP*npy.sqrt(npts)*(abs(npy.fft.ifft(Sfo*Sf_scaled,norm='ortho')).real)
            
            '''
            ropta=((1/dP)/npy.sqrt(npts)*abs(Sfo[ifmin:ifmax]*Sf_scaled[ifmin:ifmax]).sum().real)
            reff=((1/dP)/npy.sqrt(npts)*(abs(npy.fft.ifft(Sfo*Sf_scaled,norm='ortho'))).sum().real)
                    
            print("Noise power per time bin:",dP)
            print("SNR measured from integral:",ropta)
            print("SNR measuredfrom filter:",reff)
            '''

    def plotSeq(self):
        
        plt.plot(self.__T, self.__St,'-',label='h(t)')
        #plt.plot(self.__T, self.__sig,'-',label='h(t)')
        #plt.plot(T[::16], signal.resample(self.getSameSample(Tsample=Tsample)*SNR,int(N/16))-(self.getSameSample(Tsample=Tsample)*SNR)[::16],'-',label='hs(t)')
        #plt.plot(T[::16], signal.resample(self.getSameSample(Tsample=Tsample)*SNR,int(N/16)),'-',label='hsp(t)')
        plt.title('Template dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    def plotMF(self,Tsample=1,SNR=1):
        N=len(self.__Filt)
        T=npy.arange(N)*self.__delta_t
        print(N)

        plt.plot(T, self.__Filt,'-',label='filt(t)')
        plt.title('Template filtré dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('rho(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    def plotHt1D(self):
        _, bins, _ = plt.hist(self.__St,bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__St)
        print("Largeur de la distribution temporelle (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.yscale('log')
        plt.plot(bins, best_fit_line)
        plt.title('Signal temporel normalisé')
        
    def plotMF1D(self):
        _, bins, _ = plt.hist(self.__Filt,bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__Filt)
        print("Largeur de la distribution filtrée (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.yscale('log')
        plt.plot(bins, best_fit_line)
        plt.title('Signal filtré temporel normalisé')
#################################################################################################################################################################################################
        
class GenDataSet:
    """Classe générant des signaux des sets de training 50% de \"signaux+bruits\" 50% \"de bruits\""""
    def __init__(self,mint=(10,50),NbB=1,tcint=(0.75,0.95),kindPSD='flat',kindTemplate='EM',Ttot=1,fe=2048,kindBank='linear',paramFile=None):
        
        if paramFile is None:
            self.__Ttot=Ttot
            self.__fe=fe
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
        
        self.__NGenerator=GenNoise(Ttot=self.__Ttot,fe=self.__fe,kindPSD=self.__kindPSD)
        self.__TGenerator=GenTemplate(fe=self.__fe,kindTemplate=self.__kindTemplate)
        
        
        self._genGrille()   # matrice contenant les colonnes m1,m2
        self._genNoiseSet() # matrice contenant les bruits
        self._genSigSet()   # matrice des donnees (signaux temporels) avec Ntemplate puis des zeros
        
        self.__Labels=npy.concatenate((npy.zeros(self.__Ntemplate*self.__NbB,dtype=int),npy.ones(self.__Ntemplate*self.__NbB,dtype=int))) # 0 <-> signal , 1 <-> noise
    
    def _readParamFile(self,paramFile):
        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='Ttot' or lignes[1][0]!='fe' or lignes[2][0]!='kindPSD' or lignes[3][0]!='mint' or lignes[4][0]!='tcint' or lignes[5][0]!='NbB' or lignes[6][0]!='kindTemplate' or lignes[7][0]!='kindBank' or len(lignes)!=8:
            raise Exception("Erreur dans le fichier de paramètres")

        self.__Ttot=float(lignes[0][1])
        self.__fe=float(lignes[1][1])
        self.__kindPSD=lignes[2][1]
        self.__mmin=min(float(lignes[3][1]),float(lignes[3][2]))
        self.__mmax=max(float(lignes[3][1]),float(lignes[3][2]))
        self.__tcmin=self.__Ttot*max(min(float(lignes[4][1]),float(lignes[4][2])),0.5)
        self.__tcmax=self.__Ttot*min(max(float(lignes[4][1]),float(lignes[4][2])),1.)
        self.__NbB=int(lignes[5][1])
        self.__kindTemplate=lignes[6][1]
        kindBank=lignes[7][1]
        if (kindBank=='linear' or kindBank=='optimal'):
            self.__kindBank=kindBank
        else:
            raise ValueError("Les valeurs pour la banque sont 'optimal' ou 'linear'")
            
    '''
    Grille des masses à generer
    '''
    
    def _genGrille(self):
        if self.__kindBank=='linear':
            self.__Ntemplate=int((self.__mmax-self.__mmin+1)*(self.__mmax-self.__mmin+2)/2)
            self.__GrilleMasses=npy.ones((self.__Ntemplate,2))
            self.__GrilleMasses.T[0:2]=self.__mmin
            c=0
            for i in range(0,int(self.__mmax-self.__mmin+1)):
                for j in range(0,i+1):
                        self.__GrilleMasses[c][0]+=i
                        self.__GrilleMasses[c][1]+=j
                        c+=1
        else:
            with open(os.path.dirname(__file__)+'/params/outputVT_sorted.dat') as mon_fichier:
                mon_fichier_reader = csv.reader(mon_fichier, delimiter=' ')
                M = npy.array([l for l in mon_fichier_reader],dtype=float)

            self.__GrilleMasses=((M[(M.T[0]>=self.__mmin) & (M.T[1]>=self.__mmin) & (M.T[0]<=self.__mmax) & (M.T[1]<=self.__mmax)]).T[:2]).T
            self.__Ntemplate=len(self.__GrilleMasses)
        
    def _genNoiseSet(self):
        self.__Noise=npy.zeros((self.__Ntemplate*2*self.__NbB,self.__NGenerator.length))
        for i in range(0,self.__Ntemplate*self.__NbB*2):
            self.__Noise[i]=self.__NGenerator.getNewSample()
        
    def _genSigSet(self):
        self.__Sig=npy.zeros((self.__Ntemplate*2*self.__NbB,self.__NGenerator.length))
        
        c=0
        for i in range(0,self.__Ntemplate):
            self.__TGenerator.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1])
            self.__Sig[c]=self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,Tsample=self.__Ttot,tc=npy.random.uniform(self.__tcmin,self.__tcmax))
            c+=1
            for j in range(1,self.__NbB):
                self.__Sig[c]=self.__TGenerator.getSameSample(Tsample=self.__Ttot,tc=npy.random.uniform(self.__tcmin,self.__tcmax))
                c+=1
        
    def getDataSet(self,SNRopt=1):
        return ((self.__Sig.T*npy.random.uniform((min(SNRopt)),(max(SNRopt)),size=self.Nsample)).T if isinstance(SNRopt,tuple) else self.__Sig*(SNRopt))+self.__Noise
        
    def plot(self,i,SNRopt=1):
        plt.plot(self.__NGenerator.T,self.getDataSet(SNRopt)[i],'.')
    
    def saveGenerator(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        fichier=dossier+str(self.__NbB)+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+self.__kindBank+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
    
    @classmethod
    def readGenerator(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
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
    parser.add_argument("--kindPSD","-kp",help="Type de PSD à choisir",choices=['flat', 'analytic'],default='flat')
    parser.add_argument("--kindTemplate","-kt",help="Type de PSD à choisir",choices=['EM', 'EOB'],default='EM')
    parser.add_argument("--time","-t",help="Durée du signal",type=float,default=1)
    parser.add_argument("-fe",help="Fréquence du signal",type=float,default=2048)
    parser.add_argument("-fmin",help="Fréquence minimale visible par le détecteur",type=float,default=20)
    parser.add_argument("-fmax",help="Fréquence maximale visible par le détecteur",type=float,default=1000)
    parser.add_argument("-snr",help="SNR (pour creation de sequence)",type=float,default=7.5)
    parser.add_argument("-m1",help="Masse du premier objet",type=float,default=20)
    parser.add_argument("-m2",help="Masse du deuxième objet",type=float,default=20)
    parser.add_argument("--set","-s",help="Choix du type de set par défaut à générer",choices=['train','test'],default=None)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de set",default=None)
    
    
    args = parser.parse_args()
    return args

'''
Main part of gendata.py



'''
def main():
    import gendata as gd
    
    args = parse_cmd_line()
    

    if args.cmd=='noise':
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax)
        NGenerator.getNewSample(whitening=True)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
        NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotTF2(fmin=args.fmin,fmax=args.fmax)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()
                    
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise1D()

        plt.legend()
        plt.show()
        
    elif args.cmd=='template':
        TGenerator=gd.GenTemplate(fe=args.fe,kindTemplate=args.kindTemplate)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=args.time,norm=True)
        
        plt.figure(figsize=(10,5))
        TGenerator.plotTF()
            
        plt.figure(figsize=(10,5))
        TGenerator.plotTFn()
        
        plt.figure(figsize=(10,5))
        #print(TGenerator.duration())
        #TGenerator.plot(Tsample=args.time,SNR=7.5)
        TGenerator.plot(Tsample=TGenerator.duration(),tc=TGenerator.duration(),SNR=args.snr)
                    
        plt.figure(figsize=(10,5))
        TGenerator.plotSignal1D()
        
        plt.legend()
        plt.show()

    elif args.cmd=='all':
        TGenerator=gd.GenTemplate(fe=args.fe,kindTemplate=args.kindTemplate)
        TGenerator.majParams(args.m1,args.m2)
        randt=npy.random.uniform(TGenerator.duration(),args.time)
        TGenerator.getNewSample(kindPSD=args.kindPSD,tc=randt,Tsample=args.time)
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax)
        NGenerator.getNewSample()
        
        #plt.figure(figsize=(10,5))
        #TGenerator.plotFilt(Tsample=args.time)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
        NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()
        TGenerator.plot(Tsample=args.time,tc=randt,SNR=7.5)
            
        #plt.legend()
        plt.show()
        
    elif args.cmd=='sequence':
        TSeq=gd.GenSequence(fe=args.fe,kindTemplate=args.kindTemplate,time=args.time,SNR=args.snr,m1=args.m1,m2=args.m2)
        TSeq.matchedFilter()

        
        plt.figure(figsize=(10,5))
        TSeq.plotSeq()
        
        
        plt.figure(figsize=(10,5))
        TSeq.plotMF()
        
        plt.figure(figsize=(10,5))
        TSeq.plotMF1D()
        
        plt.figure(figsize=(10,5))
        TSeq.plotHt1D()
        
        '''
        plt.figure()
        NGenerator.plotPSD(fmin=8)
        NGenerator.plotTF(fmin=8)
        
        plt.figure()
        NGenerator.plotNoise()
        TGenerator.plot(Tsample=args.time,tc=randt,SNR=7.5)
        '''
        plt.legend()
        plt.show()
        
        
        
    else: # Logically there is only set remaining, so dataset
        if args.set=='train':
            chemin='params/default_trainGen_params.csv'
            Generator=gd.GenDataSet(paramFile=chemin)
        elif args.set=='test':
            chemin='params/default_testGen_params.csv'
            Generator=gd.GenDataSet(paramFile=chemin)
        else:
            Generator=gd.GenDataSet(paramFile=args.paramfile)
        
        chemin = 'generators/'
        Generator.saveGenerator(chemin)

############################################################################################################################################################################################
if __name__ == "__main__":
    main()
    
    
