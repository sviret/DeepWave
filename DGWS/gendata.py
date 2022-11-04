import numpy as npy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
from pycbc.waveform import get_td_waveform

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 25,'axes.labelsize': 25,'axes.titlesize': 25,'figure.titlesize': 30,'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25,'legend.title_fontsize': 25,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [6.4*3.5,4.8*3.5]}
plt.rcParams.update(parameters)


'''
Class handling noise generation

Option:

->Ttot    : signal duration, in seconds (Default is 1)
->fe      : sampling frequency, in Hz (Default is 2048)
->kindPSD : noise type: 'flat' or 'analytic' (Default is flat)
->nsamp   : for each frequency the PSD is considered as a gaussian centered on PSD
            of width PSD/sqrt(nsamp), number of samples used to estimate the PSD

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution, still it's not based on data


'''

class GenNoise:
    """Classe générant du bruit"""
    def __init__(self,Ttot=1,fe=2048,kindPSD='flat',nsamp=16):
        """constructeur: Choix du temps total Ttot et de sa fréquence d'échantillonnage fe et du type de bruit kindPSD"""
        if not((isinstance(Ttot,int) or isinstance(Ttot,float)) and (isinstance(fe,int) or isinstance(fe,float))):
            raise TypeError("Ttot et fe doivent être des ints ou des floats")
        if not(isinstance(kindPSD,str)):
            raise TypeError("kindPSD doit être de type str")
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées pour kindPSD sont 'flat' et 'analytic'")
            
        self.__Ttot=Ttot #temps total du signal
        self.__fe=fe #frequence d'echantillonage
        self.__N=int(self.__Ttot*self.__fe) #nombre de points temporel
        self.__delta_t=1/self.__fe #pas de temps
        self.__delta_f=self.__fe/self.__N #pas de fréquence
        
        self.__T=npy.arange(self.__N)*self.__delta_t #vecteur des temps
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f #vecteur des frequences [0->fe/2,-fe/2->-1]
        #print(self.__F)
        self.__kindPSD=kindPSD
        self.__nsamp=nsamp
        self._genPSD() #Génération de la PSD du bruit constant dans le cas d'un bruit blanc (f>0)
        
        self.__Nt=npy.zeros(self.__N) #vecteur d'une realisation du bruit
        self.__Nf=npy.zeros(self.__N,dtype=complex) #TF de la realisation du bruit
        self.__Nfr=npy.zeros(self.__N)
        self.__Nfi=npy.zeros(self.__N)
        
    '''
    Noise generation for analytic option, account for shot, thermal, quantum and seismic noises
    This is the one sided PSD, as defined in part IV.A of:
    https://arxiv.org/pdf/gr-qc/9301003.pdf
    
    I
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
        
        
        #return npy.sqrt(2)*(Squant+Sshot+Sthermal+Sseismic)
        return (Squant+Sshot+Sthermal+Sseismic)
        #return npy.sqrt(2)*Squant
    
    '''
    Produce the PSD, so in the noise power in frequency domain
    '''

    def _genPSD(self):
        self.__PSD=npy.ones(self.__N)
        if self.__kindPSD=='flat':
            sigma=1e-21
            self.__PSD[:]=sigma**2
        elif self.__kindPSD=='analytic':
            self.__PSD[:]=self.Sh(abs(self.__F[:]))

    '''
    Option change
    '''

    def changePSD(self,kindPSD):
        del self.__PSD
        if kindPSD!='flat' and kindPSD!='analytic':
            raise ValueError("Les seules valeurs autorisées sont 'flat' et 'analytic'")
        self.__kindPSD=kindPSD
        self._genPSD()

    '''
    Get noise signal in time domain from signal in frequency domain (inverse FFT)

    https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

    If whitening option is set to true signal is normalized. So in flat mode (white noise) the whitened noise should be a gaussian centered on 0 and of width 1.

    '''

    def _genNtFromNf(self,whitening):
        # We whiten that with the average PSD
        self.__Nt[:]=npy.fft.ifft(self.__Nf/npy.sqrt(self.__PSD),norm='ortho').real if whitening else npy.fft.ifft(self.__Nf,norm='ortho').real
    
 
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
    a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.

    PSD is single sided and Nf defined over the full range, so Nf(f)**2 should be centered around 2*PSD(f)

    '''
 
    def _genNfFromPSD(self):
        
        # The power at a given frequency is taken around the corresponding PSD value
        self.__Nfr[0:self.__N//2+1]=npy.random.normal(npy.sqrt(self.__PSD[0:self.__N//2+1]),npy.sqrt(self.__PSD[0:self.__N//2+1])/npy.sqrt(self.__nsamp))
        self.__Nfi[0:self.__N//2+1]=self.__Nfr[0:self.__N//2+1]

        # The initial phase is randomized
        for i in range(len(self.__Nfr)):
            phi0 = float(npy.random.randint(1000))/1000.*8*math.atan(1.);
            self.__Nfr[i]*=math.cos(phi0) # real part
            self.__Nfi[i]*=math.sin(phi0) # imaginary part
                
        # Then we can define the components
        self.__Nf[0:self.__N//2+1].real=self.__Nfr[0:self.__N//2+1]
        self.__Nf[0:self.__N//2+1].imag=self.__Nfi[0:self.__N//2+1]
        
        # OK so this line means that we iterate from the last to the middle with -1 steps
        self.__Nf[-1:-self.__N//2:-1]=npy.conjugate(self.__Nf[1:self.__N//2])
        
    def getNewSample(self,whitening=True):
        if not(isinstance(whitening,bool)):
            raise TypeError("Un booléen est attendu")
        self._genNfFromPSD()
        self._genNtFromNf(whitening=whitening)
        return self.__Nt.copy()
    
    def plot(self):
        plt.plot(self.__T, self.__Nt,'-',label='n(t)')
        plt.title('Réalisation d\'un bruit dans le domaine temporel')
        plt.xlabel('t (s)')
        plt.ylabel('n(t) (no unit)')
        
    def plotTF(self,fmin=None,fmax=None):
    
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Réalisation d\'un bruit dans le domaine fréquentiel')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/Hz)')
        plt.yscale('log')
        plt.xscale('log')
        
    def plotPSD(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)
        #plt.plot(self.__F[ifmin:ifmax],1/(npy.sqrt(2))*npy.sqrt(self.__PSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__PSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.title('PSD analytique du bruit')
        plt.xlabel('f (Hz)')
        plt.ylabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    def plotPSD1D(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)
        plt.hist(npy.abs(self.__Nf[ifmin:ifmax]),bins=100, range=[0, 4e-21])
        plt.title('PSD analytique du bruit')
        plt.xlabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        #plt.yscale('log')
        #print(self.__PSD[ifmin:ifmax])
        


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
->nsamp        :

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution, still it's not based on data


''''''
Class handling noise generation

Option:

->Ttot    : signal duration, in seconds (Default is 1)
->fe      : sampling frequency, in Hz (Default is 2048)
->kindPSD : noise type: 'flat' or 'analytic' (Default is flat)
->nsamp   : for each frequency the PSD is considered as a gaussian centered on PSD
            of width PSD/sqrt(nsamp), number of samples used to estimate the PSD

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution, still it's not based on data


'''

class GenTemplate:
    """Classe générant des signaux d'OG"""
    def __init__(self,fe=2048,fDmin=20,kindTemplate='EM'):
        """constructeur: Choix de la fréquence d'échantillonnage fe et la fréqeunce minimale du détecteur fDmin"""
        if not(isinstance(fe,int) or isinstance(fe,float)) and not(isinstance(fDmin,int) or isinstance(fDmin,float)):
            raise TypeError("fe et fDmin doivent être des ints ou des floats")
    
        if not(isinstance(kindTemplate,str)):
            raise TypeError("kindTemplate doit être de type str")
        if kindTemplate!='EM' and kindTemplate!='EOB':
            raise ValueError("Les seules valeurs autorisées pour kindTemplate sont 'EM' et 'EOB'")
        
        
        self.__fDmin=20               # Frequence min detecteur en Hz
        self.__fe=fe                  # Frequence d'echantillonage en Hz
        self.__delta_t=1/self.__fe    # Période
        self.__Tdepass=0.1            # Temps rajouté à la fin du signal pour éviter les pics de TF
        self.__m1=10*Msol             # m1,m2 donnee en masse solaire
        self.__m2=10*Msol             #
        self.__D=1*MPC                # D en Mpc
        self.__Phic=0                 # Phase initiale, en radian
        self.__M=self.__m1+self.__m2  # Masse totale du systeme
        self.__eta=self.__m1*self.__m2/(self.__M**2)   # Masse réduite
        self.__MC=npy.power(self.__eta,3./5.)*self.__M # Chrip mass
        self.__rSC=2.*G*self.__MC/(c**2)               # Rayon Schwarzchild du chirp
        self.__tSC=self.__rSC/c                        # Temps Schwarzchild du chirp
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M) # Fréquence de la dernière orbite stable
        self.__Tchirp=self.getTchirp(self.__fDmin/2)        # Durée du chirp entre fmin et la coalescence
        self.__kindTemplate=kindTemplate                    # EM ou EOB
    
        #rajout de Tdepass pour eviter les oscillation dans la TF puis padding jusqu'à la première seconde entière (npy.ceil)
        self.__Ttot=int(npy.ceil(self.getTchirp(self.__fDmin/2)+self.__Tdepass))
        self.__N=int(self.__Ttot*self.__fe) # Nombre de points echantillonnés
        self.__delta_f=self.__fe/self.__N   # Step en fréquence
        self.__tc=self.__Ttot#0.95*Ttot #if tc is None else min(tc,Ttot*0.95) #en secondes
        #print(self.getTchirp(self.__fDmin/2),self.__Ttot,self.__N)
        self.__T=npy.arange(self.__N)*self.__delta_t  # Tableau contenant le temps de chaque mesure
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f # Toutes les fréquences pour la TF (double sided)
        self.__St=npy.zeros(self.__N)                # Template dans le domaine temporel
        self.__Sf=npy.zeros(self.__N,dtype=complex)  # Template dans le domaine fréquenciel
        self.__Sfno=npy.zeros(self.__N,dtype=complex)
                
    def phi(self,t):
        return -npy.power(2.*(self.__tc-t)/(5.*self.__tSC),5./8.)
                        
    def h(self,t):
        A=npy.power(2./5.,-1./4.)*self.__rSC/(32*self.__D)
        #print(t,A*npy.power((self.__tc-t)/self.__tSC,-1./4.)*npy.cos(2*(self.phi(t)+self.__Phic)))
        return A*npy.power((self.__tc-t)/self.__tSC,-1./4.)*npy.cos(2*(self.phi(t)+self.__Phic))
        
    
    '''
    Here the time is computed with the equation (1.68) of the following document:

    http://sviret.web.cern.ch/sviret/docs/Chirps.pdf
    
    f0 is the frequency at which the detector start to be sensitive to the signal, we define t0=0
    The output is tc-t0
    Value is computed with the newtonian approximation, maybe we should change for EOB
    '''
        
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
        
        self.__Ttot=int(npy.ceil(self.getTchirp(self.__fDmin/2)+self.__Tdepass))
        self.__N=int(self.__Ttot*self.__fe)
        self.__delta_f=self.__fe/self.__N
        self.__tc=self.__Ttot
        
        del self.__T
        self.__T=npy.arange(self.__N)*self.__delta_t
        del self.__F
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        del self.__St
        self.__St=npy.zeros(self.__N)
        del self.__Sf
        self.__Sf=npy.zeros(self.__N,dtype=complex)
        #print(self.getTchirp(self.__fDmin/2),self.__Ttot,self.__N)
        
    '''
    Produce the time serie of the template
    '''

    def _genStFromParams(self):
        # From bin 0 to itmin-1 the signal is at 0, then put the signa
        itmin=int((self.__Ttot-self.__Tchirp-self.__Tdepass)/self.__delta_t)
        #print((self.__Ttot-self.__Tchirp-self.__Tdepass),itmin,self.__N)
        if self.__kindTemplate=='EM':
            self.__St[:]= npy.concatenate((npy.zeros(itmin),self.h(self.__T[itmin:-1]),npy.zeros(1)))
            #print(self.__St)
        elif self.__kindTemplate=='EOB':
        
            #
            # Use pyCBC here:
            # https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
            #
        
            fmin=self.get_f(self.getTchirp(self.__fDmin/2)+self.__Tdepass)*2
            hp,_ = get_td_waveform(approximant='SEOBNRv4', mass1=self.__m1/Msol,mass2=self.__m2/Msol,delta_t=self.__delta_t,f_lower=fmin)
            
            c=0
            for c in range(len(hp)-1,-1,-1): # Don't consider 0 at the end
                if abs(hp.numpy()[c])>1e-24:
                    break
            #print(c)
            hp_tab=hp.numpy()[:c]
            if hp.sample_times.numpy()[c]>=0:
                self.__TchirpAndTdepass=hp.sample_times.numpy()[c]-hp.sample_times.numpy()[0] # Total chirp length
                print(self.__TchirpAndTdepass)
                self.__St[:]=npy.concatenate((npy.zeros(self.__N-len(hp_tab)),hp_tab))
            else:
                raise Exception("Erreur le temps de coalescence n'est pas pris en compte dans le template")
        else:
            raise ValueError("Valeur pour kindTemplate non prise en charge")
        #print(len(self.__St))
        
    def _genSfFromSt(self):
        S=npy.zeros(self.__N)
        S[:]=self.__St[:]
        # Compute the bins where the signal will be screend to avoid discontinuity
        iwmin=int((self.__Ttot-self.__Tchirp-self.__Tdepass)/self.__delta_t) if self.__kindTemplate=='EM' else int((self.__Ttot-self.__TchirpAndTdepass)/self.__delta_t)
        iwmax=int((self.__Ttot-self.__Tchirp)/self.__delta_t) if self.__kindTemplate=='EM' else int((self.__Ttot-self.__TchirpAndTdepass+self.__Tdepass)/self.__delta_t)
        
        print("Blackman window will be applied to the signal start between bins",iwmin,"and",iwmax)
        #Fenetre de Blackmann à basse fréquence
        S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[:iwmax-iwmin]
        
        #Fenetre de Blackmann à haute fréquence (EM only)
        if self.__kindTemplate=='EM':
            time=0.01 #10ms
            weight=int(time/self.__delta_t)
            S[-weight:]*=npy.blackman(weight*2)[-weight:]
            
        #TF
        self.__Sf[:]=npy.fft.fft(S,norm='ortho')
        
        del S
    
    '''
    Compute rhoOpt, which is the output we would get when filtering the template with noise only
    In other words this corresponds to the power we should get in absence of signal
    
    '''
      
    def rhoOpt(self,Noise,kindPSD='flat',Tsample=1):
        ifmax=int(min(self.__fisco,self.__fe/2)/self.__delta_f) if self.__kindTemplate=='EM' else int(self.__fe/2)
        ifmin= int(max(self.get_f(Tsample)*2,self.__fDmin)/self.__delta_f)
        # <x**2(t)> calculation
        return 2*npy.sqrt(self.__delta_f*(self.__Sf[ifmin:ifmax]*npy.conjugate(self.__Sf[ifmin:ifmax])/Noise.PSD[ifmin:ifmax]).sum().real)
    
    def _whitening(self,kindPSD,Tsample,norm):
        if kindPSD is None:
            rho=self.rhoOpt(Tsample=Tsample)
            self.__St[:]=npy.fft.ifft(self.__Sf,norm='ortho').real/(rho if norm else 1)
            return self.__St
        
        # We create a noise instance to do the whitening
        
        Noise=GenNoise(Ttot=self.__Ttot,fe=self.__fe, kindPSD=kindPSD)
        rho=self.rhoOpt(kindPSD=kindPSD,Tsample=Tsample,Noise=Noise) # The measurement when template is filtered with noise only
        Sf=npy.zeros(self.__N,dtype=complex)
        Sf[:]=self.__Sf/npy.sqrt(2*Noise.PSD) # Whithening
        self.__St[:]=npy.fft.ifft(Sf,norm='ortho').real/(rho if norm else 1)
        print(len(self.__St))
    '''
    Generate a signal
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
        if tc<=self.__Ttot:
            S[:itc]=self.__St[-itc:]
        else:
            S[itc-self.__N:itc]=self.__St[:]
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
        
    
    def plot(self,Tsample=1,SNR=1):
        N=int(Tsample*self.__fe)
        T=npy.arange(N)*self.__delta_t
        print(N)
        plt.plot(T, self.getSameSample(Tsample=Tsample)*SNR,'-',label='h(t)')
        plt.title('Template dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (No Unit)')
        plt.grid(True, which="both", ls="-")
            
    def plotTF(self,fmin=None,fmax=None):
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)
        ifmin=int(self.__fDmin/self.__delta_f) if fmin is None else max(int(fmin/self.__delta_f),0)
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sf[ifmin:ifmax]),'.')
        
    
    @property
    def length(self):
        return self.__N
    
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
        return ((self.__Sig.T*npy.random.uniform(min(SNRopt),max(SNRopt),size=self.Nsample)).T if isinstance(SNRopt,tuple) else self.__Sig*SNRopt)+self.__Noise
        
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
    parser.add_argument("cmd", help="commande à choisir",choices=['noise', 'template', 'set', 'all'])
    parser.add_argument("--kindPSD","-kp",help="Type de PSD à choisir",choices=['flat', 'analytic'],default='flat')
    parser.add_argument("--kindTemplate","-kt",help="Type de PSD à choisir",choices=['EM', 'EOB'],default='EM')
    parser.add_argument("--time","-t",help="Durée du signal",type=float,default=1)
    parser.add_argument("-fe",help="Fréquence du signal",type=float,default=2048)
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
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD)
        NGenerator.getNewSample()
        
        plt.figure()
        NGenerator.plotPSD(fmin=8)
        NGenerator.plotTF(fmin=8)
            
        plt.figure()
        NGenerator.plot()

        plt.legend()
        plt.show()
    
    elif args.cmd=='template':
        TGenerator=gd.GenTemplate(fe=args.fe,kindTemplate=args.kindTemplate)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=args.time)
        
        plt.figure()
        TGenerator.plotTF()
            
        plt.figure()
        TGenerator.plot(Tsample=args.time)
            
        plt.legend()
        plt.show()

    elif args.cmd=='all':
        TGenerator=gd.GenTemplate(fe=args.fe,kindTemplate=args.kindTemplate)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=args.time)
        NGenerator=gd.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD)
        NGenerator.getNewSample()
        
        #plt.figure()
        #TGenerator.plotTF()
        
        plt.figure()
        NGenerator.plotPSD(fmin=8)
        NGenerator.plotTF(fmin=8)
        
        plt.figure()
        NGenerator.plot()
        TGenerator.plot(Tsample=args.time,SNR=7.5)
            
        #plt.legend()
        plt.show()
        
    else: # Logically there is only set remaining, so dataset
        if args.set=='train':
            chemin=os.path.dirname(__file__)+'/params/default_trainGen_params.csv'
            Generator=gd.GenDataSet(paramFile=chemin)
        elif args.set=='test':
            chemin=os.path.dirname(__file__)+'/params/default_testGen_params.csv'
            Generator=gd.GenDataSet(paramFile=chemin)
        else:
            Generator=gd.GenDataSet(paramFile=args.paramfile)
        
        chemin = os.path.dirname(__file__)+'/generators/'
        Generator.saveGenerator(chemin)

############################################################################################################################################################################################
if __name__ == "__main__":
    main()
    
    
