import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
import time
import matplotlib.cm as cm
import gen_noise as gn
import gen_template as gt

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)
  
#################################################################################################################################################################################################
'''
Class handling gravidational wave dataset production (either for training or testing)

Options:

--> mint     : Solar masses range (mmin,mmax) for the (m1,m2) mass grid to be produced
--> step     : Mass step between two samples
--> NbB      : Number of background realisations per template
--> tcint    : Range for the coalescence time within the last signal chunk (in sec, should be <Ttot)
               if the last chunk has the length T, the coalescence will be randomly placed within the range
               [tcint[0]*T,tcint[1]*T]
--> choice   : The template type (EOB or EM)
--> kindBank : The type of mass grid used (linear or optimal)
--> whitening: Frequency-domain or time-domain whitening (1 or 2) 
'''


class GenDataSet:

    """Classe générant des signaux des sets de training 50% dsignaux+bruits 50% de bruits"""

    def __init__(self,mint=(10,50),NbB=1,tcint=(0.75,0.95),kindPSD='flat',kindTemplate='EM',Ttot=1,fe=2048,kindBank='linear',paramFile=None,step=0.1,choice='train',length=100,whitening=1,ninj=0,customPSD=None):
    
                
        self.__custPSD=[]
        if customPSD!=None:
            self.__custPSD=customPSD

        self.__choice=choice
        self.__length=length
        self.__ninj=ninj
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")

        if self.__choice!='frame':
            start_time = time.time()
            print("Starting dataset generation")

            # Noise stream generator
            self.__NGenerator=gn.GenNoise(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,whitening=self.__whiten,customPSD=self.__custPSD)
            # Template stream generator
            self.__TGenerator=gt.GenTemplate(Tsample=self.__listTtot,fe=self.__listfe,kindTemplate=self.__kindTemplate,whitening=self.__whiten,customPSD=self.__custPSD)
    
            self.__listDelta_t=[] # pour le plot des portions de SNR
            self.__listSNRevolTime=[]
            self.__listSNRevol=[]
            self.__listfnames=[]
            self.__listSNRchunksAuto=[[] for x in range(self.__nTtot)]
            self.__listSNRchunksSpecial=[[] for x in range(self.__nTtot)]
            self.__listSNRchunksBalance=[[] for x in range(self.__nTtot)]    
            self.__tmplist=[]

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
    
        else:
            start_time = time.time()
            print("Starting input frame generation")

            # Noise stream generator
            self.__NGenerator=gn.GenNoise(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,whitening=self.__whiten,customPSD=self.__custPSD)
            self.__TGenerator=gt.GenTemplate(Tsample=self.__listTtot,fe=self.__listfe,kindTemplate=self.__kindTemplate,whitening=self.__whiten,customPSD=self.__custPSD)
            
            self.__listDelta_t=[] # pour le plot des portions de SNR
            self.__listSNRevolTime=[]
            self.__listSNRevol=[]
            self.__listfnames=[]
            self.__listSNRchunksAuto=[[] for x in range(self.__nTtot)]
            self.__listSNRchunksSpecial=[[] for x in range(self.__nTtot)]
            self.__listSNRchunksBalance=[[] for x in range(self.__nTtot)]
    
            self._genNoiseSequence() # The noises (one realization per template)
        
            ninj=int(self.__ninj)
            interval=self.__length/(self.__ninj+1)

            print(ninj,"signals will be injected in the data stream")
            self.__injections=[]
            
            for i in range(ninj):

                m1=npy.random.uniform(10,75)
                m2=npy.random.uniform(10,75)
                SNR=npy.random.uniform(3,20)
                self.__TGenerator.majParams(m1,m2)

                self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,Tsample=self.__TGenerator.duration(),norm=True)
                data=self.__TGenerator.signal()
    
    
                #randt=npy.random.uniform(self.__TGenerator.duration(),self.__length)
                randt=npy.random.normal((i+1)*interval,interval/5.)
                inj=[m1,m2,SNR,randt]
                self.__injections.append(inj)

                print("Injection",i,"(m1,m2,SNR,tc)=(",f'{m1:.1f}',f'{m2:.1f}',f'{SNR:.1f}',f'{randt:.1f}',")")
        
                idxstart=int((randt-self.__TGenerator.duration())*self.__listfe[0])
    
                for i in range(len(data)):
                    self.__Signal[0][idxstart+i]+=SNR*data[i]
    
                randt=npy.random.uniform(self.__TGenerator.duration(),self.__length)
                data=[]
                
            self.__Noise[0] += self.__Signal[0] # Hanford
            self.__Noise[1] += self.__Signal[0] # Livingston
            
            npts=float(len(self.__Noise[0]))
            norm=self.__length/npts
            plt.figure(figsize=(10,5))
            plt.xlabel('t (s)')
            plt.ylabel('h(t)')
            plt.grid(True, which="both", ls="-")
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Noise[0])
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Signal[0])
            plt.show()
            
    '''
    DATASET 1/
    
    Parser of the parameters file
    '''

    def _readParamFile(self,paramFile):

        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='Ttot' or lignes[1][0]!='fe' or lignes[2][0]!='kindPSD' or lignes[3][0]!='mint' or lignes[4][0]!='tcint' or lignes[5][0]!='NbB' or lignes[6][0]!='kindTemplate' or lignes[7][0]!='kindBank' or lignes[8][0]!='step' or lignes[9][0]!='whitening'or len(lignes)!=10:
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
        self.__tcmin=self.__Ttot+self.__listTtot[0]*(max(min(float(lignes[4][1]),float(lignes[4][2])),0.5)-1.)
        self.__tcmax=self.__Ttot+self.__listTtot[0]*(min(max(float(lignes[4][1]),float(lignes[4][2])),1.)-1.)
        self.__NbB=int(lignes[5][1])
        self.__kindTemplate=lignes[6][1]
        self.__step=float(lignes[8][1])
        self.__whiten=int(lignes[9][1])
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
            Mtmp=[]
            with open(os.path.dirname(__file__)+'/params/VT_bank_1band.txt') as mon_fichier:
                lines=mon_fichier.readlines()
                for line in lines:
                    if '#' in line:
                        continue
                    if '!' in line:
                        continue
                    data=line.strip()
                    pars=data.split(' ')
                    if (float(pars[1])+float(pars[2])<50.):
                        self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),0])
                        continue
                    Mtmp.append([float(pars[1]),float(pars[2])])
                    self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),1])
            M=npy.asarray(Mtmp)
            self.__GrilleMasses=M
            #self.__GrilleMasses=((M[(M.T[0]>=self.__mmin) & (M.T[1]>=self.__mmin) & (M.T[0]<=self.__mmax) & (M.T[1]<=self.__mmax)]).T[:2]).T
            self.__Ntemplate=len(self.__GrilleMasses)
            #print(self.__tmplist)
        
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
            if c%100==0:
                print("Producing sample ",c,"over",self.__Ntemplate*self.__NbB)
            self.__TGenerator.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1])
            norm=True
            if self.__whiten==0:
                norm=False
            temp=self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,
                                                             Tsample=self.__Ttot,
                                                             tc=npy.random.uniform(self.__tcmin,self.__tcmax),norm=norm)

            self.__listSNRevol=npy.append(self.__listSNRevol,self.__TGenerator._rawSnr)

            # Fill the corresponding data
            for j in range(self.__nTtot):
                #print(len(temp[j]))
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


    def _genNoiseSequence(self):

        nsamples=int(self.__length/self.__NGenerator.Ttot)
        self.__Noise=[]
        self.__Signal=[]
        chunck_V=[]
        chunck_H=[]
        for i in range(nsamples):
            self.__NGenerator.getNewSample()
            chunck_V.append(self.__NGenerator.getNoise())
            self.__NGenerator.getNewSample()
            chunck_H.append(self.__NGenerator.getNoise())

        self.__Noise.append(npy.ravel(npy.squeeze(chunck_V))) # Hanford
        self.__Noise.append(npy.ravel(npy.squeeze(chunck_H))) # Livingston
        self.__Signal.append(npy.zeros(len(self.__Noise[0])))
        self.__Signal.append(npy.zeros(len(self.__Noise[0])))
        chunck=[]
        for j in range(self.__nTtot): # Loop over samples
            
            # The following lines are for the SNR repartition
            self.__listSNRchunksAuto[j].append(npy.full((1),(1.0/self.__nTtot)))
            self.__listSNRchunksBalance[j].append(npy.full((1),(1.0/self.__nTtot)))
            if j==0:
                self.__listSNRchunksSpecial[j].append(npy.full((1),(1.0)))
            else:
                self.__listSNRchunksSpecial[j].append(npy.full((1),(0.0)))

        self.__listSNRchunksAuto=npy.reshape(self.__listSNRchunksAuto, (self.__nTtot, 1))
        self.__listSNRchunksSpecial=npy.reshape(self.__listSNRchunksSpecial, (self.__nTtot, 1))
        self.__listSNRchunksBalance=npy.reshape(self.__listSNRchunksBalance, (self.__nTtot, 1))
        self.__listSNRchunksAuto=npy.transpose(self.__listSNRchunksAuto)
        self.__listSNRchunksSpecial=npy.transpose(self.__listSNRchunksSpecial)
        self.__listSNRchunksBalance=npy.transpose(self.__listSNRchunksBalance)
             


    '''
    DATASET 5/
    
    Get a dataset from the noise and signal samples
    '''

    def getDataSet(self,SNRopt=1,weight='auto'):
        nbands=self.__nTtot
        dset=[]
        fdset = []
        finaldset=[]
        
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
        

        for i in range(self.Nsample):
            tempset=[]
            for j in range(nbands):
                sect=npy.asarray(dset[j][i])
                tempset.append(sect)
            sec=npy.concatenate(tempset)
            finaldset.append(sec)
        fdset=npy.asarray(finaldset)
                
        return fdset, list_weights

    def getFrame(self,weight='auto',det=0):
        nbands=self.__nTtot
        dset=[]
        fdset = []
        finaldset=[]
        
        if weight=='auto':
            list_weights=self.__listSNRchunksAuto
        if weight=='balance':
            list_weights=self.__listSNRchunksBalance
        if weight=='special':
            list_weights=self.__listSNRchunksSpecial
            
        print("Getting a frame which will be analyzed with",nbands,"frequency bands")

        #dset.append(self.__Sig[0]*(SNRopt)+self.__Noise[0])
        if (det==0):
            dset.append(self.__Noise[0])
        else:
            dset.append(self.__Noise[1])
        
        fdset=npy.asarray(dset[0])
                
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
            fname=dossier+self.__choice+'set_chunk-'+str(k)+'of'+str(self.__nTtot)+'_samples'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the object without the samples
        self.__Sig=[]
        self.__Noise=[]
        fichier=dossier+self.__choice+'-'+str(self.__nTtot)+'chunk'+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+self.__kindBank+'-'+str(self.__whiten)+'-'+str(self.__Ttot)+'s'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
                
    def saveFrame(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")

        # Save the sample in an efficient way
        fname=dossier+self.__choice+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+str(self.__length)+'s'+'-data'
        #npy.savez_compressed(fname,self.__Noise[0],self.__Sig[0])
        npy.savez_compressed(fname,self.__Noise[0],self.__Noise[1])
        self.__listfnames.append(fname)
        
        # Save the object without the samples (basically just the weights)
        self.__Signal=[]
        self.__Noise=[]
        fichier=dossier+self.__choice+'-'+str(self.__nTtot)+'chunk'+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+str(self.__whiten)+'-'+str(self.__length)+'s'+'.p'
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
    
    @classmethod
    def readFrame(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        
        print("Opening the frame")

        data=npy.load(str(obj.__listfnames[0])+'.npz')
        #obj.__Sig.append(data['arr_1'])
        obj.__Noise.append(data['arr_0'])
        obj.__Noise.append(data['arr_1'])
        data=[] # Release space
        f.close()
        return obj
    
    def getTruth(self):
        return self.__injections
    
    def getBkParams(self):
        return self.__tmplist
    
    def getTemplate(self,rank=0):
        return self.__Sig[0][rank]

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
    parser.add_argument("cmd", help="commande à choisir",choices=['noise', 'template', 'set', 'chunck', 'all'])
    parser.add_argument("--kindPSD","-kp",help="Type de PSD à choisir",choices=['flat', 'analytic','realistic'],default='analytic')
    parser.add_argument("--kindTemplate","-kt",help="Type de PSD à choisir",choices=['EM', 'EOB'],default='EM')
    parser.add_argument("--time","-t",help="Durée du signal",type=float,nargs='+',default=[1., 9.])
    parser.add_argument("-fe",help="Fréquence du signal",type=float,nargs='+',default=[2048, 512])
    parser.add_argument("-fmin",help="Fréquence minimale visible par le détecteur",type=float,default=20)
    parser.add_argument("-fmax",help="Fréquence maximale visible par le détecteur",type=float,default=1000)
    parser.add_argument("-white",help="Whitening type: none (0), frequency domain (1), time domain (2)",type=int,default=1)
    parser.add_argument("-snr",help="SNR (pour creation de sequence)",type=float,default=7.5)
    parser.add_argument("-m1",help="Masse du premier objet",type=float,default=20)
    parser.add_argument("-m2",help="Masse du deuxième objet",type=float,default=20)
    parser.add_argument("-n",help="Number of injections for frame gen",type=float,default=10)
    parser.add_argument("-length",help="Length of data chunck (in s)",type=float,default=300)
    parser.add_argument("--set","-s",help="Choix du type de set par défaut à générer",default=None)
    parser.add_argument("-step",help="Pas considéré pour la génération des paires de masses",type=float,default=0.1)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de set",default=None)
    parser.add_argument("--verbose","-v",help="verbose mode",type=bool,default=False)
    
    args = parser.parse_args()
    return args

'''
Main part of gendata.py
'''

def main():

    #import gen_noise as gn
    import gendata as gd
    
    args = parse_cmd_line()

    _brutePSD=[]

    if args.kindPSD=='realistic':

        f=open('data/PSDs_1369200000_50000_new.p', mode='rb')
        #f=open('PSDs_1369200000_50000.p', mode='rb')
        #f=open('PSDs_1369200000_100.p', mode='rb')
        storeL,storeH=pickle.load(f)
        f.close()

        print('Into realistic loop')

        for psds in storeL:
            _brutePSD.append(psds[1])                    

        print("Retrieved",len(_brutePSD),"different PSDs")


    if args.cmd=='noise': # Simple Noise generation
        NGenerator=gn.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax,whitening=args.white,verbose=args.verbose,customPSD=_brutePSD)
        NGenerator.getNewSample()
    
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()

        if args.verbose:
            plt.figure(figsize=(10,5))
            NGenerator.plotNoiseTW()
        
            plt.figure(figsize=(10,5))
            NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
            NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
            plt.figure(figsize=(10,5))
            NGenerator.plotTF2(fmin=args.fmin,fmax=args.fmax)
    
            for j in range(len(args.time)):
                plt.figure(figsize=(10,5))
                NGenerator.plotNoise1D(j)

            plt.figure(figsize=(10,5))
            NGenerator.plotinvPSD(fmin=args.fmin,fmax=args.fmax)

        plt.show()
        
    elif args.cmd=='template': # Simple Template generation
        TGenerator=gt.GenTemplate(Tsample=args.time,fe=args.fe,kindTemplate=args.kindTemplate,fDmin=args.fmin,fDmax=args.fmax,whitening=args.white,verbose=args.verbose,customPSD=_brutePSD)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=True)
        
    
        plt.figure(figsize=(10,5))
        TGenerator.plotTF()
            
        plt.figure(figsize=(10,5))
        TGenerator.plotTFn()
        
        print(f"TGenerator.duration() = {TGenerator.duration()}")
        plt.figure(figsize=(10,5))
        TGenerator.plot(Tsample=TGenerator.duration(),tc=TGenerator.duration(),SNR=args.snr)
        
        plt.figure(figsize=(10,5))
        TGenerator.plotSignal1D()
        
        plt.figure(figsize=(10,5))
        TGenerator.plotSNRevol()
        
        #plt.show()
        
    elif args.cmd=='all':
        TGenerator=gt.GenTemplate(Tsample=args.time,fe=args.fe,kindTemplate=args.kindTemplate,whitening=args.white,verbose=args.verbose,customPSD=_brutePSD)
        TGenerator.majParams(args.m1,args.m2)
        
        if (len(args.time)>1):
            randt=npy.random.uniform(TGenerator.duration(),sum(args.time))
        else :
            randt=npy.random.uniform(TGenerator.duration(),args.time)
            
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=TGenerator.duration(),tc=randt,norm=True)
        NGenerator=gn.GenNoise(Ttot=args.time,fe=args.fe,kindPSD=args.kindPSD,fmin=args.fmin,fmax=args.fmax,whitening=args.white,verbose=args.verbose,customPSD=_brutePSD)
        NGenerator.getNewSample()
                
        
        if args.verbose:
            plt.figure(figsize=(10,5))
            NGenerator.plotPSD(fmin=args.fmin,fmax=args.fmax)
            NGenerator.plotTF(fmin=args.fmin,fmax=args.fmax)
        
        plt.figure(figsize=(10,5))
        NGenerator.plotNoise()
        if (len(args.time)>1):
            TGenerator.plot(Tsample=TGenerator.duration(),tc=TGenerator.duration(),SNR=args.snr)
        else:
            TGenerator.plot(Tsample=args.time,tc=TGenerator.duration(),SNR=args.snr)

        plt.show()

    else: # Logically there is only set remaining, so dataset
        cheminout = './generators/'

        if args.set=='train':
            chemin='./params/default_trainGen_params.csv'
            set='train'
            Generator=gd.GenDataSet(paramFile=chemin,choice=set,customPSD=_brutePSD)
            Generator.saveGenerator(cheminout)
        elif args.set=='test':
            print('here')
            chemin='./params/default_testGen_params.csv'
            set='test'
            Generator=gd.GenDataSet(paramFile=chemin,choice=set,customPSD=_brutePSD)
            Generator.saveGenerator(cheminout)
        elif args.set=='frame':
            set='frame'
            Generator=gd.GenDataSet(paramFile=args.paramfile,choice=set,length=args.length,ninj=args.n,customPSD=_brutePSD)
            Generator.saveFrame(cheminout)
        else:
            Generator=gd.GenDataSet(paramFile=args.paramfile,choice=args.set,customPSD=_brutePSD)
            Generator.saveGenerator(cheminout)
    



############################################################################################################################################################################################
if __name__ == "__main__":
    main()
