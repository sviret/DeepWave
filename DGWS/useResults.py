from mxnet import np, npx
from d2l import mxnet as d2l
import numpy as npy
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

##############################################################################################################################################################################################
npx.set_np()

parameters = {'font.size': 25,'axes.labelsize': 25,'axes.titlesize': 25,'figure.titlesize': 30,'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25,'legend.title_fontsize': 25,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [6.4*3.5,4.8*3.5]}
plt.rcParams.update(parameters)

##############################################################################################################################################################################################


def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    #print(X_exp)
    #print(partition)
    #print(X_exp / partition)
    return X_exp / partition 

# Slow version (loop over elements)
def usoftmax(X):
    X2=X
    for elem in X:
        #print(elem[0],elem[1])
        diff=elem[0]-elem[1]
        soft=[1./(1.+np.exp(-diff)),1./(1.+np.exp(diff))]
        #print (soft)
        elem[0]=soft[0]
        elem[1]=soft[1]
    return X2

# Faaaast version
def usoftmax_f(X):
    diff=np.diff(X,1)
    X2=np.concatenate((1./(1.+np.exp(diff)),1./(1.+np.exp(-diff))),axis=1)
    return X2


def accuracy(yhat,N,seuil=0.5):
    return (sensitivity(yhat,N,seuil)+1-FAR(yhat,N,seuil))/2

def sensitivity(yhat,N,seuil=0.5):
    #superieur au seuil
    return ((yhat[:N//2].T[0].astype(npy.float32)>=seuil)*npy.ones(N//2)).mean()

def FAR(yhat,N,seuil=0.5):
    return 1-((yhat[-N//2:].T[0].astype(npy.float32)<seuil)*npy.ones(N//2)).mean()

def Threshold(yhat,N,FAR=0.005):
    l=yhat[-N//2:].T[1]
    ind=int(npy.floor(FAR*(N//2)))
    if ind==0:
        print('Attention aucune fausse détection autorisée')
        ind=1
    seuil=1-npy.sort(l)[ind-1]
    return seuil

class Results:
    """Classe stockant les résultats de l'entrainement"""
    def __init__(self,TestGenerator,SNRtest=10):
        self.__SNRtest=SNRtest # à généraliser à plusieurs SNRtests
        
        self.__testGenerator=TestGenerator
        self.__NsampleTest=self.__testGenerator.Nsample
        self.__testSet=(np.array(self.__testGenerator.getDataSet(self.__SNRtest).reshape(self.__NsampleTest,1,-1),dtype=np.float32),
        np.array(self.__testGenerator.Labels,dtype=np.int32))
        
        self.__kindPSD=self.__testGenerator.kindPSD
        self.__mInt=self.__testGenerator.mInt
            
        self.__listEpochs=[]
        self.__testOut=[]
        self.__trainOut=[]
        self.__listTrainLoss=[]
        self.__listTestLoss=[]
        self.__lastOuts=[]
            
        self.__cTrainer=None
        self.__NsampleTrain=None
        
        self.__device=d2l.try_gpu()
        
    def setMyTrainer(self,mytrainer):
        self.__cTrainer=mytrainer
        self.__NsampleTrain=self.__cTrainer.trainGenerator.Nsample
        self.__kindTraining=self.__cTrainer.kindTraining
        self.__batch_size=self.__cTrainer.batch_size
        self.__lr=self.__cTrainer.lr
        self.__minSNR=self.__cTrainer.tabSNR[-1]
        self.__kindTemplate=self.__cTrainer.trainGenerator.kindTemplate
        self.__kindBank=self.__cTrainer.trainGenerator.kindBank

    def Fill(self):
        self.__listEpochs.append(0) if len(self.__listEpochs)==0 else self.__listEpochs.append(self.__listEpochs[-1]+1)
        
        self.__testOut.append(usoftmax_f(self.__cTrainer.net(self.__testSet[0].as_in_ctx(self.__device))).asnumpy())
        self.__listTestLoss.append(self.__cTrainer.loss(self.__cTrainer.net(self.__testSet[0].as_in_ctx(self.__device)), self.__testSet[1].as_in_ctx(self.__device)).mean().asnumpy())
        
        self.__trainOut.append(usoftmax_f(self.__cTrainer.net(self.__cTrainer.cTrainSet[0].as_in_ctx(self.__device))).asnumpy())
        self.__listTrainLoss.append(self.__cTrainer.loss(self.__cTrainer.net(self.__cTrainer.cTrainSet[0].as_in_ctx(self.__device)), self.__cTrainer.cTrainSet[1].as_in_ctx(self.__device)).mean().asnumpy())

    def finishTraining(self):
        for snr in range(4,20):
            TestSet=(np.array(self.__testGenerator.getDataSet(SNRopt=snr/2).reshape(self.__NsampleTest,1,-1),dtype=np.float32),np.array(self.__testGenerator.Labels,dtype=np.int32))
            self.__lastOuts.append(usoftmax_f(self.__cTrainer.net(TestSet[0].as_in_ctx(self.__device))).asnumpy())
            del TestSet
            
        del self.__cTrainer
        self.__cTrainer=None
            
    def accuracy(self,epoch,seuil=0.5):
        return accuracy(self.__trainOut[epoch],self.__NsampleTrain,seuil), accuracy(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def sensitivity(self,epoch,seuil=0.5):
        return sensitivity(self.__trainOut[epoch],self.__NsampleTrain,seuil), sensitivity(self.__testOut[epoch],self.__NsampleTest,seuil)
      
    def FAR(self,epoch,seuil=0.5):
        return FAR(self.__trainOut[epoch],self.__NsampleTrain,seuil), FAR(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def Threshold(self,epoch,FAR=0.005):
        return Threshold(self.__trainOut[epoch],self.__NsampleTrain,FAR), Threshold(self.__testOut[epoch],self.__NsampleTest,FAR)
            
    def saveResults(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        fichier=dossier+self.__kindTraining+'-'+self.__kindTemplate+'-'+self.__kindBank+'-'+self.__kindPSD+'-'+str(self.__lr)+'-'+str(self.__minSNR)+'-1.p'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier=dossier+self.__kindTraining+'-'+self.__kindTemplate+'-'+self.__kindBank+'-'+self.__kindPSD+'-'+str(self.__lr)+'-'+str(self.__minSNR)+'-'+str(c)+'.p'
            
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
      
    @classmethod
    def readResults(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        f.close()
        return obj
      
    @property
    def SNRtest(self):
        return self.__SNRtest
    @property
    def testOut(self):
        return self.__testOut
    @property
    def lastOuts(self):
        return self.__lastOuts
    @property
    def NsampleTest(self):
        return self.__NsampleTest
    @property
    def mInt(self):
        return self.__mInt
    @property
    def listEpochs(self):
        return self.__listEpochs
    @property
    def TrainLoss(self):
        return self.__listTrainLoss
    @property
    def TestLoss(self):
        return       self.__listTestLoss
    @property
    def lr(self):
        return self.__lr
    @property
    def kindTraining(self):
        return self.__kindTraining
    @property
    def kindPSD(self):
        return self.__kindPSD
    @property
    def minSNR(self):
        return self.__minSNR
    @property
    def kindTemplate(self):
        return       self.__kindTemplate
    @property
    def kindBank(self):
        return self.__kindBank

class Printer:
    """Classe permettant d'afficher les résultats"""
    def __init__(self):
        self.__nbDist=0
        self.__nbMapDist=0
        self.__nbROC=0
        self.__nbSens=0
            
    def plotDistrib(self,result,epoch,FAR=0.005):
        self.__nbDist+=1
        distsig=result.testOut[epoch][:result.NsampleTest//2].T[0]
        distnoise=result.testOut[epoch][-result.NsampleTest//2:].T[0]
            
        seuil=result.Threshold(epoch,FAR)[1]
        plt.figure('Distribution_epoch'+str(epoch)+'-'+str(self.__nbDist))
        plt.axvline(x=seuil,color='r',label='FAR='+str(FAR),linestyle='--')
        plt.hist(distnoise,bins=npy.arange(101)/100,label='noise distrib')
        plt.hist(distsig,bins=npy.arange(101)/100,label='sig distrib')
        plt.text(seuil-0.1,100,'seuil='+str(npy.around(seuil,3)))
            
        plt.xlabel('Probability')
        plt.ylabel('Number')
        plt.yscale('log')
        plt.title(label='Sortie du reseau sur le testSet à SNR='+ str(result.SNRtest)+ ' à l\'epoque '+str(epoch))
        #plt.title(label='Noise realisation:'+str(NbparT)+'  learning rate:'+str(lr)+'  SNRopt:'+str(Snr)+'\nEpochs='+str(num_epochs))
        plt.legend()
            
    def plotMapDistrib(self,result,epoch):
        self.__nbMapDist+=1
        distsig=result.testOut[epoch][:result.NsampleTest//2].T[0]
        mlow=int(npy.floor(result.mInt[0]))
        mhigh=int(npy.ceil(result.mInt[1]))
        Nbmasses=mhigh-mlow
            
        X, Y = npy.meshgrid(npy.linspace(mlow, mhigh, Nbmasses+1), npy.linspace(mlow, mhigh, Nbmasses+1))
        Z=npy.zeros((Nbmasses,Nbmasses))
        c=0
        for i in range(Nbmasses):
            for j in range(i+1):
                Z[i][j]=distsig[c]
                c+=1
            
        plt.figure('MapDistribution_epoch'+str(epoch)+'-'+str(self.__nbMapDist))
        plt.pcolormesh(X,Y,Z.T)
        plt.xlabel('m1')
        plt.ylabel('m2')

        plt.colorbar()
        plt.title(label='Output of softmax regression for signal sample in the plan (m1,m2)')
        #plt.legend()
            
    def plotROC(self,results,FAR=0.005):
        self.__nbROC+=1
        plt.figure('ROC-'+str(self.__nbROC))
            
        if isinstance(results,list):
            taille=results[0].listEpochs[-1]
            for result in results:
                if taille!=result.listEpochs[-1]:
                    raise Exception("Tous les entrainements n'ont pas la même longueur")
        else:
            results=[results]
        
        label=self._findlabel(results)
        
        if label==None:
            meanTrainLoss=[]
            meanTestLoss=[]
            stdTrainLoss=[]
            stdTestLoss=[]
            meanStrain=[]
            stdStrain=[]
            meanStest=[]
            stdStest=[]
            
            for epoch in results[0].listEpochs:
                l=[result.TrainLoss[epoch] for result in results]
                meanTrainLoss.append(npy.mean(l))
                stdTrainLoss.append(npy.std(l))
                
                l=[result.TestLoss[epoch] for result in results]
                #print(l)
                meanTestLoss.append(npy.mean(l))
                stdTestLoss.append(npy.std(l))
                
                lseuil=[result.Threshold(epoch,FAR) for result in results]
                
                l=[result.sensitivity(epoch,seuil[0])[0] for result,seuil in zip(results,lseuil)]
                meanStrain.append(npy.mean(l))
                stdStrain.append(npy.std(l))
                
                l=[result.sensitivity(epoch,seuil[1])[1] for result,seuil in zip(results,lseuil)]
                meanStest.append(npy.mean(l))
                stdStest.append(npy.std(l))
                  
            plt.errorbar(results[0].listEpochs,meanTrainLoss,stdTrainLoss,label='TrainLoss')
            plt.errorbar(results[0].listEpochs,meanTestLoss,stdTestLoss,linestyle='--',label='TestLoss')
            plt.errorbar(results[0].listEpochs,meanStrain,stdStrain,label='Training Sensitivity')
            plt.errorbar(results[0].listEpochs,meanStest,stdStest,linestyle='--',label='Test Sensitivity')
            
            plt.text(0.34*results[0].listEpochs[-1], .55, 'TrainSensitivity='+str(npy.around(meanStrain[-1],3)))
            plt.text(0.34*results[0].listEpochs[-1], .45, 'TestSensitivity(snr:'+str(results[0].SNRtest)+')='+str(npy.around(meanStest[-1],3)))
            
            plt.xlabel('Epochs')
            plt.ylabel('No Unit')
            plt.title(label='Evolution du training sur le testSet à SNR='+ str(result.SNRtest)+ ' FAR(train)= '+str(FAR))
            plt.legend()
        else:
            self.__nbROC-=1
            for s_results in self._souslists(results,label):
                self.plotROC(s_results,FAR=FAR)
    
    def _findlabel(self,results):
        
        if isinstance(results,list)==False:
            raise TypeError("results doit être une liste de résultats")
            
        label=[]
        c_minSNR=results[0].minSNR
        c_lr=results[0].lr
        c_kindTraining=results[0].kindTraining
        c_kindPSD=results[0].kindPSD
        c_kindTemplate=results[0].kindTemplate
        c_kindBank=results[0].kindBank
      
        for result in results:
            if c_minSNR!=result.minSNR:
                label.append('SNRtrain')
                break
        for result in results:
            if c_lr!=result.lr:
                label.append('lr')
                break
        for result in results:
            if c_kindTraining!=result.kindTraining:
                label.append('kindTraining')
                break
        for result in results:
            if c_kindPSD!=result.kindPSD:
                label.append('kindPSD')
                break
        for result in results:
            if c_kindTemplate!=result.kindTemplate:
                label.append('kindTemplate')
                break
        for result in results:
            if c_kindBank!=result.kindBank:
                label.append('kindBank')
                break
      
        if len(label)==0:
            return None
        elif len(label)==1:
            return label[0]
        else:
            raise Exception("Plusieurs hyperparamètres sont différents dans les jeux de résultats: "+str(label))

    def _souslists(self,results,label):
        dic={}
        label='minSNR' if label=='SNRtrain' else label
        for result in results:
            if dic.__contains__(result.__getattribute__(label)):
                dic[result.__getattribute__(label)].append(result)
            else:
                dic[result.__getattribute__(label)]=[result]
        return dic.values()

    def plotSensitivity(self,results,FAR=0.005):
        self.__nbSens+=1
        plt.figure('Sensitivity_Vs_SNRtest-'+str(self.__nbSens))
        SNRlist=[i for i in range(4,20)]
        
        if isinstance(results,list):
            if len(results)==1:
                results=results[0]
            
        if isinstance(results,list):
            i=0
            label=self._findlabel(results)

            if label==None:
                meanSensitivity=[]
                stdSensitivity=[]
                        
                for i in range(len(results[0].lastOuts)):
                    lseuil=[Threshold(result.lastOuts[i],result.NsampleTest,FAR) for result in results]
                              
                    l=[sensitivity(result.lastOuts[i],result.NsampleTest,seuil)*100 for result,seuil in zip(results,lseuil)]
                    meanSensitivity.append(npy.mean(l))
                    stdSensitivity.append(npy.std(l))
                        
                plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label='Mutiple Training')
                        
            else:
                for s_results in self._souslists(results,label):
                    meanSensitivity=[]
                    stdSensitivity=[]

                    for i in range(len(results[0].lastOuts)):
                        lseuil=[Threshold(result.lastOuts[i],result.NsampleTest,FAR) for result in s_results]
                        l=[sensitivity(result.lastOuts[i],result.NsampleTest,seuil)*100 for result,seuil in zip(s_results,lseuil)]
                        meanSensitivity.append(npy.mean(l))
                        stdSensitivity.append(npy.std(l))

                    if label=='SNRtrain':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].minSNR)
                    elif label=='lr':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].lr)
                    elif label=='kindTraining':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].kindTraining)
                    elif label=='kindPSD':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].kindPSD)
                    elif label=='kindTemplate':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].kindTemplate)
                    elif label=='kindBank':
                        plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label=s_results[0].kindBank)
                    else:
                        raise AttributeError("Les attributs possibles pour les labels sont: SNRtrain, lr, kindTraining, kindPSD")
        else:
            print('here')
            Sensitivitylist=[]
            for yhat in results.lastOuts:
                seuil=Threshold(yhat,results.NsampleTest,FAR)
                print(seuil)
                Sensitivitylist.append(100*sensitivity(yhat,results.NsampleTest,seuil))
            realSNRlist = [float(x) / 2. for x in SNRlist]
            plt.plot(realSNRlist,Sensitivitylist,'.-',label='Sensitivity')
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
        plt.title(label='Sensitivity Vs SNRopt de Test pour un FAR='+str(FAR))
        plt.legend()
        
    def plotMultiSensitivity(self,results):
        self.__nbSens+=1
        plt.figure('Sensitivity_Vs_SNRtest-'+str(self.__nbSens))
        SNRlist=[i for i in range(4,20)]
        
        if isinstance(results,list):
            if len(results)==1:
                results=results[0]

        for i in range(4):
            Sensitivitylist=[]
            
            for yhat in results.lastOuts:
                FAR=10**(-float(i+1))
                seuil=Threshold(yhat,results.NsampleTest,FAR)
                print(seuil)
                Sensitivitylist.append(100*sensitivity(yhat,results.NsampleTest,seuil))
            realSNRlist = [float(x) / 2. for x in SNRlist]
            plt.plot(realSNRlist,Sensitivitylist,'.-',label='FAP='+str(FAR))
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
        #plt.title(label='Sensitivity Vs SNRopt de Test pour un FAR='+str(FAR))
        plt.legend()
        
    def savePrints(self,dossier,name):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
            
        c_dossier=dossier+name+'-1'
        c=1
        while os.path.isdir(c_dossier):
            c+=1
            c_dossier=dossier+name+'-'+str(c)
        dossier=c_dossier
        os.mkdir(dossier)
        fichier=dossier+'/all_results.pdf'
            
        pp = PdfPages(fichier)
            
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
            fig.savefig(dossier+'/'+fig.get_label()+'.png',format='png')
        pp.close()
            
def parse_cmd_line():
    import argparse
    """Parseur pour la commande trainCNN"""
    parser = argparse.ArgumentParser()
    parser.add_argument("nom_etude", help="Nom de l'anlayse sous lequel stocker les figures résultats")
    parser.add_argument("resultats", nargs='+', help="Fichier(s) pickle(s) contenant les résultats d'entraînement")
    parser.add_argument("-FAR",help="Choix du taux de fausse alarame",type=float,default=0.005)
    parser.add_argument("--display","-d",help="Affiche les résulats dans des fenêtres",action="store_true")
    
    args = parser.parse_args()
    return args


def main():
    import useResults as ur
    args = parse_cmd_line()
    
    NbTraining=len(args.resultats)
    #récupération des fichiers résultats
    results=[]
    for i in range(NbTraining):
        results.append(ur.Results.readResults(args.resultats[i]))
    
    
    #définition du Printer
    printer=ur.Printer()
    
    if NbTraining==1:
        printer.plotDistrib(results[0],results[0].listEpochs[0],FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*25//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*50//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*75//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1],FAR=args.FAR)
        printer.plotMapDistrib(results[0],results[0].listEpochs[-1])
    
    printer.plotROC(results,FAR=args.FAR)
    
    printer.plotSensitivity(results,FAR=args.FAR)
        
    printer.plotMultiSensitivity(results)
    
    ## Sauvegarde des figures
    cheminsave='prints/'
    printer.savePrints(cheminsave,args.nom_etude)
    
    if args.display:
        plt.show()
      
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
    

