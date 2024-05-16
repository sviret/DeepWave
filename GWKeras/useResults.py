import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import pickle
import os
import copy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 1000

##############################################################################################################################################################################################
#npx.set_np()

#parameters = {'font.size': 25,'axes.labelsize': 25,'axes.titlesize': 25,'figure.titlesize': 30,'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25,'legend.title_fontsize': 25,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [6.4*3.5,4.8*3.5]}
#plt.rcParams.update(parameters)
##############################################################################################################################################################################################

# Slow version (loop over elements)
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def usoftmax_f(X):
  diff=tnp.diff(X,1)
  X2=tnp.concatenate((1./(1.+tnp.exp(diff)),1./(1.+tnp.exp(-diff))),axis=1)
  return X2

def accuracy(yhat,N,seuil=0.5):
    return (sensitivity(yhat,N,seuil)+1-FAR(yhat,N,seuil))/2

def sensitivity(yhat,N,seuil=0.5):
    #superieur au seuil
    return ((yhat[:N//2].T[1].astype(np.float32)>=seuil)*np.ones(N//2)).mean()

def FAR(yhat,N,seuil=0.5):
    return 1-((yhat[-N//2:].T[0].astype(np.float32)<seuil)*np.ones(N//2)).mean()

def Threshold(yhat,N,FAR=0.005):
    l=np.sort(yhat[-N//2:].T[1]) # Proba d'être signal assignée au bruit
    ind=len(l)-int(np.floor(FAR*(N//2)))
    if ind==0:
        print('Sample is too small to define a threshold with FAP',FAR)
        ind=1
    seuil=l[ind-1]
    return seuil

'''
Class handling training results


'''


class Results:

    def __init__(self,SNRtest=10):
        self.__SNRtest=SNRtest # à généraliser à plusieurs SNRtests
        
        self.__listEpochs=[]
        self.__testOut=[]
        #self.__trainOut=[]
        self.__listTrainLoss=[]
        self.__listTestLoss=[]
        self.__listTrainAcc=[]
        self.__listTestAcc=[]
        self.__lastOuts=[]
        self.__snrOuts=[]
        
        self.__cTrainer=None
        
    def setMyTrainer(self,mytrainer):
        self.__cTrainer=mytrainer
        self.__NsampleTrain=int(self.__cTrainer.trainGenerator.Nsample) # Limit size of the result file
        self.__NsampleTest=int(self.__cTrainer.testGenerator.Nsample)
        self.__tsize=self.__NsampleTrain
        self.__kindTraining=self.__cTrainer.kindTraining
        self.__batch_size=self.__cTrainer.batch_size
        self.__lr=self.__cTrainer.lr # Learning rate
        self.__weight=self.__cTrainer.weight #
        self.__minSNR=self.__cTrainer.tabSNR[-1] # Last SNR range
        self.__kindTemplate=self.__cTrainer.trainGenerator.kindTemplate
        self.__kindBank=self.__cTrainer.trainGenerator.kindBank
        self.__kindPSD=self.__cTrainer.trainGenerator.kindPSD
        self.__mInt=self.__cTrainer.testGenerator.mInt
        self.__step=self.__cTrainer.testGenerator.mStep
            
        # Use the complete dataset here
        self.__Xtest=self.__cTrainer.cTestSet[0]
        self.__ytest=self.__cTrainer.cTestSet[1] # O for noise and 1 for signal
        self.__Wtest=self.__cTrainer.cTestSet[2]

    # Fill the results obtained at the end of one epoch
    def Fill(self):
        # (0,1,2,....) until the last epoch
        #self.__listEpochs.append(0) if len(self.__listEpochs)==0 else self.__listEpochs.append(self.__listEpochs[-1]+1)
        
        # Get the net output for validation sample only
        print(self.__Xtest[0].shape)
        #outTest=self.__cTrainer.net(self.__Xtest,self.__Wtest)
        outTest = self.__cTrainer.net.model.predict(self.__Xtest,verbose=0)
                
        #self.__testOut.append(outTest)
        self.__testOut.append(usoftmax_f(outTest))
    
    # Here we do the calculations for the ROC curve
    # this is called at the end
    def finishTraining(self):
    
        self.__listTrainLoss=self.__cTrainer.final_loss
        self.__listTestLoss=self.__cTrainer.final_loss_t
        self.__listTrainAcc=self.__cTrainer.final_acc
        self.__listTestAcc=self.__cTrainer.final_acc_t
        
        self.__listEpochs=np.arange(len(self.__listTrainLoss))
    
        for snr in range(0,20):
            rsnr=0.5*snr
            sample=self.__cTrainer.testGenerator.getDataSet(rsnr,weight=self.__weight)
            data=np.array(sample[0].reshape(self.__NsampleTest,-1,1),dtype=np.float32)
            labels=np.array(self.__cTrainer.testGenerator.Labels,dtype=np.int32)
            weight_sharing=np.array(sample[1],dtype=np.float32)
            TestSet=(data,labels,weight_sharing)
            
            cut_top = 0
            cut_bottom = 0
            list_inputs_val=[]
                    
            for i in range(self.__cTrainer.net.nb_inputs):
                cut_top += int(self.__cTrainer.net.list_chunks[i])
                list_inputs_val.append(data[:,cut_bottom:cut_top,:])
                cut_bottom = cut_top
            
            outTest = self.__cTrainer.net.model.predict(list_inputs_val,verbose=0)
            #self.__lastOuts.append(outTest)
            self.__lastOuts.append(usoftmax_f(outTest))
            self.__snrOuts.append(rsnr)
        del TestSet,list_inputs_val,data,sample,weight_sharing,labels
        self.__Xtest=None
        self.__ytest=None
        self.__Wtest=None
        self.__cTrainer=None

     
       
    def accuracy(self,epoch,seuil=0.5):
        return accuracy(self.__testOut[epoch],self.__NsampleTrain,seuil), accuracy(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def sensitivity(self,epoch,seuil=0.5):
        return sensitivity(self.__testOut[epoch],self.__NsampleTrain,seuil), sensitivity(self.__testOut[epoch],self.__NsampleTest,seuil)
      
    def FAR(self,epoch,seuil=0.5):
        return FAR(self.__testOut[epoch],self.__NsampleTrain,seuil), FAR(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def Threshold(self,epoch,FAR=0.005):
        return Threshold(self.__testOut[epoch],self.__NsampleTrain,FAR), Threshold(self.__testOut[epoch],self.__NsampleTest,FAR)
            
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
    def snrOuts(self):
        return self.__snrOuts
    @property
    def NsampleTest(self):
        return self.__NsampleTest
    @property
    def mInt(self):
        return self.__mInt
    @property
    def mStep(self):
        return self.__step
    @property
    def listEpochs(self):
        return self.__listEpochs
    @property
    def TrainLoss(self):
        return self.__listTrainLoss
    @property
    def TestLoss(self):
        return self.__listTestLoss
    @property
    def TrainAcc(self):
        return self.__listTrainAcc
    @property
    def TestAcc(self):
        return self.__listTestAcc
        
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
        return self.__kindTemplate
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
        distsig=result.testOut[epoch][:result.NsampleTest//2].T[1]
        distnoise=result.testOut[epoch][-result.NsampleTest//2:].T[1]
            
        seuil=result.Threshold(epoch,FAR)[1] # Threshold for FAP on test sample
        #print(epoch,seuil)
        plt.figure('Distribution_epoch'+str(epoch)+'-'+str(self.__nbDist))
        plt.axvline(x=seuil,color='r',label='FAR='+str(FAR),linestyle='--')
        plt.hist(distnoise,bins=np.arange(101)/100,label='noise distrib')
        plt.hist(distsig,bins=np.arange(101)/100,label='sig distrib')
        plt.text(seuil-0.1,100,'seuil='+str(np.around(seuil,3)))
            
        plt.xlabel('Probability')
        plt.ylabel('Number')
        plt.yscale('log')
        plt.title(label='Sortie du reseau sur le testSet à SNR='+ str(result.SNRtest)+ ' à l\'epoque '+str(epoch))
        plt.legend()
            
    def plotMapDistrib(self,result,epoch,granularity=1):
        self.__nbMapDist+=1
        distsig=result.testOut[epoch][:result.NsampleTest//2].T[1]
        mlow=int(np.floor(result.mInt[0]))
        mhigh=int(np.ceil(result.mInt[1]))
        mstep=result.mStep
        #mstep=0.2
        Nbmasses=int((mhigh-mlow)/mstep)
        #print(len(distsig),Nbmasses,result.NsampleTest)
        X, Y = np.meshgrid(np.linspace(mlow, mhigh, Nbmasses+1), np.linspace(mlow, mhigh, Nbmasses+1))
        Z=np.zeros((Nbmasses,Nbmasses))
        c=0
        for i in range(Nbmasses):
            if c==len(distsig):
                break
            for j in range(i+1):
                #print(i,j,c)
                Z[i][j]=distsig[c]
                c+=1
                if c==len(distsig):
                    break
        
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
                meanTrainLoss.append(np.mean(l))
                stdTrainLoss.append(np.std(l))
                
                l=[result.TestLoss[epoch] for result in results]
                meanTestLoss.append(np.mean(l))
                stdTestLoss.append(np.std(l))
                
                l=[result.TrainAcc[epoch] for result in results]
                meanStrain.append(np.mean(l))
                stdStrain.append(np.std(l))
                
                l=[result.TestAcc[epoch] for result in results]
                meanStest.append(np.mean(l))
                stdStest.append(np.std(l))
            
            plt.errorbar(results[0].listEpochs,meanTrainLoss,stdTrainLoss,label='TrainLoss')
            plt.errorbar(results[0].listEpochs,meanTestLoss,stdTestLoss,linestyle='--',label='TestLoss')
            plt.errorbar(results[0].listEpochs,meanStrain,stdStrain,label='Training Sensitivity')
            plt.errorbar(results[0].listEpochs,meanStest,stdStest,linestyle='--',label='Test Sensitivity')
            
            plt.xlabel('Epochs')
            plt.ylabel('No Unit')
            plt.title(label='Evolution du training (TestSet SNR='+ str(result.SNRtest)+')')
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
        
        
        if isinstance(results,list):
            if len(results)==1:
                results=results[0]
        
        SNRlist=results.snrOuts
        
        if isinstance(results,list):
            i=0
            label=self._findlabel(results)

            if label==None:
                meanSensitivity=[]
                stdSensitivity=[]
                        
                for i in range(len(results[0].lastOuts)):
                    lseuil=[Threshold(result.lastOuts[i],result.NsampleTest,FAR) for result in results]
                              
                    l=[sensitivity(result.lastOuts[i],result.NsampleTest,seuil)*100 for result,seuil in zip(results,lseuil)]
                    meanSensitivity.append(np.mean(l))
                    stdSensitivity.append(np.std(l))
                        
                plt.errorbar(SNRlist,meanSensitivity,stdSensitivity,label='Mutiple Training')
                        
            else:
                for s_results in self._souslists(results,label):
                    meanSensitivity=[]
                    stdSensitivity=[]

                    for i in range(len(results[0].lastOuts)):
                        lseuil=[Threshold(result.lastOuts[i],result.NsampleTest,FAR) for result in s_results]
                        l=[sensitivity(result.lastOuts[i],result.NsampleTest,seuil)*100 for result,seuil in zip(s_results,lseuil)]
                        meanSensitivity.append(np.mean(l))
                        stdSensitivity.append(np.std(l))

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

            Sensitivitylist=[]
            for yhat in results.lastOuts:
                seuil=Threshold(yhat,results.NsampleTest,FAR)
                #print(seuil)
                Sensitivitylist.append(100*sensitivity(yhat,results.NsampleTest,seuil))

            plt.plot(SNRlist,Sensitivitylist,'.-',label='Sensitivity')
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
        plt.title(label='Sensitivity Vs SNRopt de Test pour un FAR='+str(FAR))
        plt.legend()
        
    def plotMultiSensitivity(self,results):
        self.__nbSens+=1
        plt.figure('Sensitivity_Vs_SNRtest-'+str(self.__nbSens))
        
        
        if isinstance(results,list):
            if len(results)==1:
                results=results[0]
        
        SNRlist=results.snrOuts
        
        for i in range(4):
            Sensitivitylist=[]
            
            for yhat in results.lastOuts:
                FAR=10**(-float(i+1))
                seuil=Threshold(yhat,results.NsampleTest,FAR)
                Sensitivitylist.append(100*sensitivity(yhat,results.NsampleTest,seuil))
           
            plt.plot(SNRlist,Sensitivitylist,'.-',label='FAP='+str(FAR))
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
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
    parser.add_argument("-g",help="Choix de la granularité des plots de type Map",type=float,default=0.1)
    
    args = parser.parse_args()
    return args

#
# Main loop starts here
#
#

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
    
    parameters = {'font.size': 25,'axes.labelsize': 25,'axes.titlesize': 25,'figure.titlesize': 30,'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25,'legend.title_fontsize': 25,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [6.4*3.5,4.8*3.5]}
    #mpl.rcParams.update(parameters)
    mpl.rcParams['lines.linewidth'] = 3
    plt.rcParams['figure.figsize'] = (8.4,5.8)

    
    nrecorded=len(results[0].testOut)
        
    if NbTraining==1:
        for i in range(nrecorded):
            printer.plotDistrib(results[0],i,FAR=args.FAR)
            printer.plotMapDistrib(results[0],i,granularity=args.g)
    
    printer.plotROC(results,FAR=args.FAR)
    printer.plotSensitivity(results,FAR=args.FAR)
    printer.plotMultiSensitivity(results)
    
    
    # Sauvegarde des figures
    cheminsave='./prints/'
    printer.savePrints(cheminsave,args.nom_etude)
    
    if args.display:
        plt.show()
      
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
