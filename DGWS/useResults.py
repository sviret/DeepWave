from mxnet import np, npx
from d2l import mxnet as d2l
import numpy as npy
import pickle
import os
import copy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['figure.max_open_warning'] = 1000

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
    diff=np.diff(-X,1)
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
        print('Attention aucune fausse détection autorisée for FAP',FAR)
        ind=1
    seuil=1-npy.sort(l)[ind-1]
    return seuil

'''
Class handling training results


'''


class Results:

    def __init__(self,TestGenerator,SNRtest=10):
        self.__SNRtest=SNRtest # à généraliser à plusieurs SNRtests
    
        self.__testGenerator=TestGenerator
        self.__NsampleTest=self.__testGenerator.Nsample
        
        self.__kindPSD=self.__testGenerator.kindPSD
        self.__mInt=self.__testGenerator.mInt
        self.__step=self.__testGenerator.mStep
                
        self.__listEpochs=[]
        self.__testOut=[]
        self.__trainOut=[]
        self.__listTrainLoss=[]
        self.__listTestLoss=[]
        self.__lastOuts=[]
        self.__snrOuts=[]
        
        self.__cTrainer=None
        self.__NsampleTrain=None
        
        self.__device=d2l.try_gpu()
        
        self._list_features_maps=[]
        
    def setMyTrainer(self,mytrainer):
        self.__cTrainer=mytrainer
        self.__NsampleTrain=int(self.__cTrainer.trainGenerator.Nsample)
        self.__tsize=self.__NsampleTrain
        self.__kindTraining=self.__cTrainer.kindTraining
        self.__batch_size=self.__cTrainer.batch_size
        self.__lr=self.__cTrainer.lr # Learning rate
        self.__weight=self.__cTrainer.weight #
        self.__minSNR=self.__cTrainer.tabSNR[-1] # Last SNR range
        self.__kindTemplate=self.__cTrainer.trainGenerator.kindTemplate
        self.__kindBank=self.__cTrainer.trainGenerator.kindBank
        
        sample=self.__testGenerator.getDataSet(self.__SNRtest,weight=self.__weight)
        data=np.array(sample[0].reshape(self.__NsampleTest,1,-1),dtype=np.float32)
        labels=np.array(self.__testGenerator.Labels,dtype=np.int32)
        weight_sharing=np.array(sample[1],dtype=np.float32)
        self.__testSet=(data,labels,weight_sharing)
                
        # Use the complete dataset here
        self.__Xtest=self.__testSet[0].as_in_ctx(self.__device)
        self.__ytest=self.__testSet[1].as_in_ctx(self.__device) # O for noise and 1 for signal
        self.__Wtest=self.__testSet[2].as_in_ctx(self.__device)

    # Fill the results obtained at the end of one epoch
    def Fill(self):
        # (0,1,2,....) until the last epoch
        self.__listEpochs.append(0) if len(self.__listEpochs)==0 else self.__listEpochs.append(self.__listEpochs[-1]+1)
        
        # Need to reload because SNR can change depending n the epoch
        self.__Xtrain=self.__cTrainer.cTrainSet[0].as_in_ctx(self.__device)
        self.__ytrain=self.__cTrainer.cTrainSet[1].as_in_ctx(self.__device)
        self.__Wtrain=self.__cTrainer.cTrainSet[2].as_in_ctx(self.__device)
        
        
        
        # Get the net output for train and test sample
        # We just pick a tenth of the training sample, to avoid too much computation
        
        outTrain=self.__cTrainer.net(self.__Xtrain[:self.__tsize],self.__Wtrain[:self.__tsize])
        outTest=self.__cTrainer.net(self.__Xtest,self.__Wtest)
        
        #print(outTrain.shape,outTest.shape)
        
        self.__testOut.append(usoftmax_f(outTest).asnumpy())
        self.__listTestLoss.append(self.__cTrainer.loss(outTest,self.__ytest).mean().asnumpy())
        
        self.__trainOut.append(usoftmax_f(outTrain).asnumpy())
        self.__listTrainLoss.append(self.__cTrainer.loss(outTrain,self.__ytrain[:self.__tsize]).mean().asnumpy())

        #print(self.__cTrainer.loss(outTrain,self.__ytrain[:self.__tsize]))
        #print(self.__testOut)
        #print(self.__listTestLoss)
        #print(self.__trainOut)
        #print(self.__listTrainLoss)

    # Here we do the calculations for the ROC curve
    # this is called at the end
    def finishTraining(self):
        for snr in range(0,25):
            rsnr=0.5*snr
            sample=self.__testGenerator.getDataSet(rsnr,weight=self.__weight)
            data=np.array(sample[0].reshape(self.__NsampleTest,1,-1),dtype=np.float32)
            labels=np.array(self.__testGenerator.Labels,dtype=np.int32)
            weight_sharing=np.array(sample[1],dtype=np.float32)
            TestSet=(data,labels,weight_sharing)
            
            self.__lastOuts.append(usoftmax_f(self.__cTrainer.net(TestSet[0].as_in_ctx(self.__device),TestSet[2].as_in_ctx(self.__device))).asnumpy())
            self.__snrOuts.append(rsnr)
            del TestSet 
     
    def specialFinishTraining(self):
        self.__cTrainer.net.delete_feature_maps
        listSnr = [x for x in range(0,110,10)]
        for i in range(11) :
            self._list_features_maps.append([])

            data=np.array(self.__testGenerator.specialGetDataSet(SNRopt=listSnr[i])[0].reshape(1, 1, -1),dtype=np.float32)
            labels=np.array(self.__testGenerator.specialLabels,dtype=np.int32)
            weight_sharing=np.array(self.__testGenerator.specialGetDataSet(self.__SNRtest,weight='balance')[1],dtype=np.float32)
            TestSet=(data,labels,weight_sharing)
            
            self.__cTrainer.net(TestSet[0].as_in_ctx(self.__device),TestSet[2].as_in_ctx(self.__device))
            self._list_features_maps[i] = cp.deepcopy(self.__cTrainer.net.feature_maps)
            self.__cTrainer.net.delete_feature_maps
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
    """   
    def plot_features_maps(self, net, feature_maps):
        if len(feature_maps[0])%len(net) != 0:
            raise ValueError("Problème de génération des features_maps")
        layer = [x for x in net]
        snr = [x for x in range(0,110,10)]
        for k in range(len(feature_maps)):
            fig, axes = plt.subplots(ncols=1, nrows=len(feature_maps[k]), figsize=(15, 15))
            fig.suptitle(f"Plots des features maps de test au SNR : {snr[k]}")
            for i, ax in enumerate(axes.flatten()):
                if np.ndim(feature_maps[k][i]) == 3:
                    toplot = feature_maps[k][i][:, 0, :]
                else:
                    toplot = feature_maps[k][i][:, :]
                ax.plot(toplot.T, linewidth=0.5)
                ax.set_title(f'{layer[i]}')
                ax.set_yticks([npy.min(npy.array(toplot.T)), npy.max(npy.array(toplot.T))])
            plt.tight_layout()
    """       
    def plot_features_maps(self, net, feature_maps):
        if len(feature_maps[0])%len(net) != 0:
            raise ValueError("Problème de génération des features_maps")
        layer = [x for x in net]
        snr = [x for x in range(0,110,10)]
        max_subplots_per_page = 4
        subplots_count = 0
        for k in range(len(feature_maps)):
            for i in range(len(feature_maps[k])):
                if np.ndim(feature_maps[k][i]) == 3 and npy.shape(feature_maps[k][i])[1] != 1:
                    num_subplots = npy.shape(feature_maps[k][i])[1]
                else:
                    num_subplots = 1
                for j in range(num_subplots):
                    if subplots_count % max_subplots_per_page == 0 and num_subplots != 1 :
                        fig1 = plt.figure()
                        fig1.suptitle(f"Plots des features maps de test au SNR : {snr[k]}")
                    if np.ndim(feature_maps[k][i]) == 3 and npy.shape(feature_maps[k][i])[1] != 1:
                        ax = plt.subplot(max_subplots_per_page, 1, (subplots_count % max_subplots_per_page) + 1)
                        toplot = feature_maps[k][i][:, j, :]
                        ax.plot(toplot.T, linewidth=0.5)
                        ax.set_title(f'{layer[i]} and filter n°{j+1}', fontsize=10, loc='left', wrap=True)
                        ax.set_yticks([npy.min(npy.array(toplot.T)), npy.max(npy.array(toplot.T))])
                        subplots_count += 1
                    else:
                        fig2 = plt.figure()
                        fig2.suptitle(f"Plots des features maps de test au SNR : {snr[k]}")
                        if np.ndim(feature_maps[k][i]) == 2:
                            toplot = feature_maps[k][i][:, :]
                        else: 
                            toplot = feature_maps[k][i][0, :, :]
                        plt.plot(toplot.T, linewidth=0.5)
                        plt.title(f'{layer[i]}', fontsize=10, loc='left', wrap=True)
                        plt.yticks([npy.min(npy.array(toplot.T)), npy.max(npy.array(toplot.T))])
                    plt.tight_layout()
            
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
            
    def plotMapDistrib(self,result,epoch,granularity=1):
        self.__nbMapDist+=1
        distsig=result.testOut[epoch][:result.NsampleTest//2].T[0]
        mlow=int(npy.floor(result.mInt[0]))
        mhigh=int(npy.ceil(result.mInt[1]))
        mstep=result.mStep
        #mstep=0.2
        Nbmasses=int((mhigh-mlow)/mstep)
        #print(len(distsig),Nbmasses,result.NsampleTest)
        X, Y = npy.meshgrid(npy.linspace(mlow, mhigh, Nbmasses+1), npy.linspace(mlow, mhigh, Nbmasses+1))
        Z=npy.zeros((Nbmasses,Nbmasses))
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
                print(epoch,l)
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
            
            #plt.text(0.34*results[0].listEpochs[-1], .55, 'TrainSensitivity='+str(npy.around(meanStrain[-1],3)))
            #plt.text(0.34*results[0].listEpochs[-1], .45, 'TestSensitivity(snr:'+str(results[0].SNRtest)+')='+str(npy.around(meanStest[-1],3)))
            
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
            #print('here')
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
                #print(seuil)
                Sensitivitylist.append(100*sensitivity(yhat,results.NsampleTest,seuil))
           
            plt.plot(SNRlist,Sensitivitylist,'.-',label='FAP='+str(FAR))
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
    parser.add_argument("-g",help="Choix de la granularité des plots de type Map",type=float,default=0.1)
    
    args = parser.parse_args()
    return args

#
# Main loop starts here
#
#

def main():
    import useResults as ur
    import trainCNN as tr
    
    args = parse_cmd_line()
    myNetwork = tr.Multiple_CNN()       
    studyNet = getattr(myNetwork,"net0")
    NbTraining=len(args.resultats)
    #récupération des fichiers résultats
    results=[]
    for i in range(NbTraining):
        results.append(ur.Results.readResults(args.resultats[i]))
        #features_maps = ur.Results.readResults(args.resultats[i])._list_features_maps
        
    #définition du Printer
    printer=ur.Printer()
    
    if NbTraining==1:
        
        printer.plotDistrib(results[0],results[0].listEpochs[0],FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*25//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*50//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1]*75//100,FAR=args.FAR)
        printer.plotDistrib(results[0],results[0].listEpochs[-1],FAR=args.FAR)
        
        printer.plotMapDistrib(results[0],results[0].listEpochs[0],granularity=args.g)
        printer.plotMapDistrib(results[0],results[0].listEpochs[-1]*25//100,granularity=args.g)
        printer.plotMapDistrib(results[0],results[0].listEpochs[-1]*50//100,granularity=args.g)
        printer.plotMapDistrib(results[0],results[0].listEpochs[-1]*75//100,granularity=args.g)
        printer.plotMapDistrib(results[0],results[0].listEpochs[-1],granularity=args.g)
        
    printer.plotROC(results,FAR=args.FAR)
    
    printer.plotSensitivity(results,FAR=args.FAR)
        
    printer.plotMultiSensitivity(results)
    
    #printer.plot_features_maps(studyNet, features_maps)
    
    # Sauvegarde des figures
    cheminsave='./prints/'
    printer.savePrints(cheminsave,args.nom_etude)
    
    if args.display:
        plt.show()
      
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
