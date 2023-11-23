from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as npy
from time import process_time
import os
import pickle
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mxnet.gluon.parameter")


npx.set_np() # permet de grantir un comportement de type Numpy en sortie des couches
             # Gluon

'''
Class defining the network

'''

class Multiple_CNN(nn.Block):

    '''
    Initialization: here we define the neural net
    
    Basically we set up one net for each frequency band, this net is the standard CNN as defined in the seminal
    paper from Huerta and George
    
    https://arxiv.org/abs/1711.03121
    
    This initialization requires the param file which was used for the training
    This is needed to retrieve the number of bands
    
    '''
    
    def __init__(self, paramsfile='./params/default_trainGen_params.csv', **kwargs):
        super().__init__(**kwargs) # héritage de de nn.Block
        
        self.list_chunks = []
        self.net = []
        
        with open(paramsfile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
        nValue=len(lignes[0])
        self.nb_inputs=nValue-1  # The number of frequency bands
        
        # The number of data points for each band is stored
        for x in range(1,nValue):
            self.list_chunks.append(float(float(lignes[0][x])*float(lignes[1][x])))
        
        # Define the networks (one per band)
        
        with self.name_scope():
            for i in range(self.nb_inputs):
                onenet = nn.Sequential()
                onenet.add(nn.BatchNorm(),
                        nn.Conv1D(channels=4, kernel_size=8),
                        nn.MaxPool1D(pool_size=3),
                        nn.Activation(activation='relu'),
                        nn.Conv1D(channels=8, kernel_size=16),
                        nn.MaxPool1D(pool_size=3),
                        nn.Activation(activation='relu'),
                        nn.Conv1D(channels=16, kernel_size=32),
                        nn.MaxPool1D(pool_size=3),
                        nn.Activation(activation='relu'),
                        nn.Flatten(),
                        nn.Dense(16, activation='relu'),
                        nn.Dense(2))
                
                self.net.append(onenet)
        
        # The last layer is a dense one which takes as input a weighted average of all the networks outputs
        # The weighting from each layer is defined later.
        
        self.output=nn.Dense(2)
                
        # Print the network
        for i in range(self.nb_inputs):
            setattr(self,f"net{i}",self.net[i])        
                
        # Record features, not used for the moment
        #self.feature_maps = [[] for i in range(len(getattr(self, "net0")))]


    '''
    Forward: here we define how the data is passing trough the NN
    
    Input are:
    
    X : the input data vectors (stored as a single vector with all the chunks)
    w : the weight to attribute to each chunk. Basically the SNR sharing. There are 3 options here:
    
            -> special : only take the first chunk (weight is 1 for it, and 0 for the rest), which is the one containing the coalescence
            -> balance : every chunks have an equal weight of 1/Nc
            -> auto    : the weight corresponds to the SNR proportion of the chunk (not really applicable for testing stage)
    
    '''
    
    def forward(self, X, w):
        list_outputs = []
        list_inputs = []
        cut_top = 0
        cut_bottom = 0
        
        # X contains the full data chunk
        # We first need to cut it back
        
        for i in range(self.nb_inputs):
            cut_top += int(self.list_chunks[i])
            list_inputs.append(X[:,:,cut_bottom:cut_top])
            cut_bottom = cut_top

        ''' # Retrieve the features, unnecessary here
        for i in range(len(getattr(self, "net0"))):
            out = list_inputs[0]
            for j in range(i):
                out = getattr(self, "net0")[j](out)
            out = getattr(self, "net0")[i](out)
            self.feature_maps[i] = out.asnumpy()
        '''
        
        # Then we pass data from chunk i in net i
        
        for i, input_data in enumerate(list_inputs):
        
            net = getattr(self, f"net{i}")
            output = net(input_data.as_in_ctx(d2l.try_gpu()))
            list_outputs.append(output)
        

        
        # And finally merge everything for the final dense layer

        combined=np.zeros((X.shape[0],2)).as_in_ctx(d2l.try_gpu())
        for i, outp in enumerate(list_outputs):
            combined = combined+(outp.T*w[:,i]).T
                
        return self.output(combined) # Return the output of the last dense layer
    
    def delete_feature_maps(self):
        self.feature_maps = [[] for i in range(len(getattr(self, "net0")))]
     
     
'''
Class defining the network training stage
'''

class MyTrainer:


    '''
    Initialization: parameters here are default values, they are overwritten by the config file
    '''
    
    def __init__(self,tabSNR=[8],tabEpochs=[10],lr=3e-3,batch_size=250,paramFile=None,
                net=None,loss=None,trainer=None,weight='special'):
        
        # 1. Use default info if there is no config file
        if paramFile is None:
            self.__batch_size=batch_size
            self.__tabSNR=tabSNR
            self.__tabEpochs=[0]+tabEpochs
            self.__lr=lr
            self.__weight=weight
            
            if len(self.__tabSNR)<=1:
                self.__kindTraining='Fix' # Training à SNR constant
            else:
                self.__kindTraining='Decr' # Training à SNR décroissant
                
            if isinstance(self.__tabSNR[0],tuple): # True si self.__tabSNR[0] est un tuple
                self.__kindTraining+='Int' # modifie __kindTraining
            else:
                self.__kindTraining+='Sca' # modifie __kindTraining
        elif os.path.isfile(paramFile): # Otherwise read the param file
            self._readParamFile(paramFile) # readParamFile est une fonction
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")
        
        # Check if there is a GPU available and use it
        self.__device=d2l.try_gpu() # try_gpu() renvoie le gpu ou cpu disponible
        print("This training phase will be done on the following processor:",self.__device)
        
        # 2. Retrieve the network and initialize it
        self.__net=Multiple_CNN() if net is None else net  # initilisation
        print(f"Architecture : {self.__net}")
        self.__net.initialize(force_reinit=True, ctx=self.__device, init=init.Xavier())
        print("Network initialized")
        # initialisation de type "Xavier" : variance des fonctions d'activation
        # sur toutes les couches
        
        # 3. The training parameters are defined next
        self.__loss= gluon.loss.SoftmaxCrossEntropyLoss() if loss is None else loss
        # initialisation de la fonction de perte
        self.__trainer=gluon.Trainer(self.__net.collect_params(), 'sgd', {'learning_rate': self.__lr}) if trainer is None else trainer
        # applique un optimiseur sur les paramètres à optimiser
        # collect_params() renvoie les paramètres enfants de nn.Block
        self.__trainGenerator=None
        self.__cTrainSet=None

    '''
    _readparamfile: parse the training param file
    
    SV note: this could be done more properly in the future
    '''
    
    def _readParamFile(self,paramFile):
        with open(paramFile) as mon_fichier:
            mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
            lignes = [x for x in mon_fichier_reader]
    # ouvre fichier csv et en sort les informations

        if lignes[0][0]!='batch_size' or lignes[1][0]!='lr' or lignes[2][0]!='kindTraining' or lignes[3][0]!='tabEpochs' or lignes[4][0]!='tabSNR' or lignes[5][0]!='weightingTrain' or lignes[6][0]!='weightingTest':
            raise Exception("Erreur dans le fichier de paramètres")
            # vérification du format du fichier csv (bandeau de gauche)

        self.__batch_size=int(lignes[0][1]) # taille des batchs
        self.__lr=float(lignes[1][1]) # learning rate
        self.__tabEpochs=[0] # tableau des époques
        for i in range(1,len(lignes[3])):
            if lignes[3][i]!='':
                self.__tabEpochs.append(int(lignes[3][i]))
        
        self.__kindTraining= ('Decr' if len(self.__tabEpochs)>2 else 'Fix' ) +lignes[2][1] # kindTraining est un string
        if self.__kindTraining=='DecrSca' or self.__kindTraining=='FixSca':
            self.__tabSNR=list(map(float,lignes[4][1:])) # map permet d'appliquer une
            # fonction à chaque élement de lignes[4] hormis le premier qui est coupé
        elif self.__kindTraining=='DecrInt' or self.__kindTraining=='FixInt' :
            self.__tabSNR=[]
            if len(lignes[4])%2==0:
                raise Exception("Pour l'entrainement avec intervalle il faut un nombre pair de valeurs dans tabEpochs")
            for i in range(1,len(lignes[4]),2):
                self.__tabSNR.append((float(lignes[4][i]),float(lignes[4][i+1])))
        else:
            raise Exception("Mauvais choix de type de Training dans le fichier de paramètre")
        self.__weight=(lignes[5][1])
    
    '''
    _clear: reset everything
    
    '''
    
    def _clear(self):
        self.__net.initialize(force_reinit=True, ctx=self.__device, init=init.Xavier())
        del self.__trainer
        self.__trainer=gluon.Trainer(self.__net.collect_params(), 'sgd', {'learning_rate': self.__lr})

        self.__trainGenerator=None
        self.__cTrainSet=None


    '''
    evaluate_accuracy_gpu:
    
    '''

    def evaluate_accuracy_gpu(self, net, data_iter, device=None):  #@save
        """Compute the accuracy for a model on a dataset using a GPU."""
        if not device:  # Query the first device where the first parameter is on
            device = list(net.collect_params().values())[0].list_ctx()[0]
            #print(device)
        # No. of correct predictions, no. of predictions
        metric = d2l.Accumulator(2)
        for X, y in data_iter:
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
        return metric[0] / metric[1] #ok
            
    def plot_kernel(self, net, epoch ,snr):
        for layer in net:
            if isinstance(layer, nn.Conv1D):
                convlayer = layer
                filters = layer.weight.data().asnumpy()
                # Plotting des filtres
                fig, axes = plt.subplots(ncols=1, nrows=filters.shape[0], figsize=(15, 15))
                fig.suptitle(f"Plots des filtres à l'époque {epoch+1} et aux SNR : {snr}")
                for i, ax in enumerate(axes.flatten()):
                    filt = filters[i, :, :]
                    ax.plot(filt.T, linewidth=0.5)
                    ax.set_title(f'{convlayer}')
                    ax.set_yticks([npy.min(npy.array(filt.T)), npy.max(npy.array(filt.T))])
                plt.tight_layout()
    
    def plot_features_maps(self, net, feature_maps, epoch, snr):
        if len(feature_maps) != len(net):
            raise ValueError("Problème de génération des features_maps")
        layer = [x for x in net]
        fig, axes = plt.subplots(ncols=1, nrows=len(feature_maps), figsize=(15, 15))
        fig.suptitle(f"Plots des features maps à l'époque {epoch+1} et aux SNR : {snr}")
        for i, ax in enumerate(axes.flatten()):
            if np.ndim(feature_maps[i]) == 3:
                toplot = feature_maps[i][:1, 0, :]
            else:
                toplot = feature_maps[i][:1, :]
            ax.plot(toplot.T, linewidth=0.5)
            ax.set_title(f'{layer[i]}')
            ax.set_yticks([npy.min(npy.array(toplot.T)), npy.max(npy.array(toplot.T))])
        plt.tight_layout()
        self.__net.delete_feature_maps()
                    
    def savePrints(self, dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        
        name = 'filters_plot' 
        c_dossier=dossier+str(name)+'-1'
        c=1
        while os.path.isdir(c_dossier):
            c+=1
            c_dossier=dossier+name+'-'+str(c)
        dossier=c_dossier
        os.mkdir(dossier)
        fichier=dossier+'/all_filters.pdf'
            
        pp = PdfPages(fichier)
        
        print(f"Figures trouvées : {plt.get_fignums()}")
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        
    def saveNet(self, dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        fichier=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-1.params'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'.params'
        self.__net.save_parameters(fichier)    
        #f=open(fichier, mode='wb')
        #pickle.dump(self,f)
        #f.close()
        
    '''
    train: this is the main macro, here, taking as input the training sample
    
    '''

    def train(self,TrainGenerator,results=None,verbose=True):
        self._clear()
        t_start=process_time() # We measure execution time
        
        # First we pick data in the training sample and adapt it to the required starting SNR
        
        self.__trainGenerator=TrainGenerator
        sample=self.__trainGenerator.getDataSet(self.__tabSNR[0],weight=self.__weight)
        # Training data at the initial SNR
        data=np.array(sample[0].reshape(self.__trainGenerator.Nsample,1,-1),dtype=np.float32)
        # Expected outputs
        labels=np.array(self.__trainGenerator.Labels,dtype=np.int32)
        # Sharing among frequency bands
        weight_sharing=np.array(sample[1],dtype=np.float32)

        
        self.__cTrainSet=(data,labels,weight_sharing)
        
        if verbose:
            print("Shape of training set",self.__cTrainSet[0].shape)
            print("Labels of training set",len(self.__cTrainSet[1]))
        
        # Put the init training properties in the results output file
        results.setMyTrainer(self)
        cTrain_iter=d2l.load_array(self.__cTrainSet,self.__batch_size)  
        results.Fill()

        
        if verbose:
            print("Epoch 0")
            print("AccuracyTraining,AccuracyTest=",results.accuracy(0))

        # Loop over all the requested SNR, each of them corresponds to a certain
        # number of epochs
        
        for i in range(len(self.__tabSNR)):
            if i>0: # We start a new SNR range, need to update the training set
                del self.__cTrainSet
                del cTrain_iter

                # Create a dataset with the corresponding SNR
                # Starting from the initial one at SNR=1
                sample=self.__trainGenerator.getDataSet(self.__tabSNR[i],weight=self.__weight)
                data=np.array(sample[0].reshape(self.__trainGenerator.Nsample,1,-1),dtype=np.float32)
                weight_sharing=np.array(sample[1],dtype=np.float32)
                self.__cTrainSet=(data,labels,weight_sharing)
                
                # cTrain_iter is an iterator over the training data
                # It cuts the data set into minibatch of size batch_size
                # GPU then deal with the batches in parallel
                # This iter thing is explained on this page:
                # https://classic.d2l.ai/chapter_linear-networks/linear-regression-scratch.html
                # see part 3.2.2
                cTrain_iter=d2l.load_array(self.__cTrainSet,self.__batch_size)
                 
            # Then run for the corresponding epochs at this SNR range/value
            for epoch in range(self.__tabEpochs[i],self.__tabEpochs[i+1]):
                # Here X is the input time serie, and y the exp output (0 for noise and 1 for signal)
                # self.__net(X) is the current network output

                compteur=0
                metric = d2l.Accumulator(3)
                metric2 = d2l.Accumulator(3)

                for X, y, w in cTrain_iter: # Exhaust all the batches (GPU should enhance this step)
                    compteur+=1
                    # The data is put on the GPU, if not already done
                    X = X.as_in_ctx(self.__device)
                    y = y.as_in_ctx(self.__device)
                    w = w.as_in_ctx(self.__device)

                    with autograd.record():
                        out = self.__net(X,w)
                        l = self.__loss(out, y)

                    l.backward() # Backward propagation of the derivatives
                    self.__trainer.step(self.__batch_size, ignore_stale_grad=True)
                    metric.add(l.sum(), d2l.accuracy(out, y), self.__batch_size)
                
                #if epoch%50==0:
                    #self.plot_features_maps(self.__net.net[0], self.__net.feature_maps, epoch, self.__tabSNR[i])
                    #self.plot_kernel(self.__net.net2, epoch, self.__tabSNR[i])

                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                print(f'epoch {epoch + 1},loss {train_l:.3f}, train acc {train_acc:.3f}')
                if epoch%5==0:
                    results.Fill()

        t_stop=process_time()
        
        if verbose:
            print("FIN! Temps d'entrainement en seconde pour ",self.__tabEpochs[-1]," époques :",t_stop-t_start)
            print('AccuracyTraining,AccuracyTest(snr:'+str(results.SNRtest)+')=',results.accuracy(-1))
            print('TrainSensitivity,TestSensitivity(snr:'+str(results.SNRtest)+')=',results.sensitivity(-1))
            print('TrainFAR,TestFAR(snr:'+str(results.SNRtest)+')=',results.FAR(-1))
        
        results.finishTraining()
        #results.specialFinishTraining()
    
    @property
    def trainGenerator(self):
        return self.__trainGenerator
    
    @property
    def cTrainSet(self):
        return self.__cTrainSet
    
    @property
    def loss(self):
        return self.__loss
    
    @property
    def net(self):
        return self.__net
        
    @property
    def kindTraining(self):
        return self.__kindTraining
        
    @property
    def batch_size(self):
        return self.__batch_size
        
    @property
    def lr(self):
        return self.__lr

    @property
    def weight(self):
        return self.__weight

    @property
    def tabSNR(self):
        return self.__tabSNR
    
'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("TrainGenerator", help="Fichier pickle contenant le générateur du DataSet d'entraînement")
    parser.add_argument("TestGenerator", help="Fichier pickle contenant le générateur du DataSet de test")
    parser.add_argument("--SNRtest","-St",help="SNR de test pour la courbe ROC",type=float,default=15)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres d'entraînement",default=None)
    parser.add_argument("--verbose","-v",help="Affiche l'évolution de l'entraînement",action="store_true")
    parser.add_argument("--number","-nb",help="Nombre d'entrainements",type=int,default=1)
    
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''

def main():
    import useResults as ur
    import gendata    as gd
    import trainCNN   as tr
    
    # Start by parsing the input options
    args = parse_cmd_line()
        
    # Then retrieve the input files (make use of gendata.py)
    TrainGenerator=gd.GenDataSet.readGenerator(args.TrainGenerator)
    TestGenerator=gd.GenDataSet.readGenerator(args.TestGenerator)
    
    # And the training option
    if args.paramfile is None:
        cheminparams='./params/default_trainer_params.csv'
    else:
        cheminparams=args.paramfile
        
    # Then define the mxnet based trainer
    mytrainer=tr.MyTrainer(paramFile=cheminparams)
    cheminresults='./results/'
    cheminsave='./prints/'
    
    # We loop over the number of trainings to be done
    for i in range(args.number):
        ## This is the basic training loop
        # First create an object to store the results
        myresults=ur.Results(TestGenerator,SNRtest=args.SNRtest)
        # Run the training loop
        mytrainer.train(TrainGenerator,myresults,verbose=args.verbose)
        # Store the results
        myresults.saveResults(cheminresults)
        mytrainer.saveNet(cheminresults)

############################################################################################################################################################################################
if __name__ == "__main__":
    main()
