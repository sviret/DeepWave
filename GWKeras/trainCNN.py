import tensorflow as tf
from keras import backend as K
from tensorflow.keras import datasets, layers, models, initializers, optimizers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from time import process_time
import os
import pickle
import csv
import warnings


'''
Class defining the network

'''

class Multiple_CNN():

    '''
    Initialization: here we define the neural net
    
    Basically we set up one net for each frequency band, this net is the standard CNN as defined in the seminal
    paper from Huerta and George
    
    https://arxiv.org/abs/1711.03121
    
    This initialization requires the param file which was used for the training
    This is needed to retrieve the number of bands
    
    '''
    
    def __init__(self, paramsfile='./params/default_trainGen_params.csv',lr=3e-3,wht='special',**kwargs):

        self.list_chunks = []
        self.net = []
        weight = []
        initializer = initializers.GlorotNormal() # Xavier initialization
        opt = optimizers.Adam(learning_rate=lr)
        self.__npts = 0

        with open(paramsfile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
        nValue=len(lignes[0])
        self.nb_inputs=nValue-1  # The number of frequency bands

        self.__fs = int(lignes[1][1])

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

        # The number of data points for each band is stored
        for x in range(1,nValue):
            self.list_chunks.append(float(float(lignes[0][x])*float(lignes[1][x])))
            self.__npts+=int(float(lignes[0][x])*float(lignes[1][x]))
            
        
        self.singleBand=False
        if self.nb_inputs==1:
            self.singleBand=True
            weight.append(1.)
        else:
            for i in range(self.nb_inputs):
                if wht=='balance':
                    weight.append(1./self.nb_inputs)
                else:
                    weight.append(0.)
        weight=np.asarray(weight)
        if wht=='special':
            weight[0]=1.
            
        # Define the networks (one per band)
        self.inputs=[]
        self.outputs=[]

        with tf.name_scope('simpleCNN') as scope:
            for i in range(self.nb_inputs):
                input=layers.Input(shape=(int(self.list_chunks[i]),1))
                x=layers.BatchNormalization()(input)
                '''
                x=layers.Conv1D(filters=4, kernel_size=8, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=8, kernel_initializer=initializer)(x)
                '''
                # Huerta & Georges legacy
                x=layers.Conv1D(filters=16, kernel_size=16, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=32, kernel_size=8, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=64, kernel_size=8, kernel_initializer=initializer)(x)
                '''
                x=layers.Conv1D(filters=16, kernel_size=8, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=8, kernel_initializer=initializer)(x)
                '''
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Flatten()(x)
                #x=layers.Dense(16, activation='relu', kernel_initializer=initializer)(x)
                #x=layers.Dense(8, activation='relu', kernel_initializer=initializer)(x)
                output=layers.Dense(2, kernel_initializer=initializer)(x)
                
                self.inputs.append(input)

                # Don't use the weighting for the moment
                #self.outputs.append(weight[i]*output)
                self.outputs.append(output)

        # The last layer is a dense one which takes as input a weighted average of all the networks outputs
        # The weighting from each layer is defined later.

        # Take note that there is no activation in the last dense layer
        # softmax activation is included in the loss function (via from_logits option)
        # enable to use the special activation function defined in https://arxiv.org/abs/2106.03741
        #

        x = layers.add(self.outputs)
        if not self.singleBand: # merge the bands if > 1
            self.out = layers.Dense(2, kernel_initializer=initializer)(x)
        else:
            self.out = self.outputs
        self.model = models.Model(self.inputs, self.out)
        
        # Init the model
        self.model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
                
        # Print the network
        print(self.model.summary())

    def getNetSize(self):
        return self.__npts
        
    def getBlockLength(self):
        return self.__Ttot
    
    def getStepSize(self):
        return self.list_chunks[0]/2
        
    def getfs(self):
        return self.__fs
        
            
    def getNband(self):
        return self.__nTtot
        
    def getListFe(self):
        return self.__listfe
 
    def getListTtot(self):
        return self.__listTtot
     
     
'''
Class defining the network training stage
'''

class MyTrainer():


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
         
        # 2. Retrieve the network and initialize it
        self.__net=Multiple_CNN(lr=self.__lr,wht=self.weight,paramsfile=self.__ptfile) if net is None else net  # initilisation
        print("Network initialized")
        
        # 3. The training parameters are defined next
       
        self.__trainGenerator=None
        self.__cTrainSet=None
        self.__testGenerator=None
        self.__cTestSet=None
    
        self.__loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #self.__loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    '''
    _readparamfile: parse the training param file
    
    SV note: this could be done more properly in the future
    '''
    
    def _readParamFile(self,paramFile):
        with open(paramFile) as mon_fichier:
            mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
            lignes = [x for x in mon_fichier_reader]
    # ouvre fichier csv et en sort les informations

        if lignes[0][0]!='batch_size' or lignes[1][0]!='lr' or lignes[2][0]!='kindTraining' or lignes[3][0]!='tabEpochs' or lignes[4][0]!='tabSNR' or lignes[5][0]!='weightingTrain' or lignes[6][0]!='weightingTest' or lignes[7][0]!='trainingParams':
            raise Exception("Erreur dans le fichier de paramètres")
            # vérification du format du fichier csv (bandeau de gauche)

        self.__ptfile=lignes[7][1] # taille des batchs
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

        self.__trainGenerator=None
        self.__cTrainSet=None
        self.__testGenerator=None
        self.__cTestSet=None

    # Here we save the network info

    def saveNet(self, dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        fichier=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-1.h5'
        fichier_js=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-1_fullnet.p'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'.h5'
            fichier_js=dossier+self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'_fullnet.p'
        self.__net.model.save(fichier)
        f=open(fichier_js, mode='wb')
        pickle.dump(self.__net,f)
        f.close()
        self.__net=None
        
    '''
    train: this is the main macro, here, taking as input the training sample
    
    '''

    def train(self,TrainGenerator,TestGenerator,SNRtest=10,results=None,verbose=True):
        self._clear()
        t_start=process_time() # We measure execution time
        
        # First we pick data in the training sample and adapt it to the required starting SNR
        self.__trainGenerator=TrainGenerator
        sample=self.__trainGenerator.getDataSet(self.__tabSNR[0],weight=self.__weight,size=0)
        # Training data at the initial SNR
        data=np.array(sample[0].reshape(self.__trainGenerator.Nsample,-1,1),dtype=np.float32)
        # Expected outputs
        labels=np.array(self.__trainGenerator.Labels,dtype=np.int32)
        # Sharing among frequency bands
        weight_sharing=np.array(sample[1],dtype=np.float32)

        # The test dataset will always be the same, pick it up once
        self.__testGenerator=TestGenerator
        sample_t=self.__testGenerator.getDataSet(SNRtest,weight=self.__weight,size=0)
        data_t=np.array(sample_t[0].reshape(self.__testGenerator.Nsample,-1,1),dtype=np.float32)
        labels_t=np.array(self.__testGenerator.Labels,dtype=np.int32)
        weight_sharing_t=np.array(sample_t[1],dtype=np.float32)
        
        if verbose:
            print("Shape of training set",data.shape)
            print("Shape of validation set",data_t.shape)
                
        cut_top = 0
        cut_bottom = 0
        list_inputs_val=[]
        list_inputs=[]
        
        for i in range(self.__net.nb_inputs):
            cut_top += int(self.__net.list_chunks[i])
            list_inputs_val.append(data_t[:,cut_bottom:cut_top,:])
            list_inputs.append(data[:,cut_bottom:cut_top,:])
            cut_bottom = cut_top
                              
        self.__cTrainSet=(list_inputs,labels,weight_sharing)
        self.__cTestSet=(list_inputs_val,labels_t,weight_sharing_t)

        # Put the init training properties in the results output file
        results.setMyTrainer(self)
        results.Fill()
        epoch=0
        
        # Loop over all the requested SNR, each of them corresponds to a certain
        # number of epochs
        
        accuracy=[]
        loss=[]
        accuracy_t=[]
        loss_t=[]
        
        
        for i in range(len(self.__tabSNR)):
            if i>0: # We start a new SNR range, need to update the training set
                del self.__cTrainSet

                # Create a dataset with the corresponding SNR
                # Starting from the initial one at SNR=1
                sample=self.__trainGenerator.getDataSet(self.__tabSNR[i],weight=self.__weight,size=0)
                data=np.array(sample[0].reshape(self.__trainGenerator.Nsample,-1,1),dtype=np.float32)
                weight_sharing=np.array(sample[1],dtype=np.float32)
                            
                cut_top = 0
                cut_bottom = 0
                list_inputs=[]
                    
                for j in range(self.__net.nb_inputs):
                    cut_top += int(self.__net.list_chunks[j])
                    list_inputs.append(data[:,cut_bottom:cut_top,:])
                    cut_bottom = cut_top
                
                self.__cTrainSet=(list_inputs,labels,weight_sharing)
                
                 
            # Then run for the corresponding epochs at this SNR range/value
            nepochs=self.__tabEpochs[i+1]-self.__tabEpochs[i]

            print("Training between epochs",self.__tabEpochs[i],"and",self.__tabEpochs[i+1])
            
            # Run the training over the epochs
            history=self.__net.model.fit(self.__cTrainSet[0],labels,batch_size=self.__batch_size,epochs=nepochs, validation_data=(list_inputs_val, labels_t))
            
            acc=np.asarray(history.history['accuracy'])
            los=np.asarray(history.history['loss'])
            acc_t=np.asarray(history.history['val_accuracy'])
            los_t=np.asarray(history.history['val_loss'])
            
            for i in range(nepochs):
                accuracy.append(acc[i])
                loss.append(los[i])
                accuracy_t.append(acc_t[i])
                loss_t.append(los_t[i])

            train_acc=np.asarray(history.history['accuracy']).mean()
            test_acc=np.asarray(history.history['val_accuracy']).mean()
            train_l=np.asarray(history.history['loss']).mean()
            test_l=np.asarray(history.history['val_loss']).mean()

            epoch+=nepochs
            if verbose:
                print(f'Train perf with this SNR range: train loss {train_l:.3f}, train acc {train_acc:.3f}')
                print(f'Validation perf at this stage: test loss {test_l:.3f}, test acc {test_acc:.3f}')

            results.Fill()

        t_stop=process_time()

        self.__final_acc=np.asarray(accuracy).flatten()
        self.__final_acc_t=np.asarray(accuracy_t).flatten()
        self.__final_loss=np.asarray(loss).flatten()
        self.__final_loss_t=np.asarray(loss_t).flatten()

        results.finishTraining()
    
    @property
    def trainGenerator(self):
        return self.__trainGenerator

    @property
    def testGenerator(self):
        return self.__testGenerator

    @property
    def cTrainSet(self):
        return self.__cTrainSet

    @property
    def cTestSet(self):
        return self.__cTestSet

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

    @property
    def final_acc(self):
        return self.__final_acc

    @property
    def final_acc_t(self):
        return self.__final_acc_t
        
    @property
    def final_loss(self):
        return self.__final_loss
        
    @property
    def final_loss_t(self):
        return self.__final_loss_t
        

'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("TrainGenerator", help="Fichier pickle contenant le générateur du DataSet d'entraînement")
    parser.add_argument("TestGenerator", help="Fichier pickle contenant le générateur du DataSet de test")
    parser.add_argument("--SNRtest","-St",help="SNR de test pour la courbe ROC",type=float,default=7.5)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres d'entraînement",default=None)
    parser.add_argument("--verbose","-v",help="Affiche l'évolution de l'entraînement",action="store_true")
    parser.add_argument("--number","-nb",help="Nombre d'entrainements",type=int,default=1)
    parser.add_argument("--paramtrain","-pt",help="Fichier csv des paramètres du training sample",default=None)
    
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
        myresults=ur.Results(SNRtest=args.SNRtest)
        # Run the training loop
        mytrainer.train(TrainGenerator,TestGenerator,SNRtest=args.SNRtest,results=myresults,verbose=args.verbose)
        # Store the results
        mytrainer.saveNet(cheminresults)
        myresults.saveResults(cheminresults)


############################################################################################################################################################################################
if __name__ == "__main__":
    main()
