from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
from time import process_time
import os
import csv


npx.set_np()

class MyCNN(nn.Block):
    """Réseau de neurone utilisé pour la détection composé de 3 couches de convolutions et de 2 couches denses"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.BatchNorm(),nn.Conv1D(channels=4, kernel_size=8), nn.MaxPool1D(pool_size=4),nn.Activation(activation='relu'),
        nn.Conv1D(channels=8, kernel_size=16), nn.MaxPool1D(pool_size=4),nn.Activation(activation='relu'),
        nn.Conv1D(channels=16, kernel_size=32), nn.MaxPool1D(pool_size=4),nn.Activation(activation='relu'),
        nn.Flatten(),nn.Dense(16, activation='relu'),nn.Dense(2))

    def forward(self, X):
        return self.net(X.as_in_ctx(d2l.try_gpu()))

'''
Class containing the training setup
'''

class MyTrainer:
    """Classe réalisant l'entrainement du réseau"""
    # Initialization
    def __init__(self,tabSNR=[8],tabEpochs=[10],lr=3e-3,batch_size=250,paramFile=None,net=None,loss=None,trainer=None):
        
        # 1. Retrieve the info from the training config file
        if paramFile is None:
            self.__batch_size=batch_size
            self.__tabSNR=tabSNR
            self.__tabEpochs=[0]+tabEpochs
            self.__lr=lr
            
            if len(self.__tabSNR)<=1:
                self.__kindTraining='Fix'
            else:
                self.__kindTraining='Decr'
                
            if isinstance(self.__tabSNR[0],tuple):
                self.__kindTraining+='Int'
            else:
                self.__kindTraining+='Sca'
        elif os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")
        
        self.__device=d2l.try_gpu()
        print("This training phase will be done on the following processor:",self.__device)
        
        # 2. Retrieve the network and initialize it
        self.__net=MyCNN() if net is None else net
        self.__net.initialize(force_reinit=True, ctx=self.__device, init=init.Xavier())
        
        # 3. The training parameters are defined next
        self.__loss= gluon.loss.SoftmaxCrossEntropyLoss() if loss is None else loss
        self.__trainer= gluon.Trainer(self.__net.collect_params(), 'sgd', {'learning_rate': self.__lr}) if trainer is None else trainer
        self.__trainGenerator=None
        self.__cTrainSet=None

            
    # Retrieve the training options
    def _readParamFile(self,paramFile):
        with open(paramFile) as mon_fichier:
            mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
            lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='batch_size' or lignes[1][0]!='lr' or lignes[2][0]!='kindTraining' or lignes[3][0]!='tabEpochs' or lignes[4][0]!='tabSNR' or len(lignes)!=5:
            raise Exception("Erreur dans le fichier de paramètres")

        self.__batch_size=int(lignes[0][1])
        self.__lr=float(lignes[1][1])
        self.__tabEpochs=[0]
        for i in range(1,len(lignes[3])):
            if lignes[3][i]!='':
                self.__tabEpochs.append(int(lignes[3][i]))
        
        self.__kindTraining= ('Decr' if len(self.__tabEpochs)>2 else 'Fix' ) +lignes[2][1]
        if self.__kindTraining=='DecrSca' or self.__kindTraining=='FixSca':
            self.__tabSNR=list(map(float,lignes[4][1:]))
        elif self.__kindTraining=='DecrInt' or self.__kindTraining=='FixInt' :
            self.__tabSNR=[]
            if len(lignes[4])%2==0:
                raise Exception("Pour l'entrainement avec intervalle il faut un nombre pair de valeurs dans tabEpochs")
            for i in range(1,len(lignes[4]),2):
                self.__tabSNR.append((float(lignes[4][i]),float(lignes[4][i+1])))
        else:
            raise Exception("Mauvais choix de type de Training dans le fichier de paramètre")
    
    def _clear(self):
        self.__net.initialize(force_reinit=True, ctx=self.__device, init=init.Xavier())
        del self.__trainer
        self.__trainer= gluon.Trainer(self.__net.collect_params(), 'sgd', {'learning_rate': self.__lr})
        self.__trainGenerator=None
        self.__cTrainSet=None

    def evaluate_accuracy_gpu(self,net, data_iter, device=None):  #@save
        """Compute the accuracy for a model on a dataset using a GPU."""
        if not device:  # Query the first device where the first parameter is on
            device = list(net.collect_params().values())[0].list_ctx()[0]
            print(device)
        # No. of correct predictions, no. of predictions
        metric = d2l.Accumulator(2)
        for X, y in data_iter:
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
        return metric[0] / metric[1]


    # The training
    def train(self,TrainGenerator,results,verbose=True):
        self._clear()
        t_start=process_time()
        
        # First we pick data in the training sample and adapt it to the required starting SNR
        self.__trainGenerator=TrainGenerator
        self.__cTrainSet=(np.array(self.__trainGenerator.getDataSet(self.__tabSNR[0]).reshape(self.__trainGenerator.Nsample,1,-1),dtype=np.float32),
        np.array(self.__trainGenerator.Labels,dtype=np.int32))
        
        # Put the init training properties in the results output file
        results.setMyTrainer(self)
        
        cTrain_iter=d2l.load_array(self.__cTrainSet,self.__batch_size)
        results.Fill()
            
        if verbose:
            print("Epoch 0")
            print("AccuracyTraining,AccuracyTest=",results.accuracy(0))
        
        for i in range(len(self.__tabSNR)):
            if i>0:
                del self.__cTrainSet
                del cTrain_iter
                # Create a dataset with the corresponding SNR
                # Starting from the initial one at SNR=1
                self.__cTrainSet=(np.array(self.__trainGenerator.getDataSet(self.__tabSNR[i]).reshape(self.__trainGenerator.Nsample,1,-1),dtype=np.float32),
                np.array(self.__trainGenerator.Labels,dtype=np.int32))
                
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
                for X, y in cTrain_iter: # Exhaust all the batches (GPU should enhance this step)
                    compteur+=1
                    # The data is put on the GPU, if not already done
                    X, y = X.as_in_ctx(self.__device), y.as_in_ctx(self.__device)
                    
                    with autograd.record():
                        y_hat = self.__net(X)
                        l = self.__loss(y_hat, y) # Get the losses for this batch
                    l.backward()                          # Backward propagation of the derivatives
                    self.__trainer.step(self.__batch_size)# Update the parameters
                    metric.add(l.sum(), d2l.accuracy(y_hat, y), self.__batch_size)
                    
                    #print(y)     # y[i] signal
                    #print(y_hat) # y_hat[i][0] / noise score || y_hat[i][1] / signal score
                    
                #print(self.evaluate_accuracy_gpu(self.__net, cTrain_iter))
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
                results.Fill()
                if verbose:
                    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
        t_stop=process_time()
        if verbose:
            print("FIN! Temps d'entrainement en seconde pour ",self.__tabEpochs[-1]," époques :",t_stop-t_start)
            print('AccuracyTraining,AccuracyTest(snr:'+str(results.SNRtest)+')=',results.accuracy(-1))
            print('TrainSensitivity,TestSensitivity(snr:'+str(results.SNRtest)+')=',results.sensitivity(-1))
            print('TrainFAR,TestFAR(snr:'+str(results.SNRtest)+')=',results.FAR(-1))
            
        results.finishTraining()
    
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
    def tabSNR(self):
        return self.__tabSNR
    
############################################################################################################################################################################################
def parse_cmd_line():
    import argparse
    """Parseur pour la commande trainCNN"""
    parser = argparse.ArgumentParser()
    parser.add_argument("TrainGenerator", help="Fichier pickle contenant le générateur du DataSet d'entraînement")
    parser.add_argument("TestGenerator", help="Fichier pickle contenant le générateur du DataSet de test")
    #parser.add_argument("--lr",help="Learnin-rate",type=float,default=3e-3)
    #parser.add_argument("--batch_size",help="Mini-batch size",type=int,default=250)
    parser.add_argument("--SNRtest","-St",help="SNR de test pour la courbe ROC",type=float,default=10)
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
    import gendata as gd
    import trainCNN as tr
    
    # Start by parsing the input options
    args = parse_cmd_line()
    
    # Then retrieve the input files
    TrainGenerator=gd.GenDataSet.readGenerator(args.TrainGenerator)
    TestGenerator=gd.GenDataSet.readGenerator(args.TestGenerator)
    
    # And the training option
    if args.paramfile is None:
        cheminparams=os.path.dirname(__file__)+'/params/default_trainer_params.csv'
    else:
        cheminparams=args.paramfile
        
    # Then define the mxnet based trainer
    mytrainer=tr.MyTrainer(paramFile=cheminparams)
    
    cheminresults='results/'
    
    # We loop over the number of trainings to be done
    
    for i in range(args.number):
    
        ## This is the basic training loop
        # First create an object to store the results
        myresults=ur.Results(TestGenerator,SNRtest=args.SNRtest)
        # Run the training loop
        mytrainer.train(TrainGenerator,myresults,verbose=args.verbose)
        # Store the results
        myresults.saveResults(cheminresults)

############################################################################################################################################################################################
if __name__ == "__main__":
    main()



