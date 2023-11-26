from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import numpy as npy
import matplotlib.pyplot as plt
from time import process_time
import os
import pickle
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mxnet.gluon.parameter")


npx.set_np() # permet de grantir un comportement de type Numpy en sortie des couches
             # Gluon


def usoftmax_f(X):
    #print('here',X)
    diff=np.diff(-X,1)
    X2=np.concatenate((1./(1.+np.exp(diff)),1./(1.+np.exp(-diff))),axis=1)
    return X2

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
    
    def __init__(self, paramsfile=None, **kwargs):
        super().__init__(**kwargs) # héritage de de nn.Block
        
        self.list_chunks = []
        self.net = []
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
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framefile","-f", help="Fichier pickle contenant les données à étudier",default=None)
    parser.add_argument("--network","-n", help="Fichier pickle contenant le reseau entrainé",default=None)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de training",default=None)
    parser.add_argument("--verbose","-v",help="Affiche l'évolution de l'entraînement",action="store_true")
    
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''

def main():
    #import useResults as ur
    import gendata    as gd
    
    # Start by parsing the input options
    args = parse_cmd_line()
        
    # Then retrieve the input files (make use of gendata.py)
    #TestGenerator=gd.GenDataSet.readGenerator(args.TestGenerator)
    
    # Check if there is a GPU available and use it
    device=d2l.try_gpu() # try_gpu() renvoie le gpu ou cpu disponible
    print("The network will be run on the following processor:",device)
        
    # 2. Retrieve the network and initialize it
    net=Multiple_CNN(paramsfile=args.paramfile)
    net.load_parameters(args.network, ctx=device)
    
    # 3. Trained network is loaded, one can use it on data

    inputFrame=gd.GenDataSet.readFrame(args.framefile)
    
    sample=inputFrame.getFrame()
     
    npts=net.getNetSize()      # Number of points fed to the network for each block
    step=int(net.getStepSize())# Step between two blocks
    fs=float(net.getfs())      # Sampling freq
    
    #print(npts,step,fs)
        
    nTot=int(net.getNband())   # Number of bands
    listFe=net.getListFe()     # List of sampling freqs
    listTtot=net.getListTtot() # List of frame sizes
    tTot=net.getBlockLength()  # Total length of a block
    
    nptsHF=int(tTot*fs)        # Size of a block in the original frame

    #print(nTot,listFe,listTtot)
    
    nblocks=int(len(sample[0])/step) # Number of steps to analyze the frame

    weight_sharing=np.array(sample[1],dtype=np.float32)
    output=[]
    output_S=[]
    Frameref=[]

    
    for j in range(nTot):
        Frameref.append([])
    
    for i in range(nblocks): # Loop over frames

        tmpfrm=[]
        finalFrame=[]
        Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
        ref=Frameref[0]
                 
        #print("Starts the preparation of the signal chunk. Resampling over",nTot,"bands")
        
        #print(ref.shape)
        
        for j in range(nTot): # Resample the block
                            
            #Pick up the data chunk
            ndatapts=int(listTtot[j]*fs)
            nttt=len(ref)
            
            Nt=ref[-ndatapts:]
            ref=ref[:nttt-ndatapts]
            decimate=int(fs/listFe[j])
            
            Frameref[j]=Nt[::int(decimate)]
            tmpfrm.append(npy.asarray(Frameref[j]))
                        
        resampledFrame=npy.concatenate(tmpfrm)
        finalFrame.append(resampledFrame)
        fFrame=npy.asarray(finalFrame)
    
        data=np.array(fFrame.reshape(1,1,-1),dtype=np.float32)

        if data.shape[2]<npts: # Safety cut at the end
            break

        TestSet=(data,weight_sharing)
        
        res=usoftmax_f(net(TestSet[0].as_in_ctx(device),TestSet[1].as_in_ctx(device))).asnumpy()
        output.append(res.T[0])
        if (res.T[0]>0.999):
            output_S.append(res.T[0])
            print("Potential signal at t=",(i*step+nptsHF)*(1/fs),res.T[0])
        else:
            output_S.append(0)
        
    finalres=np.array(output)
    finalres_S=np.array(output_S)
    
    plot1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
    plot2 = plt.subplot2grid((3, 1), (2, 0))
    
    plot1.plot(npy.arange(len(sample[0]))/fs, sample[0],'.')
    plot2.plot(npy.arange(len(finalres))*step/fs, finalres,'.')
    plot2.plot(npy.arange(len(finalres))*step/fs, finalres_S,'.')
    plt.tight_layout()
    plt.show()
    

    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
