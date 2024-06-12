import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers, optimizers
import numpy as npy
import matplotlib.pyplot as plt
from time import process_time
import os
import sys
import pickle
import csv

import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def usoftmax_f(X):
  diff=tnp.diff(X,1)
  X2=tnp.concatenate((1./(1.+tnp.exp(diff)),1./(1.+tnp.exp(-diff))),axis=1)
  return X2

'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framefile","-f", help="Fichier pickle contenant les données à étudier",default=None)
    parser.add_argument("--network","-n", help="Fichier h5 contenant le reseau entrainé",default=None)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres du training sample",default=None)
    parser.add_argument("--verbose","-v",help="Affiche l'évolution de l'entraînement",action="store_true")
    
    args = parser.parse_args()
    return args

def split_sequence(array, n_steps):
    ## split a univariate sequence into samples
    # Dimension of input array: [nsample][npts][1]
    # Dimension of output array: [nsample][npts][n_steps]

    splitted=[]
    #X, y = list(), list()
    for data in array:
        #print(data[0:10])
        # Zero padding
        seq = npy.concatenate((npy.zeros(int(n_steps/2)), data.reshape(-1), npy.zeros(int(n_steps/2))))
        # Splitting
        ssequence = npy.array([npy.array(seq[i:i+n_steps]) for i in range(len(seq)-n_steps)])
        #print(ssequence[0:10])
        splitted.append(ssequence)

    final=npy.asarray(splitted)
    #print(final.shape)
    return npy.expand_dims(final,axis=-1)

'''
The main training macro starts here
'''

def main():
    #import trainEncoder   as tr
    import gendata    as gd
    
    # 1. Start by parsing the input options
    args = parse_cmd_line()
                
    # 2. Then retrieve the network and initialize it
    f=open(args.network,mode='rb')
    net=pickle.load(f)
    f.close()
    print(net.model.summary())

    npts=net.getNetSize()      # Number of points fed to the network for each block
    step=int(net.getStepSize())# Step between two blocks
    fs=float(net.getfs())      # Sampling freq
    nTot=int(net.getNband())   # Number of bands
    listFe=net.getListFe()     # List of sampling freqs
    listTtot=net.getListTtot() # List of frame sizes
    tTot=net.getBlockLength()  # Total length of a block
    nptsHF=int(tTot*fs)        # Size of a block in the original frame
    feature=net.getFeatureSize() 
    step=int(1.*fs) 


    # 3. Trained network is loaded, one can use it on data
    inputFrame=gd.GenDataSet.readFrame(args.framefile)
    
    sample=inputFrame.getFrame()

    #print(len(sample[0]),len(sample[2]))
    nblocks=int(len(sample[0])/step) # Number of steps to analyze the frame
    weight_sharing=npy.array(sample[1],dtype=npy.float32)
    output=[]
    output_S=[]
    Frameref=[]
    Truthref=[]

    for j in range(nTot):
        Frameref.append([])
        Truthref.append([])

    # 4. Loop over frames to perform the inference

    for i in range(nblocks):

        tmpfrm=[]
        finalFrame=[]
        finalTruth=[]
        tmptru=[]
        Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
        Truthref[0]=sample[2][i*step:i*step+nptsHF] # The truth (if available)
        ref=Frameref[0]
        truth=Truthref[0]

        #print(ref.shape,truth.shape)

        for j in range(nTot): # Resample the block
                            
            #Pick up the data chunk
            ndatapts=int(listTtot[j]*fs)
            nttt=len(ref)
            decimate=int(fs/listFe[j])
                       
            Nt=ref[-ndatapts:]
            ref=ref[:nttt-ndatapts]
            Frameref[j]=Nt[::int(decimate)]
            tmpfrm.append(npy.asarray(Frameref[j]))

            Nt=truth[-ndatapts:]
            truth=truth[:nttt-ndatapts]
            Truthref[j]=Nt[::int(decimate)]
            tmptru.append(npy.asarray(Truthref[j]))

        resampledFrame=npy.concatenate(tmpfrm)
        finalFrame.append(resampledFrame)
        fFrame=npy.asarray(finalFrame)

        resampledTruth=npy.concatenate(tmptru)
        finalTruth.append(resampledTruth)
        fTruth=npy.asarray(finalTruth)
        
        data=npy.array(fFrame.reshape(1,-1,1),dtype=npy.float32)
        truth=npy.array(fTruth.reshape(1,-1,1),dtype=npy.float32)

        if data.shape[1]<npts: # Safety cut at the end
            break
        
        #print(data.shape)
        data=split_sequence(data, feature)
        #print(data.shape)

        TestSet=(data,truth)
                
        cut_top = 0
        cut_bottom = 0
        list_inputs_val=[]
                    
        for k in range(nTot):
            cut_top += int(listTtot[k]*fs)
            list_inputs_val.append(data[:,cut_bottom:cut_top,:])
            cut_bottom = cut_top
            #print(cut_bottom,cut_top,)
        
        res = net.model.predict(list_inputs_val,verbose=0)
        res=npy.squeeze(res)
        #print(res.shape,truth.shape)

        output.append(res)
        output_S.append(truth)
        

    finalres=npy.array(output).flatten()
    finalres_S=npy.array(output_S).flatten()
    print(len(finalres),len(finalres_S))
    plot1 = plt.subplot2grid((4, 1), (0, 0), rowspan = 2)
    plot2 = plt.subplot2grid((4, 1), (2, 0))
    plot3 = plt.subplot2grid((4, 1), (3, 0))
    
    plot1.plot(npy.arange(len(sample[0]))/fs, sample[0],'.')
    plot2.plot(npy.arange(len(finalres))/fs, finalres,'.')
    plot3.plot(npy.arange(len(finalres))/fs, finalres_S,'.')
    #plot3.plot(npy.arange(len(finalres))/fs, finalres,'.')
    plt.tight_layout()
    plt.show()
    

    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
