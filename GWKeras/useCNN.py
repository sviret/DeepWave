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

'''
The main training macro starts here
'''

def main():
    import trainCNN   as tr
    import gendata    as gd
    
    # 1. Start by parsing the input options
    args = parse_cmd_line()
                
    # 2. Then retrieve the network and initialize it
    net=tr.Multiple_CNN(paramsfile=args.paramfile)
    net.model.load_weights(args.network)

    npts=net.getNetSize()      # Number of points fed to the network for each block
    step=int(net.getStepSize())# Step between two blocks
    fs=float(net.getfs())      # Sampling freq
    nTot=int(net.getNband())   # Number of bands
    listFe=net.getListFe()     # List of sampling freqs
    listTtot=net.getListTtot() # List of frame sizes
    tTot=net.getBlockLength()  # Total length of a block
    nptsHF=int(tTot*fs)        # Size of a block in the original frame

    step=int(0.05*fs) 


    # 3. Trained network is loaded, one can use it on data
    inputFrame=gd.GenDataSet.readFrame(args.framefile)
    
    sample=inputFrame.getFrame()
    nblocks=int(len(sample[0])/step) # Number of steps to analyze the frame
    weight_sharing=npy.array(sample[1],dtype=npy.float32)
    output=[]
    output_S=[]
    Frameref=[]

    for j in range(nTot):
        Frameref.append([])

    # 4. Loop over frames to perform the inference

    for i in range(nblocks):

        tmpfrm=[]
        finalFrame=[]
        Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
        ref=Frameref[0]
                         
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
    
        data=npy.array(fFrame.reshape(1,-1,1),dtype=npy.float32)

        if data.shape[1]<npts: # Safety cut at the end
            break

        TestSet=(data,weight_sharing)
        
        #res=usoftmax_f(net(TestSet[0].as_in_ctx(device),TestSet[1].as_in_ctx(device))).asnumpy()
        
        cut_top = 0
        cut_bottom = 0
        list_inputs_val=[]
                    
        for k in range(nTot):
            cut_top += int(listTtot[k]*fs)
            list_inputs_val.append(data[:,cut_bottom:cut_top,:])
            cut_bottom = cut_top
            #print(cut_bottom,cut_top,)
        
        res = usoftmax_f(net.model.predict(list_inputs_val,verbose=0))
        
        output.append(res.T[1])
        if (res.T[1]>0.999):
            output_S.append(res.T[1])
            print("Potential signal at t=",(i*step+nptsHF)*(1/fs),res.T[1])
        else:
            output_S.append(0)
        
    finalres=npy.array(output)
    finalres_S=npy.array(output_S)
    
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
