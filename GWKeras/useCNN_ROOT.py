import ROOT as root
from array import array
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

Define here the input parameters of the useCNN macro

'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framefile","-f", help="Fichier pickle contenant les données à étudier",default=None)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de training",default=None)
    parser.add_argument("--verbose","-v",help="Affiche l'évolution de l'entraînement",action="store_true")
    parser.add_argument("--network","-n", help="Fichier h5 contenant le reseau entrainé",default=None)
    parser.add_argument("--output","-o", help="Fichier root contenant les resultats",default='result.root')
    args = parser.parse_args()
    return args

'''
The main testing macro starts here
'''

def main():

    import trainCNN   as tr
    import gendata    as gd   # To process the input data frame
    
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
    
    # 3. Create a ROOTuple to store the relavant info for subsequent analysis
    dettype= array('d', [0])
    wclus  = array('d', [0])
    tclus  = array('d', [0])
    mclus  = array('d', [0])
    sclus  = array('d', [0])
    clus_t = array('d', [0])
    MC     = array('d', [0])
    ratio  = array('d', [0])
    t_coal = array('d', [0])
    outnet_H  = array('d', [0])
    outnet_L  = array('d', [0])
    outtime = array('d', [0])
    
    M1     = array('d', [0])
    M2     = array('d', [0])
    SNR_H  = array('d', [0])
    SNR_L  = array('d', [0])
    
    file = root.TFile.Open(args.output, 'recreate')
    tree = root.TTree("net", "net")
    tree2 = root.TTree("netout", "netout")
    tree3 = root.TTree("injections", "injections")
    
    
    # Creating branches
    tree.Branch("detector",   dettype,'dettype/D')
    tree.Branch("width",      wclus,  'wclus/D')
    tree.Branch("time",       tclus,  'tclus/D')
    tree.Branch("val",        mclus,  'mclus/D')
    tree.Branch("sig",        sclus,  'sclus/D')
    tree.Branch("matched",    clus_t, 'clus_t/D')
    tree.Branch("mchirp",     MC,     'MC/D')
    tree.Branch("mratio",     ratio,  'ratio/D')
    tree.Branch("tcoal",      t_coal, 't_coal/D')

    tree2.Branch("hanford",   outnet_H, 'outnet_H/D')
    tree2.Branch("livingston",outnet_L, 'outnet_L/D')
    tree2.Branch("time",      outtime,'outtime/D')

    tree3.Branch("mass1",     M1,     'M1/D')
    tree3.Branch("mass2",     M2,     'M2/D')
    tree3.Branch("tcoal",     t_coal, 't_coal/D')
    tree3.Branch("SNR_H",     SNR_H,  'SNR_H/D')
    tree3.Branch("SNR_L",     SNR_L,  'SNR_L/D')


    # 4. Trained network is loaded, one can use it on data
    
    inputFrame=gd.GenDataSet.readFrame(args.framefile) # Get the input data
    
    injections=inputFrame.getTruth() # Retrieve the MC injections
    
    for inj in injections:
            
        M1[0]=inj[0]
        M2[0]=inj[1]
        SNR_H[0]=inj[2]
        SNR_L[0]=inj[2]
        t_coal[0]=inj[3]
        tree3.Fill()

    ninj=len(injections)
    
    # The weight
    # 'special' means you use only the last chunk CNN
    # 'balance' means you use all the chunks (one CNN per frequency band) (SV Still under development)
    
    sample_H=inputFrame.getFrame(weight='special',det=0) # Hanford
    sample_L=inputFrame.getFrame(weight='special',det=1) # Livingston

    # Those lines are for real data
    '''
    filei  = root.TFile.Open("batch0123_white.root","READ")
    treei  = filei.Get("white")

    h_of_t=[]
    h_of_ti=[]
    timing=[]
    
    for entry in treei:
        h_of_t.append(entry.h_of_t_white_L)
        h_of_ti.append(entry.h_of_ti_white_L)
        timing.append(entry.time)
        #print(entry.tme)
        
    filei.Close()

    h_of_t=npy.asarray(h_of_t)
    h_of_ti=npy.asarray(h_of_ti)
    timing=npy.asarray(timing)
    t_zero=timing[0]
    
    print("We are starting the analysis at t=",t_zero)
  
    sample_H_new=h_of_t
    sample_L_new=h_of_t
    '''
    
    step=int(0.05*fs)          # Step between two blocks (important to take a short value to build the clusters)

    nblocks=int(len(sample_H[0])/step) # Number of steps to analyze the total frame

    weight_sharing=npy.array(sample_H[1],dtype=npy.float32)
  
    sample=sample_H
    highSigs_overall=[]
    batchSize=200  
    output_H=[]
    output_L=[]
    
    for det in range(2):
    
        if (det==1):
            sample=sample_L
    
        output=[]
        Frameref=[]
        hits=[]
        sigs=[]
        highSigs=[]
        weights=[]

        for j in range(batchSize):
            weights.append(weight_sharing[0])
        
        batch_wght=npy.asarray(weights)

        for j in range(nTot):
            Frameref.append([])
    
        for i in range(int(nblocks/batchSize)): # Loop over all steps

            if i%1000==0:
                print(i,"/",int(float(nblocks)/float(batchSize)),"processed...")

            tmpfrm=[]
            finalFrame=[]
            Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
            ref=Frameref[0]
            fdset = []
            finaldset=[]
            
            for j in range(batchSize):
                tmpset=[]
                # The temporal realization at max sampling
                ref=sample[0][(i*batchSize+j)*step:(i*batchSize+j)*step+nptsHF]
                for k in range(nTot): # Resample the block
                
                    #Pick up the data chunk
                    ndatapts=int(listTtot[k]*fs)
                    nttt=len(ref)
            
                    Nt=ref[-ndatapts:]
                    ref=ref[:nttt-ndatapts]
                    decimate=int(fs/listFe[k])
                    Frameref[k]=Nt[::int(decimate)]
                    tmpset.append(npy.asarray(Frameref[k]))
                
                sec=npy.concatenate(tmpset)
                if len(sec)<npts:
                    break
                finaldset.append(sec)
                
            if (len(finaldset)==0):
                break

            fdset=npy.asarray(finaldset)

            if fdset.shape[0]<batchSize:
                weights=[]
                for jj in range(fdset.shape[0]):
                    weights.append(weight_sharing[0])
        
                batch_wght=npy.asarray(weights)
            
            if fdset.shape[0]<batchSize:
                data=npy.array(fdset.reshape(fdset.shape[0],-1,1),dtype=npy.float32)
            else:
                data=npy.array(fdset.reshape(batchSize,-1,1),dtype=npy.float32)
            
            weight_final=npy.array(batch_wght,dtype=npy.float32)
            
            if data.shape[1]<npts: # Safety cut at the end
                break
            
            # We finally have the input info to feed the network
            TestSet=(data,weight_final)
        
            cut_top = 0
            cut_bottom = 0
            list_inputs_val=[]
                    
            for k in range(nTot):
                cut_top += int(listTtot[k]*fs)
                list_inputs_val.append(data[:,cut_bottom:cut_top,:])
                cut_bottom = cut_top
       
            res = usoftmax_f(net.model.predict(list_inputs_val,verbose=0))

            for dat in res:
                netres=dat.T[1] # 0=pure background, 1=pure signal
                output.append(netres) # Value for all the blocks
        
        # Inference is done, output of the network is stored in vector output
         
        i=0

        # Loop over the output to form clusters

        for netres in output:
        
            if det==0:
                output_H.append(netres)
            else:
                output_L.append(netres)
        
            nclust=len(highSigs)
            #
            # Look if this value can belong to a cluster
            #
            missflag=0
            if (netres>0.98): # Interesting value, does it belong to a cluster
                if nclust==0: # No cluster yet, create one
                    highSigs.append([i])
                else:         # Clusters exist, check is we are in or not
                    curr_clust=highSigs[nclust-1]
                    sclust=len(curr_clust)
                
                    # End of the last cluster is the previous block, we add the new hit to the cluster
                    if (i-curr_clust[sclust-1]==1):
                        curr_clust.append(i)
                        highSigs.pop()
                        highSigs.append(curr_clust)
                    # End of the last cluster is the next to previous block, we add the new hit to the cluster
                    # As we accept one missed hit (just one)
                    elif (i-curr_clust[sclust-1]==2 and missflag==0):
                        #curr_clust.append(i-1)
                        curr_clust.append(i)
                        highSigs.pop()
                        highSigs.append(curr_clust)
                        missflag=1
                    # Last cluster is too far away, create a new cluster
                    else:
                        if sclust==1:
                            highSigs.pop() # The last recorded cluster was one block only, remove the artefact
                        highSigs.append([i]) # New cluster
                        missflag=0
            i+=1
            
            
        # End of cluster building stage
            
        nclust=len(highSigs)
        if nclust==0:
            print("No clus!!")
            sys.exit()
    
        # Remove the last cluster if only one block long
        if len(highSigs[len(highSigs)-1])==1:
            highSigs.pop()


        # Now determine the cluster coordinates
        #
        # Center, average network output value, sigma of this average

        clust_truth=[]  # Clus is matched to an injections
        clust_vals=[]   # Cluster properties
    
        for clust in highSigs:
            clust_truth.append(-1)
        
            clust_val=0
            clust_cen=0
            clust_sd=0
        
            for val in clust:
                res=output[val]
                clust_val+=float(res)
                clust_cen+=float(val)
            clust_val/=len(clust)
            clust_cen/=len(clust)
    
            for val in clust:
                res=float(output[val])
                clust_sd+=(res-clust_val)*(res-clust_val)
            
            clust_sd=npy.sqrt(clust_sd/len(clust))
            clust_vals.append([clust_val,clust_cen,(clust_cen*step+nptsHF)*(1/fs),clust_sd])


        # Now establish the correspondence between clusters and injection

        found=0
        idx_inj=0
    
        # Look at which injections have lead to a cluster
        for inj in injections:
    
            inj.append(0)
            tcoal=inj[3]
        
            # Check if it's into one cluster
   
            idx_clus=0
            for clust in highSigs:
        
                tstart=(clust[0]*step+nptsHF)*(1/fs)
                tend=(clust[len(clust)-1]*step+nptsHF)*(1/fs)
        
                #if (len(clust)>10):
                #    print(tcoal,tstart,tend,len(clust))

                if (tcoal>tstart-0.1 and tcoal<tend): # Injection is in the cluster
                    found+=1
                    inj[4+2*det]=len(clust)
                    clust_truth[idx_clus]=idx_inj
                    inj.append(clust_vals[idx_clus])
                    break
                idx_clus+=1
            idx_inj+=1
        
        print("Found",found,"injections out of",ninj)
        print("Among",len(highSigs),"clusters in total")
    
        # Everything done, one could now store the final result in the ROOTuple

        # Loop over clusters
        for i in range(len(highSigs)):
            clus=highSigs[i]
            clus_par=clust_vals[i]
            dettype[0]=det
            wclus[0]=float(len(clus))
            tclus[0]=clus_par[2]
            mclus[0]=clus_par[0]
            sclus[0]=clus_par[3]
            if clust_truth[i]>=0: # Matched to an injection ??
                inj=injections[clust_truth[i]]
                clus_t[0]=inj[2]
            
                M=inj[0]+inj[1]
                eta=inj[0]*inj[1]/(M**2)
                MC[0]=npy.power(eta,3./5.)*M
                ratio[0]=inj[0]/inj[1]
                t_coal[0]=inj[3]
            else:                 # Noise cluster
                clus_t[0]=0
                MC[0]=0
                ratio[0]=0
                t_coal[0]=0

            tree.Fill()
    
        # Loop of unmatched injections to retrive them in the ROOTuple
        for inj in injections:
            if not inj[4+2*det]:
                inj.append(-1)
                dettype[0]=det
                wclus[0]=-1  # No cluster info there
                tclus[0]=-1
                mclus[0]=-1
                sclus[0]=-1
        
                clus_t[0]=inj[2]
                M=inj[0]+inj[1]
                eta=inj[0]*inj[1]/(M**2)
                MC[0]=npy.power(eta,3./5.)*M
                ratio[0]=inj[0]/inj[1]
                t_coal[0]=inj[3]
                tree.Fill()
    
        highSigs_overall.append(clust_vals)
        highSigs_overall.append(clust_truth)
    
    # End of loop over detectors
    
    '''
    for vec in highSigs_overall:
        print("New det")
        for clus in vec:
            print(clus)
    '''

    for inj in injections:
        print(inj)
    
    for i in range(len(output_L)):
        
        outnet_H[0]=output_H[i]
        outnet_L[0]=output_L[i]
        outtime[0]=(i*step+nptsHF)*(1/fs)
        #outtime[0]=timing[i]
        #print(i,timing[i])
        tree2.Fill()
            
    file.Write()
    file.Close()


    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
