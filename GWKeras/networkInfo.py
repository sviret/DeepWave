import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, initializers, optimizers
import numpy as npy
import matplotlib.pyplot as plt
from time import process_time
import os
import sys
import pickle
#import visualkeras
from tensorflow.keras.utils import plot_model
npy.set_printoptions(threshold=npy.inf)



'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network","-n", help="Fichier pickle contenant le reseau entrainé",default=None)
   
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''

def main():
    
    # 1. Start by parsing the input options
    args = parse_cmd_line()
                
    # 2. Then retrieve the network and initialize it
    f=open(args.network,mode='rb')
    net=pickle.load(f)
    f.close()

    netname=((args.network).split("/")[1]).split(".p")[0]

    model=net.model

    n_mult=0
    n_add=0

    f = open(f'output_{netname}.txt','w')

    f.write(f"Print network information for input net {netname}\n")
    f.write("\n")

    layers=model.layers
    compt=0

    f.write(f"This network contains {len(layers)} layers\n")
    f.write("\n")    

    for layer in layers:
        
        data=model.get_layer(layer.name).get_weights()
        f.write(f"-> Layer {compt}\n")
        f.write(f"Name: {layer.name}\n")

        compt+=1

        if compt==1: # Just input vector as first entry 
            continue

        f.write(f"Input size {layer.input_shape}\n")
        f.write(f"Output size {layer.output_shape}\n")
        f.write(f"Parameters:\n")

        if len(data)==0:
            continue
        for wgh in data:
            wght=npy.asarray(wgh)
            f.write(f"   ->Param container size {wght.shape}\n")
            f.write(f"   {wght}\n")

            if 'conv1d' in layer.name and len(wght.shape)==3:
                
                mult=layer.output_shape[1]*wght.shape[0]*wght.shape[1]*wght.shape[2]
                add=mult
                #print(layer.name,mult)
                n_mult+=mult
                n_add+=add

            if 'batch_norm' in layer.name:
                #https://keras.io/api/layers/normalization_layers/batch_normalization/
                mult=layer.output_shape[1]*2
                add=layer.output_shape[1]*3
                #print(layer.name,mult)
                n_mult+=mult/4
                n_add+=add/4

            if 'dense' in layer.name and len(wght.shape)==2:
       
                mult=wght.shape[0]*wght.shape[1]
                add=wght.shape[0]
                #print(layer.name,mult)
                n_mult+=mult
                n_add+=add

        f.write(f"\n")


    f.write(f"Summary     :\n")
    f.write(f"-> Total number of additions      : {n_add}\n")
    f.write(f"-> Total number of multiplications: {n_mult}\n")
    f.close()
    plot_model(model, to_file=f'output_{netname}.png', show_shapes=True)
    #visualkeras.layered_view(model, to_file=f'output_{netname}_vk.png')


    

    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
