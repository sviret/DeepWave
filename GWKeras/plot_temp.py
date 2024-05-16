from numpy import load
import matplotlib.pyplot as plt
import pandas as pd 
import pickle

def main():
    import gendata    as gd
    data=gd.GenDataSet.readGenerator('generators/tmpls-1chunk-analytic-EOB-optimal-0-10.0s.p')

    tmpl=data.getTemplate(rank=100)
    gfg = pd.Series(tmpl) 

    tmplist=data.getBkParams()
    print(tmplist)

    gfg.plot() 
    plt.show() 


if __name__ == "__main__":
    main()