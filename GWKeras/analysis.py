import numpy as npy
import os
import math
import ROOT as root
from array import array


file1 = root.TFile.Open("output_0123.root","READ")
file2 = root.TFile.Open("test_ungated.root","READ")

mctruth = file1.Get("injections")
network = file2.Get("netout")

injections=[]
data=[]

cut=0.7
delta_t=0
t0=1369200000
for entry in mctruth:
    inj=[]
    
    if (entry.tcoalL>10):
        inj.append(entry.tcoalL-t0)
        inj.append(entry.tcoalH-t0)
    else:
        inj.append(entry.tcoal+entry.tcoalL-t0)
        inj.append(entry.tcoal+entry.tcoalH-t0)
    inj.append(entry.mass1)
    inj.append(entry.mass2)
    inj.append(entry.SNR_H)
    inj.append(entry.SNR_L)
    inj.append(entry.s1x)
    inj.append(entry.s1y)
    inj.append(entry.s1z)
    inj.append(entry.s2x)
    inj.append(entry.s2y)
    inj.append(entry.s2z)
    inj.append(entry.chi_eff)
    inj.append(entry.chi_p)
    injections.append(inj)

highSigs=[]
compt=0
for entry in network:

    dat=[]
    
    dat.append(entry.time-delta_t)
    dat.append(entry.livingston)
    dat.append(entry.hanford)
    dat.append(-1)

    time=entry.time-delta_t-3
    
    dtoinj=time%6
    rkinj=int(time/6.)
    
    if time<0:
        dist=dtoinj
    elif dtoinj<=3.:
        dist=dtoinj
    else:
        dist=dtoinj-12.
        rkinj=rkinj+1

    dat.append(dist)
    dat.append(rkinj)
    #print(dist,rkinj)
    data.append(dat)

    if (dat[1]>cut or dat[2]>2*cut ):
        highSigs.append(dat)
    

file1.Close()
file2.Close()

print("We have a total of",len(injections),"injection in this chunk")
print("The network found",len(highSigs),"points with a network output > ",cut,"for at least one interferometer")
nfoundH=0
nfoundL=0
nfoundH_h=0
nfoundL_h=0
ntot=0

for inj in injections:
    tL = inj[0]
    tH = inj[1]
    foundL=0
    foundH=0
    for sig in highSigs:
        diffL=sig[0]-tL
        diffH=sig[0]-tH
        if diffL<-1 and diffH<-1:
            continue
        if diffL>1 and diffH>1:
            break
        if -1<diffL<1 and sig[1]>cut:
            foundL+=1
            sig[3]=1
        if -1<diffH<1 and sig[2]>cut:
            foundH+=1
    inj.append(foundL)
    inj.append(foundH)
    if foundH>=1:
        nfoundH+=1
        nfoundH_h+=foundH
    if foundL>=1:
        nfoundL+=1
        nfoundL_h+=foundL
    ntot+=1
    if ntot%1000==0:
        print(nfoundH,"/",nfoundL,"/",ntot)

for sig in highSigs:
    if math.fabs(sig[4])>1:
        print(sig[1],sig[0],sig[4],sig[5]*24+21)

print(nfoundH,"/",nfoundL,"/",ntot)
print(nfoundH_h,"/",nfoundL_h,"/",len(highSigs))

M1      = array('d', [0])
M2      = array('d', [0])
t_coalH = array('d', [0])
t_coalL = array('d', [0])
SNR_H   = array('d', [0])
SNR_L   = array('d', [0])
s1x     = array('d', [0])
s1y     = array('d', [0])
s1z     = array('d', [0])
s2x     = array('d', [0])
s2y     = array('d', [0])
s2z     = array('d', [0])
chi_eff = array('d', [0])
chi_p   = array('d', [0])
matchedL= array('d', [0])
matchedH= array('d', [0])
        
file3 = root.TFile.Open('final.root', 'recreate')
tree3 = root.TTree("injections", "injections")
    
tree3.Branch("mass1",     M1,     'M1/D')
tree3.Branch("mass2",     M2,     'M2/D')
tree3.Branch("tcoalH",    t_coalH,'t_coalH/D')
tree3.Branch("tcoalL",    t_coalL,'t_coalL/D')
tree3.Branch("SNR_H",     SNR_H,  'SNR_H/D')
tree3.Branch("SNR_L",     SNR_L,  'SNR_L/D')
tree3.Branch("s1x",       s1x,    's1x/D')
tree3.Branch("s1y",       s1y,    's1y/D')
tree3.Branch("s1z",       s1z,    's1z/D')
tree3.Branch("s2x",       s2x,    's2x/D')
tree3.Branch("s2y",       s2y,    's2y/D')
tree3.Branch("s2z",       s2z,    's2z/D')
tree3.Branch("chi_eff",   chi_eff,'chi_eff/D')
tree3.Branch("chi_p",   chi_p,'chi_p/D')
tree3.Branch("matchedH",   matchedH,  'matchedH/D')
tree3.Branch("matchedL",   matchedL,  'matchedL/D')

for inj in injections:

    t_coalL[0]=inj[0]
    t_coalH[0]=inj[1]
    M1[0]=inj[2]
    M2[0]=inj[3]
    SNR_H[0]=inj[4]
    SNR_L[0]=inj[5]
    s1x[0]=inj[6]
    s1y[0]=inj[7]
    s1z[0]=inj[8]
    s2x[0]=inj[9]
    s2y[0]=inj[10]
    s2z[0]=inj[11]
    chi_eff[0]=inj[12]
    chi_p[0]=inj[13]
    matchedL[0]=inj[14]
    matchedH[0]=inj[15]
    tree3.Fill()

file3.Write()
file3.Close()

