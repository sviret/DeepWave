///////////////////////////////////
//
//  Exercise 1: play with the chirp signal
//
//  Goal of the exercise:
//
//  Opening the root file created with og_ana code and
//  display the signal and noise characteristics
//
//  This macro is the answer of question 2.2.4 of the following page:
//
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////


#include <string>
#include <vector>
#include <iostream>

#include "TStyle.h"
#include "TSystem.h"
#include "TFile.h"
#include "TChain.h"
#include "TBranch.h"
#include "TH2F.h"
#include "TCanvas.h"


//
// Plot the signal s(t), aligne with the correct t_c value
//


void plot_Signal(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *Signalinfo = new TChain("Chirpinfo");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *H = new std::vector<double>;
  double t_init;
  double t_bin;
  double tchirp;
    
  Signalinfo->SetBranchAddress("Signal",&H);
  Signalinfo->SetBranchAddress("t_init",&t_init);
  Signalinfo->SetBranchAddress("t_bin",&t_bin);
  Signalinfo->SetBranchAddress("tchirp",&tchirp);
    
    // Get the signal
    Signalinfo->GetEntry(0);
    
    int length=static_cast<int>(H->size());
    std::cout << "The signal contains " << length << " data points" << std::endl;
    
    double t_min=100;
    double t_max=-100;
    
    // We do a first loop to get the time limit of the chirp produced
    
    for (int i=0;i<length;++i)
    {
        if (H->at(i)==0) continue;
        if (t_init+i*t_bin<t_min) t_min=t_init+i*t_bin;
        if (t_init+i*t_bin>t_max) t_max=t_init+i*t_bin;
    }
    
    std::cout << "Signal is comprised between " << t_min << " and " << t_max << "  secs " << std::endl;
    
    // Now we create an histogram to plot the signal:
    
    int n_bins = int((t_max-t_min)/t_bin);
    TH2F *Chirp = new TH2F("Signal","Signal", n_bins,t_min+tchirp,t_max+tchirp, 200,-100.,100);
    
    for (int i=0;i<length;++i)
    {
        if (H->at(i)==0) continue;
        Chirp->Fill(t_init+i*t_bin+tchirp,H->at(i));
    }
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Chirp theoretical signal",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.08);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);

    Chirp->GetXaxis()->SetLabelSize(0.03);
    Chirp->GetXaxis()->SetTitle("time (in s)");
    Chirp->GetYaxis()->SetTitle("h (a.u.)");
    Chirp->SetMarkerStyle(7);
    Chirp->Draw();
    
    c1->Update();
    //c1->SaveAs("Chirp.png");
}

//
// Plot the raw signal h(t)=s(t)+n(t) in black, and superimpose the signal s(t) in red
//

void plot_RawSignal(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *Signalinfo = new TChain("Chirpinfo");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *H = new std::vector<double>;
  std::vector<double>  *Raw = new std::vector<double>;
  double t_init;
  double t_bin;
  double tchirp;
    
  Signalinfo->SetBranchAddress("Signal",&H);
  Signalinfo->SetBranchAddress("H",&Raw);
  Signalinfo->SetBranchAddress("t_init",&t_init);
  Signalinfo->SetBranchAddress("t_bin",&t_bin);
  Signalinfo->SetBranchAddress("tchirp",&tchirp);
    
    // Get the signal
    Signalinfo->GetEntry(0);
    
    int length=static_cast<int>(H->size());
    std::cout << "The signal contains " << H->size() << " data points" << std::endl;
    
    double t_min=100;
    double t_max=-100;
    
    // We do a first loop to get the time limit of the chirp produced
    
    for (int i=0;i<length;++i)
    {
        if (H->at(i)==0) continue;
        if (t_init+i*t_bin<t_min) t_min=t_init+i*t_bin;
        if (t_init+i*t_bin>t_max) t_max=t_init+i*t_bin;
    }
    
    std::cout << "Signal is comprised between " << t_min << " and " << t_max << "  secs " << std::endl;
    
    // Now we create an histogram to plot the signal:
    
    int n_bins = int((t_max-t_min)/t_bin);
    TH2F *Chirp = new TH2F("Signal","Signal", n_bins,t_min+tchirp,t_max+tchirp, 400,-200.,200);
    TH2F *Raws = new TH2F("Raws","Raws", 6000,-30,30, 400,-200.,200);
    
    for (int i=0;i<length;++i)
    {
        Raws->Fill(t_init+i*t_bin,Raw->at(i));
        if (H->at(i)==0) continue;
        Chirp->Fill(t_init+i*t_bin+tchirp,H->at(i));
    }
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Signals and raw data",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.08);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);

    Raws->GetXaxis()->SetLabelSize(0.03);
    Raws->GetXaxis()->SetTitle("time (in s)");
    Raws->GetYaxis()->SetTitle("h (a.u.)");
    Raws->SetMarkerStyle(20);
    Chirp->SetMarkerStyle(20);
    Raws->SetMarkerSize(0.3);
    Chirp->SetMarkerSize(0.2);
    Chirp->SetMarkerColor(2);
    Raws->Draw();
    Chirp->Draw("same");
    
    c1->Update();
    //c1->SaveAs("Chirp.png");
    
}

//
// Plot the raw signal Fourier transform amplitude h^tilde(f) as a function of f
//

void plot_RawSpectrum(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *Signalinfo = new TChain("Chirpinfo");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *RawHFr = new std::vector<double>;
  std::vector<double>  *RawHFi = new std::vector<double>;
  double f_init;
  double f_bin;
    
  Signalinfo->SetBranchAddress("Hfr",&RawHFr);
  Signalinfo->SetBranchAddress("Hfi",&RawHFi);
  Signalinfo->SetBranchAddress("f_init",&f_init);
  Signalinfo->SetBranchAddress("f_bin",&f_bin);
       
  // Get the signal
  Signalinfo->GetEntry(0);
    
  int length=static_cast<int>(RawHFi->size());
  std::cout << "The signal contains " << length << " data points" << std::endl;
    
  double f_max=f_init+length*f_bin;
    
  std::cout << "Signal is comprised between " << f_init << " and " << f_max << "  Hz " << std::endl;
  
  double max_ampl=0;
  
  for (int i=0;i<length;++i)
  {
    if (sqrt(RawHFi->at(i)*RawHFi->at(i)+RawHFr->at(i)*RawHFr->at(i))>max_ampl)
        max_ampl=sqrt(RawHFi->at(i)*RawHFi->at(i)+RawHFr->at(i)*RawHFr->at(i));
  }
    
  // Now we create an histogram to plot the signal:
    
  TH2F *RawTF = new TH2F("Fourier","Fourier", 1000,f_init,f_max, 400,0.,1.1*max_ampl);

  for (int i=0;i<length;++i)
  {
     RawTF->Fill(f_init+i*f_bin,sqrt(RawHFi->at(i)*RawHFi->at(i)+RawHFr->at(i)*RawHFr->at(i)));
  }
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Raw signal spectrum",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.08);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);

    RawTF->GetXaxis()->SetLabelSize(0.03);
    RawTF->GetXaxis()->SetTitle("Frequency (in Hz)");
    RawTF->GetYaxis()->SetTitle("Amplitude (a.u.)");
    RawTF->SetMarkerStyle(20);
    RawTF->SetMarkerSize(0.2);
    RawTF->Draw();

    c1->Update();
    //c1->SaveAs("RawSpectrum.png");
    
}
