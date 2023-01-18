///////////////////////////////////
//
//  Exercise 1: play with the FFT
//
//  Goal of the exercise:
//
//  Opening the root file created with og_ana code and
//  display the signal, its FFT component, and the inverse FFT
//
//  This macro contains the answers to exercise 1 of the following page:
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
    
  TChain *Signalinfo = new TChain("FFT");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *H = new std::vector<double>;
  std::vector<double>  *Hfr = new std::vector<double>;
  std::vector<double>  *Hfi = new std::vector<double>;
  std::vector<double>  *Hn = new std::vector<double>;
  double t_init;
  double t_bin;
  double f_init;
  double f_bin;
    
  Signalinfo->SetBranchAddress("Signal",&H);
  Signalinfo->SetBranchAddress("Signal_r",&Hn);
  Signalinfo->SetBranchAddress("Hfr",&Hfr);
  Signalinfo->SetBranchAddress("Hfi",&Hfi);
  Signalinfo->SetBranchAddress("t_init",&t_init);
  Signalinfo->SetBranchAddress("t_bin",&t_bin);
  Signalinfo->SetBranchAddress("f_init",&f_init);
  Signalinfo->SetBranchAddress("f_bin",&f_bin);
    
  // Get the signal
  Signalinfo->GetEntry(0);
    
  int length=static_cast<int>(H->size());
  std::cout << "The signal contains " << length << " data points" << std::endl;
    
  double t_min=100;
  double t_max=-100;
  double f_min=100;
  double f_max=-100;
    
  // We do a first loop to get the time limit of the chirp produced
    
  for (int i=0;i<length;++i)
  {
    if (H->at(i)==0) continue;
    if (t_init+i*t_bin<t_min) t_min=t_init+i*t_bin;
    if (t_init+i*t_bin>t_max) t_max=t_init+i*t_bin;
    if (f_init+i*f_bin<f_min) f_min=f_init+i*f_bin;
    if (f_init+i*f_bin>f_max) f_max=f_init+i*f_bin;
  }
    
    std::cout << "Signal is comprised between " << t_min << " and " << t_max << "  secs " << std::endl;
    
    // Now we create an histogram to plot the signal:
    
    int n_bins = int((t_max-t_min)/t_bin);
    int n_binsf = int((f_max-f_min)/f_bin);
    TH2F *Chirp  = new TH2F("Signal","Signal", n_bins,t_min,t_max, 200,-1.1,1.1);
    TH2F *Chirpn = new TH2F("Signaln","Signaln", n_bins,t_min,t_max, 200,-1.1,1.1);
    TH2F *Chirpf = new TH2F("Signalf","Signalf", n_binsf,f_min,f_max/2, 200,0.,1.1);
    
    for (int i=0;i<length;++i)
    {
        if (H->at(i)==0) continue;
        Chirp->Fill(t_init+i*t_bin,H->at(i));
        Chirpn->Fill(t_init+i*t_bin,Hn->at(i));
        Chirpf->Fill(f_init+i*f_bin,1/f_bin*sqrt(Hfr->at(i)*Hfr->at(i)+Hfi->at(i)*Hfi->at(i)));
    }
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Chirp theoretical signal",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.04);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);
    c1->Divide(1,2);
    c1->cd(1);

    Chirp->GetXaxis()->SetLabelSize(0.05);
    Chirp->GetXaxis()->SetTitle("Time (in s)");
    Chirp->SetMarkerStyle(5);
    Chirp->Draw();
    Chirpn->SetMarkerStyle(24);
    Chirpn->SetMarkerColor(2);
    Chirpn->Draw("same");
    
    
    c1->cd(2);

    Chirpf->GetXaxis()->SetLabelSize(0.05);
    Chirpf->GetXaxis()->SetTitle("Frequency (in Hz)");
    Chirpf->SetMarkerStyle(8);
    Chirpf->Draw();
    
    c1->Update();
}
