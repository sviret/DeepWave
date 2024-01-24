///////////////////////////////////
//
//  Exercise 2: play with the chirp signal
//
//  Goal of the exercises:
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
// Plot the signal s(t), coalescence at t=0
// Question 2.2.4
//


void plot_Signal(std::string filename)
{
  // First of all one has to retrieve all the data
  //
  // Their meaning is provided in the file chirpgen.h
  //
  // https://github.com/sviret/DeepWave/blob/OG_lab/MFLight/chirpgen.h#L55
    
  TChain *Signalinfo = new TChain("Chirpinfo");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *H = new std::vector<double>;
  double t_init;
  double t_bin;
  double tchirp;
  double snr;
    
  Signalinfo->SetBranchAddress("Signaln",&H);
  Signalinfo->SetBranchAddress("t_init",&t_init);
  Signalinfo->SetBranchAddress("t_bin",&t_bin);
  Signalinfo->SetBranchAddress("SNR",&snr);
    
  // Get the signal
  Signalinfo->GetEntry(0);
    
  int length=static_cast<int>(H->size());
  std::cout << "The signal contains " << length << " data points" << std::endl;
    
  double t_min=100;
  double t_max=-100;
  double ti;
    
  // We do a first loop to get the time/amplitude limits of the chirp produced
    
  for (int i=0;i<length;++i)
  {
    if (H->at(i)==0) continue;
    ti=t_init+i*t_bin;
    if (ti<t_min) t_min=ti;
    if (ti>t_max) t_max=ti;
  }
    
  std::cout << "Signal is comprised between " << t_min << " and " << t_max << "  secs " << std::endl;
  
  // Now we create a graph to plot the signal:
    
  std::vector<double>  *Signal = new std::vector<double>;
  std::vector<double>  *Time = new std::vector<double>;
    
  Signal->clear();
  Time->clear();
    
  for (int i=0;i<length;++i)
  {
    if (H->at(i)==0) continue;
    ti=t_init+i*t_bin-t_max;

    Time->push_back(ti);
    Signal->push_back(snr*H->at(i));
  }
    
  TGraph* Chp    = new TGraph(Time->size(), &Time->at(0), &Signal->at(0));
    
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

    Chp->GetXaxis()->SetLabelSize(0.03);
    Chp->GetXaxis()->SetTitle("Time before coalescence (in s)");
    Chp->GetYaxis()->SetTitle("h_of_t (no units)");
    Chp->SetLineWidth(2);
    Chp->Draw("AL");
    c1->Update();
    c1->SaveAs("Chirp.png");
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
    double snr;
    
    Signalinfo->SetBranchAddress("Signaln",&H);
    Signalinfo->SetBranchAddress("H",&Raw);
    Signalinfo->SetBranchAddress("t_init",&t_init);
    Signalinfo->SetBranchAddress("t_bin",&t_bin);
    Signalinfo->SetBranchAddress("tchirp",&tchirp);
    Signalinfo->SetBranchAddress("SNR",&snr);
    
    // Get the signal
    Signalinfo->GetEntry(0);
    
    int length=static_cast<int>(H->size());
    std::cout << "The signal contains " << H->size() << " data points" << std::endl;
    
    double t_min=100;
    double t_max=-100;
    double ti;
    double h_max=0;
    
    // We do a first loop to get the limits of the raw signal
    
    for (int i=0;i<length;++i)
    {
        ti=t_init+i*t_bin;
        if (ti<t_min) t_min=ti;
        if (ti>t_max) t_max=ti;
        if (fabs(Raw->at(i))>h_max) h_max=fabs(Raw->at(i));
    }
    int n_bins = int(4/t_bin);
    TH2F *Raws = new TH2F("Raws","Raws", n_bins,-2,2, 400,-1.05*h_max,1.05*h_max);
    
    std::vector<double>  *Signal = new std::vector<double>;
    std::vector<double>  *Time = new std::vector<double>;
    
    Signal->clear();
    Time->clear();
    std::cout << "We will display the data around the coalescence " << std::endl;
    
    // Now we create an histogram to plot the signal:

    for (int i=0;i<length;++i)
    {
        ti=t_init+i*t_bin-tchirp;
        if (ti>-2 && ti<2) Raws->Fill(ti,Raw->at(i));
            
        // For the signal tc=tmax
        ti=t_init+i*t_bin-t_max;
        
        if (ti>-2)
        {
            Time->push_back(ti);
            Signal->push_back(snr*H->at(i));
        }
    }
    
    TGraph* Chirp    = new TGraph(Time->size(), &Time->at(0), &Signal->at(0));
 
    
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Signals and raw data",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.05);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);
    c1->Divide(1,2);
    c1->cd(1);
    c1->cd(1)->SetGridy();
    Raws->SetMarkerStyle(20);
    Raws->SetMarkerSize(0.1);
    Raws->GetXaxis()->SetTitle("Time before coalescence (in s)");
    Raws->GetXaxis()->SetLabelSize(0.05);
    Raws->GetXaxis()->SetTitleSize(0.05);
    Raws->GetXaxis()->SetTitleOffset(1);
    Raws->GetXaxis()->SetTitleFont(42);
    Raws->GetYaxis()->SetTitle("h_of_t");
    Raws->GetYaxis()->SetLabelFont(42);
    Raws->GetYaxis()->SetLabelSize(0.05);
    Raws->GetYaxis()->SetTitleSize(0.06);
    Raws->GetYaxis()->SetTitleOffset(0.25);
    Raws->GetYaxis()->SetTitleFont(42);
    Raws->Draw();
    Chirp->SetLineWidth(2);
    Chirp->SetLineColor(2);
    Chirp->Draw("Lsame");
    
    c1->cd(2);
    c1->cd(2)->SetGridy();
    Chirp->SetMarkerStyle(20);
    Chirp->SetMarkerSize(0.3);
    Chirp->SetMarkerColor(2);
    Chirp->GetXaxis()->SetTitle("Time before coalescence (in s)");
    Chirp->GetXaxis()->SetLabelSize(0.05);
    Chirp->GetXaxis()->SetTitleSize(0.05);
    Chirp->GetXaxis()->SetTitleOffset(1);
    Chirp->GetXaxis()->SetTitleFont(42);
    Chirp->GetYaxis()->SetTitle("h_of_t");
    Chirp->GetYaxis()->SetLabelFont(42);
    Chirp->GetYaxis()->SetLabelSize(0.05);
    Chirp->GetYaxis()->SetTitleSize(0.06);
    Chirp->GetYaxis()->SetTitleOffset(0.25);
    Chirp->GetYaxis()->SetTitleFont(42);
    Chirp->SetLineWidth(2);
    Chirp->Draw("AL");
    
    c1->Update();
    //c1->SaveAs("Chirp.png");
    
}

//
// Plot the raw signal Fourier transform amplitude h^tilde(f) as a function of f
// (Optional)

void plot_RawSpectrum(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *Signalinfo = new TChain("Chirpinfo");
  Signalinfo->Add(filename.c_str());

  std::vector<double>  *RawHFr = new std::vector<double>;
  std::vector<double>  *RawHFi = new std::vector<double>;
  std::vector<double>  *SFr = new std::vector<double>;
  std::vector<double>  *SFi = new std::vector<double>;
  std::vector<double>  *SnFr = new std::vector<double>;
  std::vector<double>  *SnFi = new std::vector<double>;
  std::vector<double>  *NFr = new std::vector<double>;
  std::vector<double>  *NFi = new std::vector<double>;
  double f_init;
  double f_bin;
  double snr;
    
  Signalinfo->SetBranchAddress("Hfr",&RawHFr);
  Signalinfo->SetBranchAddress("Hfi",&RawHFi);
  Signalinfo->SetBranchAddress("Sfr",&SFr);
  Signalinfo->SetBranchAddress("Sfi",&SFi);
  Signalinfo->SetBranchAddress("Snfr",&SnFr);
  Signalinfo->SetBranchAddress("Snfi",&SnFi);
  Signalinfo->SetBranchAddress("Nfr",&NFr);
  Signalinfo->SetBranchAddress("Nfi",&NFi);
  Signalinfo->SetBranchAddress("f_init",&f_init);
  Signalinfo->SetBranchAddress("f_bin",&f_bin);
  Signalinfo->SetBranchAddress("SNR",&snr);
    
  // Get the signal
  Signalinfo->GetEntry(0);
    
  int length=static_cast<int>(RawHFi->size());
  std::cout << "The signal contains " << length << " data points" << std::endl;
    
  double f_max=f_init+length*f_bin;
    
  std::cout << "Signal is comprised between " << f_init << " and " << f_max << "  Hz " << std::endl;
  
  double max_ampl=0;
  double max_ampls=0;
  double max_amplsn=0;
  double max_ampln=0;
    
  double ampls,amplh,ampln,amplsn;

  for (int i=0;i<length;++i)
  {
    amplh=sqrt(RawHFi->at(i)*RawHFi->at(i)+RawHFr->at(i)*RawHFr->at(i));
    ampls=sqrt(SFi->at(i)*SFi->at(i)+SFr->at(i)*SFr->at(i));
    amplsn=sqrt(SnFi->at(i)*SnFi->at(i)+SnFr->at(i)*SnFr->at(i));
    ampln=sqrt(NFi->at(i)*NFi->at(i)+NFr->at(i)*NFr->at(i));
      
    if (amplh>max_ampl) max_ampl=amplh;
    if (ampls>max_ampls) max_ampls=ampls;
    if (amplsn>max_amplsn) max_amplsn=amplsn;
    if (ampln>max_ampln) max_ampln=ampln;
  }
    
  // Now we create an histogram to plot the signal:
    
  TH2F *RawTF = new TH2F("Raw FFT","Raw FFT", 500,f_init,f_max, 400,0.,1.1*max_ampl);
  TH2F *STF = new TH2F("Signal FFT","Signal FFT", 500,f_init,f_max, 400,0.,1.1*max_ampls);
  TH2F *SnTF = new TH2F("Signal FFT","Signal FFT", 500,f_init,f_max, 400,0.,1.1*max_amplsn);
  TH2F *NTF = new TH2F("Noise FFT","Noise FFT", 500,f_init,f_max, 400,0.,1.1*max_ampln);
    
  for (int i=0;i<length;++i)
  {
    amplh=sqrt(RawHFi->at(i)*RawHFi->at(i)+RawHFr->at(i)*RawHFr->at(i));
    ampls=sqrt(SFi->at(i)*SFi->at(i)+SFr->at(i)*SFr->at(i));
    amplsn=sqrt(SnFi->at(i)*SnFi->at(i)+SnFr->at(i)*SnFr->at(i));
    ampln=sqrt(NFi->at(i)*NFi->at(i)+NFr->at(i)*NFr->at(i));
      
    RawTF->Fill(f_init+i*f_bin,amplh);
    STF->Fill(f_init+i*f_bin,ampls);
    SnTF->Fill(f_init+i*f_bin,amplsn);
    NTF->Fill(f_init+i*f_bin,ampln);
  }
    
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Raw signal spectrum",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.04);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);
    c1->Divide(2,2);
    c1->cd(1);
    RawTF->GetXaxis()->SetLabelSize(0.05);
    RawTF->GetXaxis()->SetTitleSize(0.05);
    RawTF->GetXaxis()->SetTitleOffset(1);
    RawTF->GetXaxis()->SetTitleFont(42);
    RawTF->GetYaxis()->SetLabelFont(42);
    RawTF->GetYaxis()->SetLabelSize(0.05);
    RawTF->GetYaxis()->SetTitleSize(0.06);
    RawTF->GetYaxis()->SetTitleOffset(0.65);
    RawTF->GetYaxis()->SetTitleFont(42);
    RawTF->GetXaxis()->SetTitle("Frequency (in Hz)");
    RawTF->GetYaxis()->SetTitle("h(f) normalized and whitened");
    RawTF->SetMarkerStyle(20);
    RawTF->SetMarkerSize(0.2);
    RawTF->Draw();

    c1->cd(2);
    SnTF->GetXaxis()->SetLabelSize(0.05);
    SnTF->GetXaxis()->SetTitleSize(0.05);
    SnTF->GetXaxis()->SetTitleOffset(1);
    SnTF->GetXaxis()->SetTitleFont(42);
    SnTF->GetYaxis()->SetLabelFont(42);
    SnTF->GetYaxis()->SetLabelSize(0.05);
    SnTF->GetYaxis()->SetTitleSize(0.06);
    SnTF->GetYaxis()->SetTitleOffset(0.65);
    SnTF->GetYaxis()->SetTitleFont(42);
    SnTF->GetXaxis()->SetTitle("Frequency (in Hz)");
    SnTF->GetYaxis()->SetTitle("s(f) raw normalized and whitened");
    SnTF->SetMarkerStyle(20);
    SnTF->SetMarkerSize(0.2);
    SnTF->Draw();
    
    c1->cd(3);
    NTF->GetXaxis()->SetLabelSize(0.05);
    NTF->GetXaxis()->SetTitleSize(0.05);
    NTF->GetXaxis()->SetTitleOffset(1);
    NTF->GetXaxis()->SetTitleFont(42);
    NTF->GetYaxis()->SetLabelFont(42);
    NTF->GetYaxis()->SetLabelSize(0.05);
    NTF->GetYaxis()->SetTitleSize(0.06);
    NTF->GetYaxis()->SetTitleOffset(0.65);
    NTF->GetYaxis()->SetTitleFont(42);
    NTF->GetXaxis()->SetTitle("Frequency (in Hz)");
    NTF->GetYaxis()->SetTitle("n(f) raw");
    NTF->SetMarkerStyle(20);
    NTF->SetMarkerSize(0.2);
    NTF->Draw();
    
    c1->cd(4);
    STF->GetXaxis()->SetLabelSize(0.05);
    STF->GetXaxis()->SetTitleSize(0.05);
    STF->GetXaxis()->SetTitleOffset(1);
    STF->GetXaxis()->SetTitleFont(42);
    STF->GetYaxis()->SetLabelFont(42);
    STF->GetYaxis()->SetLabelSize(0.05);
    STF->GetYaxis()->SetTitleSize(0.06);
    STF->GetYaxis()->SetTitleOffset(0.65);
    STF->GetYaxis()->SetTitleFont(42);
    STF->GetXaxis()->SetTitle("Frequency (in Hz)");
    STF->GetYaxis()->SetTitle("s(f) raw");
    STF->SetMarkerStyle(20);
    STF->SetMarkerSize(0.2);
    STF->Draw();
    c1->Update();
    //c1->SaveAs("RawSpectrum.png");
    
}
