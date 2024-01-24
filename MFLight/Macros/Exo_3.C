///////////////////////////////////
//
//  Exercise 3: play with the template bank
//
//  Goal of the exercise:
//
//  Opening the root file created with og_ana code and
//  display some bank properties
//
//  This macro is the answer of question 2.3.3 of the following page:
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
#include "TGraph.h"

//
// Plot M_chirp vs T_util for all the templates
// Exercise 2.3.3


void plot_Coverage(std::string filename)
{
  // First of all one has to retrieve all the data
  // The content of the tree is described here:
  // https://github.com/sviret/DeepWave/blob/OG_lab/MFLight/bankgen.cxx#L252
    
  TChain *Bank = new TChain("Bank");
  Bank->Add(filename.c_str());

  double t_i;
  double t_f;
  double m_a;
  double m_b;
    
  Bank->SetBranchAddress("mass1",&m_a);
  Bank->SetBranchAddress("mass2",&m_b);
  Bank->SetBranchAddress("t_i",&t_i);
  Bank->SetBranchAddress("t_f",&t_f);
    
  // Get the number of templates
  int Nt = Bank->GetEntries();
  std::cout << "The bank contains " << Nt << " templates" << std::endl;
    
  // Do a first loop over the templates to get the plot range
  
  double m_c_max=0;
  double t_u_max=0;
  
  double m_c,t_u;
    
  for (int i=0;i<Nt;++i)
  {
      Bank->GetEntry(i);
  
      // Compute t_util and the chirp mass
      
      t_u=t_f-t_i;
      m_c=(m_a+m_b)*pow(m_a*m_b/pow(m_a+m_b,2.),3./5.);

      if (t_u>t_u_max) t_u_max=t_u;
      if (m_c>m_c_max) m_c_max=m_c;
  }

  std::cout << "Longest signal " << t_u_max << " seconds" << std::endl;
  std::cout << "Highest chirp mass " << m_c_max << " solar masses" << std::endl;
    
  // We can create the plot and fill it
  TH2F *t_vs_c = new TH2F("","", 100,0,t_u_max, 100,0.,m_c_max);

  for (int i=0;i<Nt;++i)
  {
     Bank->GetEntry(i);
    
     // Compute t_util and the chirp mass
        
     t_u=t_f-t_i;
     m_c=(m_a+m_b)*pow(m_a*m_b/pow(m_a+m_b,2.),3./5.);

     t_vs_c->Fill(t_u,m_c);
  }
        
    std::cout << "Do some plots" << std::endl;
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    
    TCanvas *c1 = new TCanvas("c1","Template distribution",451,208,1208,604);
    c1->SetFillColor(0);
    c1->SetGridx();
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.08);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);

    t_vs_c->GetXaxis()->SetLabelSize(0.03);
    t_vs_c->GetXaxis()->SetTitle("Time in detector sensitivity (in s)");
    t_vs_c->GetYaxis()->SetTitle("Chirp mass (in solar masses)");
    t_vs_c->SetMarkerStyle(20);
    t_vs_c->Draw();
    
    c1->Update();
    //c1->SaveAs("Chirp.png");
}



//
// Plot the template Fourier transform amplitude s^tilde(f) as a function of f
// For the template of mass m_1 and m_2
// Exo 2.3.4

void plot_Spectrum(std::string filename, int m_1, int m_2)
{
  // First of all one has to retrieve all the data
    
  TChain *Bank = new TChain("Bank");
  Bank->Add(filename.c_str());

  double m_a;
  double m_b;
    
  int m_a_ref = std::min(m_1,m_2);
  int m_b_ref = std::max(m_1,m_2);
      
  Bank->SetBranchAddress("mass1",&m_a);
  Bank->SetBranchAddress("mass2",&m_b);
    
  std::vector<double>  *Hfr = new std::vector<double>;
  std::vector<double>  *Hfi = new std::vector<double>;
  double f_init;
  double f_bin;
  double SNRmax;
    
  Bank->SetBranchAddress("Hfr",&Hfr);
  Bank->SetBranchAddress("Hfi",&Hfi);
  Bank->SetBranchAddress("f_init",&f_init);
  Bank->SetBranchAddress("f_bin",&f_bin);
  Bank->SetBranchAddress("SNRmax",&SNRmax);
    
  // First check if the signal is in the bank
  int idx=-1;
  // Get the number of templates
  int Nt = Bank->GetEntries();
  
  for (int i=0;i<Nt;++i)
  {
     Bank->GetEntry(i);
    
     // Compute t_util and the chirp mass
     if (m_a==m_a_ref && m_b==m_b_ref)
     {
        idx=i;
        break;
     }
  }
    
  if (idx==-1) return; // Not in the bank
    
  Bank->GetEntry(idx);
    
  int length=static_cast<int>(Hfi->size());
  std::cout << "The signal contains " << length << " data points" << std::endl;
    
  double f_max=f_init+length*f_bin;
    
  std::cout << "Signal is comprised between " << f_init << " and " << f_max << "  Hz " << std::endl;
  
    
  std::vector<double>  *FFTAmp = new std::vector<double>;
  std::vector<double>  *Freq = new std::vector<double>;
  FFTAmp->clear();
  Freq->clear();
    
  for (int i=0;i<length;++i)
  {
    Freq->push_back(f_init+i*f_bin);
    FFTAmp->push_back(1/SNRmax*sqrt(Hfr->at(i)*Hfr->at(i)+Hfi->at(i)*Hfi->at(i)));
  }
    
    
  // Now we create a graph to plot the signal:
  TGraph* fftAmp = new TGraph(length, &Freq->at(0), &FFTAmp->at(0));
    
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

  fftAmp->GetXaxis()->SetLabelSize(0.03);
  fftAmp->GetXaxis()->SetTitle("Frequency (in Hz)");
  fftAmp->GetYaxis()->SetTitle("Amplitude (a.u.)");
  fftAmp->SetMarkerStyle(20);
  fftAmp->SetMarkerSize(0.2);
  fftAmp->Draw("AL");

  c1->Update();
    //c1->SaveAs("RawSpectrum.png");
    
}
