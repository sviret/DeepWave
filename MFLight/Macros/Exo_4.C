///////////////////////////////////
//
//  Exercise 5: display with the signal to noise ratio
//
//  Goal of the exercise:
//
//  Opening the root file created with og_ana code and
//  display some bank properties
//
//  This macro is the answer of question 2.4.3 of the following page:
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
#include "TColor.h"
#include "TGraph2D.h"


//
// Plot rho(t) for template Ma=m1 / Mb=m2
//


void plot_Rho(std::string filename, int m1, int m2)
{
  // First of all one has to retrieve all the data
    
  TChain *Res = new TChain("Filtreinfo");
  Res->Add(filename.c_str());

  std::vector<double> *rho = new std::vector<double>;
  std::vector<double> *temp= new std::vector<double>;
  double m_a;
  double m_b;
    
  Res->SetBranchAddress("c_mass1",&m_a);
  Res->SetBranchAddress("c_mass2",&m_b);
  Res->SetBranchAddress("Hfin",&rho);
  Res->SetBranchAddress("Tfin",&temp);
    
  // Get the number of comparisons
  int Nt = Res->GetEntries();
  std::cout << "The bank contains " << Nt << " filtered signals" << std::endl;
  TGraph* t_vs_c;

  for (int i=0;i<Nt;++i)
  {
     Res->GetEntry(i);
     if (m_a!=m1 || m_b!=m2) continue;
      
     // Fill the histogram
     int length = static_cast<int>(rho->size());
      
     t_vs_c = new TGraph(length, &temp->at(0), &rho->at(0));
  }
        
  std::cout << "Do some plots" << std::endl;
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
    
  TCanvas *c1 = new TCanvas("c1","S/N distribution",451,208,1208,604);
  c1->SetFillColor(0);
  c1->SetGridx();
  c1->SetGridy();
  c1->SetBorderSize(2);
  c1->SetLeftMargin(0.08);
  c1->SetFrameBorderMode(0);
  c1->SetFrameBorderMode(0);

  t_vs_c->GetXaxis()->SetLabelSize(0.03);
  t_vs_c->GetXaxis()->SetTitle("Time (in s)");
  t_vs_c->GetYaxis()->SetTitle("Signal to noise ratio");
  t_vs_c->SetMarkerStyle(20);
  t_vs_c->Draw("AL");

  c1->Update();
    //c1->SaveAs("Chirp.png");
}


//
// Get all the signals which are passing the S/N threshold given by cut
//

void get_signals(std::string filename, double cut)
{
  // First of all one has to retrieve all the data
    
  TChain *Res = new TChain("Filtreinfo");
  Res->Add(filename.c_str());

  std::vector<double> *rho = new std::vector<double>;
  std::vector<double> *temp= new std::vector<double>;
  double m_a;
  double m_b;
    
  Res->SetBranchAddress("c_mass1",&m_a);
  Res->SetBranchAddress("c_mass2",&m_b);
  Res->SetBranchAddress("Hfin",&rho);
  Res->SetBranchAddress("Tfin",&temp);
    
  // Get the number of comparisons
  int Nt = Res->GetEntries();
  std::cout << "The bank contains " << Nt << " filtered signals" << std::endl;

  
    
    
  for (int i=0;i<Nt;++i)
  {
     Res->GetEntry(i);

     int length = static_cast<int>(rho->size());
     for (int j=0;j<length;++j)
     {
         if (sqrt(rho->at(j)*rho->at(j))>cut)
         {
         std::cout << "Pass the cut for " << m_a << " / " << m_b <<
                " = " << sqrt(rho->at(j)*rho->at(j)) << " - t_c = " <<
                temp->at(j) << std::endl;
         }
     }
  }
}


//
// Plot a map showing the max SNR measured for all templates elements
//

void get_peaks(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *Res = new TChain("Filtreinfo");
  Res->Add(filename.c_str());

  double m_a;
  double m_b;
  double m_c;
  double max;
  TGraph2D *Chirp = new TGraph2D();
  Chirp->SetTitle("Maximum SNR measured per template");
    
  TGraph2D *Chirp2 = new TGraph2D();
  Chirp2->SetTitle("Chirp mass of templates");
  Res->SetBranchAddress("c_mass1",&m_a);
  Res->SetBranchAddress("c_mass2",&m_b);
  Res->SetBranchAddress("m_chirp",&m_c);
  Res->SetBranchAddress("peakrho",&max);
    
  // Get the number of comparisons
  int Nt = Res->GetEntries();
  std::cout << "The bank contains " << Nt << " filtered signals" << std::endl;
  int npts=0;
  for (int i=0;i<Nt;++i)
  {
     Res->GetEntry(i);
     if (m_a==m_b)
     {
      Chirp->SetPoint(npts,m_a,m_b,max);
      Chirp2->SetPoint(npts,m_a,m_b,m_c);
      ++npts;
     }
     else
     {
      Chirp->SetPoint(npts,m_a,m_b,max);
      Chirp2->SetPoint(npts,m_a,m_b,m_c);
      ++npts;
      Chirp->SetPoint(npts,m_b,m_a,max);
      Chirp2->SetPoint(npts,m_b,m_a,m_c);
      ++npts;
     }
  }
    
  TH2D* h = new TH2D();
  h = Chirp->GetHistogram();
  TH2D* h2 = new TH2D();
  h2 = Chirp2->GetHistogram();
    
  TCanvas *c1 = new TCanvas("c1","Peaks",200,200,600,600);
  c1->SetFillColor(0);
  c1->SetGridx();
  c1->SetGridy();
  c1->SetBorderSize(2);
  c1->SetLeftMargin(0.08);
  c1->SetFrameBorderMode(0);
  c1->SetFrameBorderMode(0);

  //t_vs_c->GetXaxis()->SetLabelSize(0.03);
  h->GetXaxis()->SetTitle("Mass 1 (in solar masses)");
  h->GetYaxis()->SetTitle("Mass 2 (in solar masses)");
  //t_vs_c->SetMarkerStyle(20);
  h->SetContour(99);
  gStyle->SetPalette(kRainBow);
  h->Draw("cont4z");

  c1->Update();
    
    TCanvas *c2 = new TCanvas("c2","Peaks",200,200,600,600);
    c2->SetFillColor(0);
    c2->SetGridx();
    c2->SetGridy();
    c2->SetBorderSize(2);
    c2->SetLeftMargin(0.08);
    c2->SetFrameBorderMode(0);
    c2->SetFrameBorderMode(0);

    //t_vs_c->GetXaxis()->SetLabelSize(0.03);
    h2->GetXaxis()->SetTitle("Mass 1 (in solar masses)");
    h2->GetYaxis()->SetTitle("Mass 2 (in solar masses)");
    //t_vs_c->SetMarkerStyle(20);
    h2->SetContour(99);
    gStyle->SetPalette(kRainBow);
    h2->Draw("cont4z");

    c2->Update();
}



//
// Plot rho_vs_t for all data
//


void plot_Rho_t(std::string filename, double dt)
{
  // First of all one has to retrieve all the data
    
  TChain *Res = new TChain("Filtreinfo");
  Res->Add(filename.c_str());

  std::vector<double> *rho = new std::vector<double>;
  std::vector<double> *temp= new std::vector<double>;
  double m_c;
    
  TGraph2D *Chirp = new TGraph2D();
  Chirp->SetTitle("SNR vs time");
  
  Res->SetBranchAddress("m_chirp",&m_c);
  Res->SetBranchAddress("Hfin",&rho);
  Res->SetBranchAddress("Tfin",&temp);
    
  // Get the number of comparisons
  int Nt = Res->GetEntries();
  std::cout << "The bank contains " << Nt << " filtered signals" << std::endl;
  int npts=0;
  int length;
  double time;
    
  for (int i=0;i<Nt;++i)
  {
     Res->GetEntry(i);
     // Fill the histogram
     length = static_cast<int>(rho->size());
     int start=0;
     int nelem=0;
     double val=0;
      
     for (int j=0;j<length;++j)
     {
         time=temp->at(j);
         if (int(time/dt)==start)
         {
             nelem++;
             val+=rho->at(j);
         }
         else
         {
             Chirp->SetPoint(npts,time,m_c,val/nelem);
             npts++;
             start++;
             val=0;
             nelem=0;
         }
     }
  }
        
  std::cout << "Do some plots" << std::endl;
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
    
  TH2D* h = new TH2D();
  h = Chirp->GetHistogram();
        
  TCanvas *c1 = new TCanvas("c1","Peaks",200,200,600,600);
    c1->SetFillColor(0);
    c1->SetGridx();
    c1->SetGridy();
    c1->SetBorderSize(2);
    c1->SetLeftMargin(0.08);
    c1->SetFrameBorderMode(0);
    c1->SetFrameBorderMode(0);

    //t_vs_c->GetXaxis()->SetLabelSize(0.03);
    h->GetXaxis()->SetTitle("Time (in s)");
    h->GetYaxis()->SetTitle("Mass chirp (in solar masses)");
    //t_vs_c->SetMarkerStyle(20);
    h->SetContour(99);
    gStyle->SetPalette(kRainBow);
    h->Draw("cont4z");

    c1->Update();
        
}
