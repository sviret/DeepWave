///////////////////////////////////
//
//  Exercise 3: display with the signal to noise ratio
//
//  Goal of the exercise:
//
//  Opening the root file created with og_ana code and
//  display some bank properties
//
//  This macro is the answer of question 2.2.6 of the following page:
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
  std::cout << "The bank contains " << Nt << " filteres signals" << std::endl;
        
  // We can create the plot and fill it
  TH2F *t_vs_c = new TH2F("","", 10000,-30,30, 100,0.,40.);

  for (int i=0;i<Nt;++i)
  {
     Res->GetEntry(i);
     if (m_a!=m1 || m_b!=m2) continue;
      
     // Fill the histogram
      std::cout << m_a << " / " << m_b << std::endl;
     int length = static_cast<int>(rho->size());
      std::cout << length << std::endl;
     for (int j=0;j<length;++j)
     {
         t_vs_c->Fill(temp->at(j),rho->at(j));
     }
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
    t_vs_c->Draw();
    
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
