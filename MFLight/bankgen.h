#ifndef BANKGEN_H
#define BANKGEN_H


#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>

#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TVirtualFFT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"

#include "chirp.h"

#include <fstream>
#include <string>

using namespace std;

///////////////////////////////////
//
//
// Base class for signal bank generation
//
// Only mass ranges are used
// Incidence angle is fixed at 45°
// Dist is fixed at 1 Mpsec
//
// Generate 1 signal for each mass pair between mass_min and mass_max
//
//
//  Author: S.Viret
//  Date:   19/01/22
//
///////////////////////////////////



class bankgen
{
 public:

    
  bankgen(double mass_min,double mass_max,std::string outfile);

  void  create_bank();  // The main method
  void  reset();
  void  initTuple();
    
 private:
    
  double f_s; // Sampling frequency in kHz (so the max freq one can obtain with FFT will freq/2)

  double m_mass1;
  double m_mass2;
    
  double mass1,mass2,t_i,t_f;
  double t_init, t_bin;
  double f_init, f_bin;
    
  double tchirp;
  double m_theta;
  double m_dist;
    
  std::vector<double> *T;
  std::vector<double> *H;
  std::vector<double> *Tf;
  std::vector<double> *Hfr;
  std::vector<double> *Hfi;

  std::string m_outf;
    
  TTree *bankparams;      // The trees
  TFile *m_outfile;       // The output file
    
};

#endif

