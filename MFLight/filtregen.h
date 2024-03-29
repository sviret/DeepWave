#ifndef filtreGEN_H
#define filtreGEN_H


#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

#include "TMath.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"

#include <fstream>
#include <fftw3.h>

using namespace std;


class filtregen
{
 public:

  filtregen(std::string bank, std::string signal, std::string outfile);

  void  do_MF();  // The main method
  void  reset();
  void  initTuple();
    
 private:

  double c_mass2;
  double c_mass1;
  double c_mc;
  double maxrho;
  std::vector<double> *Htfr;
  std::vector<double> *Htfi;
  std::vector<double> *Tfin;
  std::vector<double> *Hfin;
  std::vector<double> *Hfinr;
  std::vector<double> *Hfini;
  double t_init, t_bin;
  double f_init, f_bin;
    
  std::string m_outs;
  std::string m_outb;
  std::string m_outf;
    
  TTree *Filtreparams;      // The trees
  TFile *m_outfile;       // The output file
    
  fftw_complex *input;
  fftw_complex *output;
  fftw_plan p;
};

#endif

