#ifndef CHIRPGEN_H
#define CHIRPGEN_H


#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TVirtualFFT.h"

#include "chirp.h"

using namespace std;

///////////////////////////////////
//
//
// Base class for test signal generation
// Header file
//
//
///////////////////////////////////



class chirpgen
{
 public:

    
  chirpgen(double mass1,double mass2,double theta, double dist, double noise, std::string outfile);

  void  create_function();  // The main method
  void  reset();
  void  initTuple();
    
 private:
    
    // Parameters definition
    
    
    double f_s;     // Sampling frequency in kHz (so the max freq one can obtain with FFT will freq/2)

    double m_mass1; // Mass of the first object (in solar masses units)
    double m_mass2; // Mass of the second object (in solar masses units)
    double tchirp;  // Coalescence time is randomly chosen between 0 and 30 sec
    double m_theta; // Incidence angle (in °)
    double m_sigma; // Gaussian noise width (in 10-21 units)
    double m_dist;  // Distance to the source (in Mpsec)

    double t_init;  // Starting time of the sampled signal (-30sec)
    double t_bin;   // Time bin width (60/f_s)
    double f_init;  // Starting frequency after FFT (0Hz)
    double f_bin;   // Frequency bin width
    
    // Here are the data vectors which are store in the
    // rootuple
    
    std::vector<double> *Signal;   // The original chirp signal produced (w/o noise, tchirp=0)
    std::vector<double> *T;        // Time coordinates
    std::vector<double> *H;        // Noisy signal with Chirp moved by tchirp (tested signal)
    std::vector<double> *N;        // Noise alone
    std::vector<double> *Tf;       // Frequency coordinates
    std::vector<double> *Hfr;      // Real part of H FFT
    std::vector<double> *Hfi;      // Imaginary part of H FFT
    std::vector<double> *Nfr;      // Real part of Noise FFT
    std::vector<double> *Nfi;      // Imaginary part of Noise FFT

    std::string m_outf;            // Output file name
    
    TTree *Chirparams;             // Output ROOT tree
    TFile *m_outfile;              // Output ROOT file
    
};

#endif

