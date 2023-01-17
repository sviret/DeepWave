#ifndef TESTFFT_H
#define TESTFFT_H


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
#include <fftw3.h>

using namespace std;

///////////////////////////////////
//
//
// Base class for test signal generation
// Header file
//
//
///////////////////////////////////



class testfft
{
 public:

    
  testfft(double freq, double fs, double length, std::string outfile);

  void  create_function();  // The main method
  void  reset();
  void  initTuple();
    
 private:
    
    // Parameters definition
    
    
    double m_fs;     // Sampling frequency in kHz (so the max freq one can obtain with FFT will freq/2)
    double m_freq;  // Signal frequency (in Hz)
    double m_length;// Signal length (in s)

    double t_init;  // Starting time of the sampled signal (-30sec)
    double t_bin;   // Time bin width (60/f_s)
    double f_init;  // Starting frequency after FFT (0Hz)
    double f_bin;   // Frequency bin width
    
    // Here are the data vectors which are store in the
    // rootuple
    
    std::vector<double> *Signal;   // The original signal
    std::vector<double> *T;        // Time coordinates
    std::vector<double> *Tf;       // Frequency coordinates
    std::vector<double> *Hfr;      // Real part of H FFT
    std::vector<double> *Hfi;      // Imaginary part of H FFT
    std::vector<double> *Signal_r; // The signal retrieved after inverse FFT
    

    std::string m_outf;            // Output file name
    
    TTree *FFTtest;                // Output ROOT tree
    TFile *m_outfile;              // Output ROOT file
    
};

#endif

