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
#include <fftw3.h>

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

    
  chirpgen(double mass1,double mass2,double snr, double noise, std::string outfile);

  void  create_function();  // The main method
  void  reset();
  void  initTuple();
    
 private:
    
    // Parameters definition
    
    
    double f_s;     // Sampling frequency in kHz (so the max freq one can obtain with FFT will freq/2)

    double time_lim;// This is the maximum sample length (we limit for the FFT)
    
    double m_mass1; // Mass of the first object (in solar masses units)
    double m_mass2; // Mass of the second object (in solar masses units)
    double tchirp;  // Coalescence time is randomly chosen between 0 and 30 sec
    double m_sigma; // Gaussian noise width (in 10-23 units)
    double m_psd;   // Power spectral density 
    double m_snr;   // Signal to noise ratio

    double t_init;  // Starting time of the sampled signal
    double t_bin;   // Time bin width
    double f_init;  // Starting frequency after FFT (0Hz)
    double f_bin;   // Frequency bin width
    
    // Here are the data vectors which are store in the
    // rootuple
    
    std::vector<double> *Signal;   // The original chirp signal produced (w/o noise, tchirp=0)
    std::vector<double> *Signaln;  // The original chirp signal whitened and normalised
    std::vector<double> *T;        // Time coordinates
    std::vector<double> *H;        // Noisy signal with Chirp moved by tchirp (tested signal)
    std::vector<double> *N;        // Noise alone
    std::vector<double> *Tf;       // Frequency coordinates
    std::vector<double> *Hfr;      // Real part of H FFT (H=SNR*S+N)
    std::vector<double> *Hfi;      // Imaginary part of H FFT
    std::vector<double> *Nfr;      // Real part of Noise FFT
    std::vector<double> *Nfi;      // Imaginary part of Noise FFT
    std::vector<double> *Sfr;      // Real part of raw Signal FFT
    std::vector<double> *Sfi;      // Imaginary part of raw Signal FFT
    std::vector<double> *Snfr;     // Real part of whitened Signal FFT
    std::vector<double> *Snfi;     // Imaginary part of whitened Signal FFT
    
    std::string m_outf;            // Output file name
    
    TTree *Chirparams;             // Output ROOT tree
    TFile *m_outfile;              // Output ROOT file
    
};

#endif

