///////////////////////////////////
//
//  A small class to test in simple conditions the FFTW behaviour
//  Particularly important to understand how FFT are normalized
//
//  Basically produce a sinusoid over a given frequency, FFT it, and then FFT it back
//
//  Authors: S.Viret
//  Date   : 12/01/2023
//
//  Input parameters:
//
//  freq     : the frequency of the sinusoid (in Hz)
//  fs       : the sampling frequency of the signal we will produce (in Hz)
//  length   : signal duration (in s)
//  outfile  : name of the rootfile containing the final data
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "testfft.h"


// Main constructor

testfft::testfft(double freq, double fs, double length, std::string outfile):
Signal(new std::vector<double>),
T(new std::vector<double>),
Tf(new std::vector<double>),
Hfr(new std::vector<double>),
Hfi(new std::vector<double>),
Signal_r(new std::vector<double>)
{
  srand(time(NULL));
    
  testfft::reset();

  // Initialize some parameters (definition in the header file)
    
  m_fs=fs;
  m_freq=freq;
  m_length=length;
  
  t_init=0;
  t_bin=1/m_fs;
  f_init=0;
  f_bin=(t_bin/m_length)*m_fs;

  m_outf=outfile;
  testfft::initTuple(); // Create the output ROOT file
    
  testfft::create_function(); // Create the signal
     
   // Write the output root file
   m_outfile->Write();
   m_outfile->Close();

}


//////////////////////////////////////////////
//
// Tested signal generation
// 
//////////////////////////////////////////////
void testfft::create_function()
{
    // Number of sampling points over measurement time
    int  n=static_cast<int>(m_length*m_fs);
    
    //Allocate an array big enough to hold the transform output
    //Transform output in 1d contains, for a transform of size N,
    //N/2+1 complex numbers, i.e. 2*(N/2+1) real numbers
    //our transform is of size n+1, because the histogram has n+1 bins
    
    double re,im;
    double *in = new double[2*((n+1)/2+1)];
    int n_size = n+1;
    Signal->clear();
    Signal_r->clear();
    T->clear();
    Tf->clear();
    Hfr->clear();
    Hfi->clear();
    
    // Time
    for(double t=t_init ; t<t_init+m_length ; t=t+t_bin) T->push_back(t);
    
    // The FFT is handled via the FFTW algorithm
    //
    // https://www.fftw.org
    //
    // The I/O coefficients of the fftw object will be
    // stored in 2 vectors of complex coordinates, input and output
    // The size is fixed and correspond to the maximal FFT size you
    // will perform.
    
    fftw_complex input[n_size];
    fftw_complex output[n_size];
    fftw_plan p;
    
    
 
    cout << "--> Signal produced:" << endl;
    cout << endl;
    cout << "Sinusoid with frequency = " << m_freq << " Hz" <<endl;
    
    int i = 0;
    double sig;
    // A loop to produce the full signal between t_init and t_init+duration (so 60 sec)
    // We also fill the noise vector
    
    
    for(double t=t_init ; t<t_init+m_length ; t=t+t_bin)
    {
        sig=sin(8*atan(1.)*m_freq*t);

        Signal->push_back(sig); // Pure signal

        ++i;
        input[i][0]=sig; // Signal coordinates for the TF
        input[i][1]=0.;
        output[i][0]=0;
        output[i][1]=0;
    }
    
    // Now perform a Fourier transform of the signal, note that the size can
    // be lower than n_size (input and output being filled up accordingly), but never larger
    
    p = fftw_plan_dft_1d(n_size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    double frequency;
    
    for (Int_t i=0; i<=T->size(); i++)
    {
        re = output[i][0]/sqrt(T->size()); // Don't forget the normalisation
        im = output[i][1]/sqrt(T->size());
        
        // FFTW by default does not apply any normalisation.
        // To get the physically relevant values, we need to express the results
        // in an orthonormal basis. In this case a scale factor of 1/sqrt(n) should be
        // applied, as explained here:
        //
        // https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT
        //
        
        
        frequency=float(i)/n_size*m_fs;
        if  (i>T->size()/2) frequency-=m_fs;
        
        Tf->push_back(frequency);
        Hfr->push_back(re);
        Hfi->push_back(im);
        
        input[i][0] = re;
        input[i][1] = im;
 
    }
    
    // Do the inverse TF to retrieve the original signal
        
    p = fftw_plan_dft_1d(n_size, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    i=0;
    
    for(double t=t_init ; t<t_init+m_length ; t=t+t_bin)
    {
        ++i;
        
        in[i]= output[i][0]/sqrt(T->size());
        Signal_r->push_back(in[i]); // The total signal
     }
    
    fftw_destroy_plan(p);
    
    // Fill the ROOT tree
    FFTtest->Fill();
    fftw_cleanup();
}

void testfft::reset()
{
  m_freq=0;
  m_fs=0;
  m_length=0;
    
  Signal->clear();
  Signal_r->clear();
  T->clear();
  Tf->clear();
  Hfr->clear();
  Hfi->clear();

  t_init=0;
  t_bin=0;
  f_init=0;
  f_bin=0;
}

//
// ROOTuple creation
//

void testfft::initTuple()
{
    m_outfile  = new TFile(m_outf.c_str(),"recreate");

    FFTtest  = new TTree("FFT","");

    FFTtest->Branch("Signal",&Signal);
    FFTtest->Branch("Signal_r",&Signal_r);
    FFTtest->Branch("T",&T);
    FFTtest->Branch("F",&Tf);
    FFTtest->Branch("Hfr",&Hfr);
    FFTtest->Branch("Hfi",&Hfi);
    FFTtest->Branch("t_init",&t_init);
    FFTtest->Branch("t_bin",&t_bin);
    FFTtest->Branch("f_init",&f_init);
    FFTtest->Branch("f_bin",&f_bin);
}


