///////////////////////////////////
//
// GW signal generation and noising
//
//  Authors: S.Viret, based on initial code from B.Branchu and E.Pont
//  Date   : 21/01/2022
//  Major Rev. : 13/01/2023
//
//  Input parameters:
//
//  mass1/mass2 : mass of the 2 objects (in solar masses)
//  snr         : the signal to noise ratio of the chirp to be produced
//  noise       : sigma of the gaussian noise to be added to the strain (in 10-23 units)
//                (only flat noise for the moment)
//  outfile     : name of the rootfile containing the final data
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Angular dependences are neglected.
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "chirpgen.h"

// Main constructor

chirpgen::chirpgen(double mass1,double mass2,double snr,double noise,std::string outfile):
Signal(new std::vector<double>),
Signaln(new std::vector<double>),
T(new std::vector<double>),
H(new std::vector<double>),
N(new std::vector<double>),
Tf(new std::vector<double>),
Hfr(new std::vector<double>),
Hfi(new std::vector<double>),
Nfr(new std::vector<double>),
Nfi(new std::vector<double>),
Sfr(new std::vector<double>),
Sfi(new std::vector<double>),
Snfr(new std::vector<double>),
Snfi(new std::vector<double>)
{
  srand(time(NULL));
    
  chirpgen::reset();

  // Initialize some parameters (definition in the header file)
    
  f_s=2.048;      // Sampling rate in kHz
  time_lim=30.;   // Maximum sample length in seconds
    
  m_sigma=noise*1e-23; // Noise value is rescaled to strain unit
  m_outf=outfile;
  m_mass1=mass1;
  m_mass2=mass2;
  m_snr=snr;

  chirpgen::initTuple(); // Create the output ROOT file
    
  chirpgen::create_function(); // Create the signal
     
   // Write the output root file
   m_outfile->Write();
   m_outfile->Close();
    

}


//////////////////////////////////////////////
//
// Tested signal generation
// 
//////////////////////////////////////////////
void chirpgen::create_function()
{
    // First one defines some initial parameters

    // Detector acceptance is important as we will use it to compute
    // normalisation factor for the filter (so called SNRmax)
    
    double fi = 15;        // Interferometer low sensitivity (in Hz)
    double ff = 1000;      // Interferometer high sensitivity (in Hz)
    
    // Then we create a chirp object via the chirp class (see corresponding doc)
    // with the corresponding mass properties
    // We use a distance of 1Mpsec, the signal will be scaled afterwards
    
    chirp *mychirp=new chirp();
    mychirp->init(m_mass1,m_mass2,1);

    // Define the time range when the signal frequency will
    // be within the detector sensitivity
    // Divide by 2 because binary orbital rotation frequency is
    // half the frequency of the wave
    // We will define the chirp signal within this range only
    
    double ti   = mychirp->get_time(0.8*fi/2); // We take some margin for the low
                                               // freq (for potential tapering)
    double tlow = mychirp->get_time(fi/2);     // Enters the acceptance
    double tf   = mychirp->get_time(ff/2);     // Exits the acceptance
    
    // Note that tapering at the end is also possible, using eg f_ISCO as f_max
    //
    

    double duration = time_lim;  // Signal duration, in seconds
    
    double timeindet = tf-tlow; // How long is the signal in detector acceptance
    
    
    //  Use it to get the number of sampling points over measurement time
    int n_size=static_cast<int>(duration*f_s*1000)+1;
    
    t_init=0;
    t_bin=(1/(f_s*1000));
    f_init=0;
    f_bin=(t_bin/duration)*1000*f_s;
    
    if (m_mass1>0 && m_mass2>0)
    {
        cout << endl;
        cout << "--> Signal produced:" << endl;
        cout << endl;
        cout << "m1 = " << m_mass1 << " solar masses" <<endl;
        cout << "m2 = " << m_mass2 << " solar masses" <<endl;
        cout << "Dist bet m1 and m2 is 10r_s at t0 = " << mychirp->get_t0() << " s " <<endl;
        cout << "At ti = " << ti << " s the wave frequency is " << 0.8*fi << " Hz" <<endl;
        cout << "At tf = " << tf << " s the wave frequency is " << ff << " Hz" <<endl;
        cout << "Coalescence at tc = 0 s " <<endl;
        cout << "Sample length to be produced: " << duration << " seconds" <<endl;
        cout << "Among which: " << timeindet << " will be in the detector acceptance" <<endl;
    }
    else
    {
        cout << endl;
        cout << "--> Just producing some noise" << endl;
        cout << endl;
    }
    
    //Allocate an array big enough to hold the FFT transform
    //Transform output in 1d contains, for a transform of size N,
    //N/2+1 complex numbers, i.e. 2*(N/2+1) real numbers
    //our transform is of size n+1, because the histogram has n+1 bins
    
    double re,im;
    double *in    = new double[2*(n_size/2+1)];
    double *noise = new double[2*(n_size/2+1)];

    // Define a vector containing the time coordinate
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin) T->push_back(t);

    // Here we define the PSD value at a given frequency bin. As we are considering
    // a flat gaussian noise this value is constant
    // The n_size factor is due to the FFT. As it is not normalized by default we should
    // provide unnormalized coefficients in the frequency domain.

    m_psd=2*m_sigma*m_sigma*n_size;
    
    /*
    Create the noise in freq domain from the PSD

    We start from a PSD which provides us the power of the noise at a given frequency
    In order to create a noise realisation, we need first to generate a random realisation
    in the frequency domain

    For each frequency we choose a random value of the power centered on the sqrt(PSD/2) value, we consider
    that power distribution is gaussian with a width equal to power/4 (rule of thumb, could be improved)
    */
    
    std::normal_distribution<double> distribution(sqrt(m_psd/2),sqrt(m_psd/2)/4);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    
    // To get a random value x based on normal distribution, you just have to do:
    // x = distribution(generator)
  
    // The FFT is handled via the FFTW algorithm
    //
    // https://www.fftw.org
    //
 
    fftw_complex input[n_size];
    fftw_complex output[n_size];
    fftw_plan p;
        
    // Loop to create the noise realisation in frequency domain
    //
    
    for (int i = 0; i<T->size()/2; i++)
    {
      // Reminder: the normalization factor is already in the PSD
      /*
       Exercise 2
       
       You have to complete this part
       
       */
        
        
    }
    
    // We don't enter the negative frequency coefficients (single-sided FFT), so
    // the inverse FFT result will have to be multiplied by 2
    
    for (int i = T->size()/2; i<T->size(); i++)
    {
        input[i][0] = 0;
        input[i][1] = 0.;
    }
        
    // Then we perform the inverse fourier transform in order to retrieve the
    // noise realisation
    // We should not forget the norm factor (as usual...)

    p = fftw_plan_dft_1d(n_size, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    int i=0;
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin)
    {
        noise[i]= 2*output[i][0]/sqrt(n_size); // Noise
        N->push_back(noise[i]);
        ++i;
    }
    
    fftw_destroy_plan(p);
    
    // If just asking noise there is no need to go further
    if (m_mass1==0 || m_mass2==0)
    {
        // Fill the ROOT tree
        Chirparams -> Fill();
        fftw_cleanup();
        return;
    }
    
    // The object has tchirp=duration by construction, we will move it a bit to test the
    // efficiency, between 0.75*duration and duration.
    
    tchirp = duration-(rand()%(static_cast<int>(1000*duration/4.)))/1000.;

 
    cout << "--> Signal produced:" << endl;
    cout << "Moved the coalescence at tc = " << tchirp << " s " <<endl;
    
    
    i = 0;

    // We artificially rescale the signal in order to avoid
    // rounding issues (!!this is not the whitening!!)
    double scaling=1e-21;
    int idx_chirp=0;
    // We will start with the signal FFT
    
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin)
    {
        if (t<=tchirp) idx_chirp=i;
        
        Signal->push_back(mychirp->get_h(t-duration)); // Pure signal

        // Note that the signal is renormalized in order to avoid pb with the FFT
        input[i][0]=mychirp->get_h(t-duration)/scaling; // Signal coordinates for the TF
        input[i][1]=0.;
        output[i][0]=0;
        output[i][1]=0;
        ++i;
    }
    idx_chirp=i-idx_chirp;
    
    // The FFT
    p = fftw_plan_dft_1d(n_size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    // We have the signal FFT in the frequency domain
    // We will whiten it as we did for the noise
    // We will also compute the SNR max value, in order
    // to normalize the temporal signal to SNR=1
    
    double SNRmax=0.;

    for (int i=0; i<=T->size()/2-1; i++)
    {
        // Non-whitened FFT
        re = output[i][0]*scaling/sqrt(T->size());
        im = output[i][1]*scaling/sqrt(T->size());

        Sfr->push_back(re);
        Sfi->push_back(im);
        
        // Whitened
        re /= sqrt(m_psd/2);
        im /= sqrt(m_psd/2);
        
        Snfr->push_back(re);
        Snfi->push_back(im);
    
        // Fill this as input for the inverse FFT
        // Compute the whitened signal in temporal domain
        
        input[i][0] = re;
        input[i][1] = im;
        
        if (i*f_bin<15) continue;
        if (i*f_bin>1000) continue;

        // Scaling factor to get the correct SNRmax
        // in the detector freq domain
        SNRmax+=(re*re+im*im);
    }
    
    for (int i = T->size()/2; i<T->size(); i++)
    {
        input[i][0] = 0;
        input[i][1] = 0.;
    }
    
    SNRmax=sqrt(2*SNRmax); // Final norm factor
    std::cout<< "SNRmax compute in the frequency range [15,1000]: " << SNRmax << std::endl;
    std::cout<< "Whitened signal with this value will have an SNR of 1 " << std::endl;
    
    // Do the inverse TF of the whitened signal
        
    p = fftw_plan_dft_1d(n_size, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    i=0;
    double pSNR=0;
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin)
    {
        in[i]= 2*(output[i][0])/(SNRmax*sqrt(T->size()));; // Normed signal
        Signaln->push_back(in[i]); // The total signal
        if (fabs(in[i])>pSNR) pSNR=fabs(in[i]);
        ++i;
     }
    
    std::cout<<"Peak amplitude (pSNR) of the normalized signal (SNR=1) is " << pSNR << std::endl;
    std::cout<<"The ratio between SNR and pSNR is thus equal to " << 1/pSNR << std::endl;
    std::cout<<"We will produce a signal with an SNR of " << m_snr << std::endl;
    
    i=0;
    
    // Finally can produce signal+noise, at the required SNR.
    
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin)
    {
        if (t<=tchirp)
        {
            in[i]= m_snr*in[i+idx_chirp]+noise[i];
        }
        else
        {
            in[i]= noise[i];
        }

        input[i][0]=in[i];
        input[i][1]=0.;
        
        H->push_back(in[i]); // The total signal
        
        ++i;
    }
    
    // Now perform a final Fourier transform of the signal+noise

    fftw_destroy_plan(p);
    
    p = fftw_plan_dft_1d(n_size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    // Retrieve all the FFT output info for the
    // rootuple
    // Here we store the whitened signal
    
    for (Int_t j=0; j<=T->size()/2-1; j++)
    {
        re = output[j][0];
        im = output[j][1];
        
       Hfr->push_back(re/sqrt(T->size()));
       Hfi->push_back(im/sqrt(T->size()));
    }
    
    fftw_destroy_plan(p);
    
    // Fill the ROOT tree
    Chirparams -> Fill();
    fftw_cleanup();
}

void chirpgen::reset()
{
  m_mass1=0;
  m_mass2=0;
  m_snr=0;

  
  Signal->clear();
  Signaln->clear();
  T->clear();
  H->clear();
  N->clear();
  Tf->clear();
  Sfr->clear();
  Sfi->clear();
  Snfr->clear();
  Snfi->clear();
  Hfr->clear();
  Hfi->clear();
  Nfr->clear();
  Nfi->clear();
    
  tchirp = 0;
  t_init=0;
  t_bin=0;
  f_init=0;
  f_bin=0;
}

//
// ROOTuple creation
//

void chirpgen::initTuple()
{
    m_outfile  = new TFile(m_outf.c_str(),"recreate");

    Chirparams  = new TTree("Chirpinfo","");

    Chirparams->Branch("Signal",&Signal);
    Chirparams->Branch("Signaln",&Signaln);
    Chirparams->Branch("H",&H);
    Chirparams->Branch("T",&T);
    Chirparams->Branch("H",&H);
    Chirparams->Branch("N",&N);
    Chirparams->Branch("mass1",&m_mass1);
    Chirparams->Branch("mass2",&m_mass2);
    Chirparams->Branch("tchirp",&tchirp);
    Chirparams->Branch("Tf",&Tf);
    Chirparams->Branch("Hfr",&Hfr);
    Chirparams->Branch("Hfi",&Hfi);
    Chirparams->Branch("Sfr",&Sfr);
    Chirparams->Branch("Sfi",&Sfi);
    Chirparams->Branch("Snfr",&Snfr);
    Chirparams->Branch("Snfi",&Snfi);
    Chirparams->Branch("Nfr",&Nfr);
    Chirparams->Branch("Nfi",&Nfi);
    Chirparams->Branch("t_init",&t_init);
    Chirparams->Branch("t_bin",&t_bin);
    Chirparams->Branch("f_init",&f_init);
    Chirparams->Branch("f_bin",&f_bin);
    Chirparams->Branch("SNR",&m_snr);
}


