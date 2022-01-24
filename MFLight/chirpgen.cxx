///////////////////////////////////
//
// GW signal generation and noising
//
//  Authors: S.Viret, based on initial code from B.Branchu and E.Pont
//  Date   : 21/01/2022
//
//  Input parameters:
//
//  mass1/mass2 : mass of the 2 objects (in solar masses)
//  theta       : angle between the observer and the normal to the binary system plane (in °)
//  dist        : distance between the observer and the system (in Mpsec)
//  noise       : sigma of the gaussian noise to be added to the strain (in 10-21 units)
//  outfile     : name of the rootfile containing the final data
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "chirpgen.h"

// Main constructor

chirpgen::chirpgen(double mass1,double mass2,double theta,double dist,double noise,std::string outfile):
Signal(new std::vector<double>),
T(new std::vector<double>),
H(new std::vector<double>),
N(new std::vector<double>),
Tf(new std::vector<double>),
Hfr(new std::vector<double>),
Hfi(new std::vector<double>),
Nfr(new std::vector<double>),
Nfi(new std::vector<double>)
{
  srand(time(NULL));
    
  chirpgen::reset();

  // Initialize some parameters (definition in the header file)
    
  f_s=4.096;
  t_init=-30;
  t_bin=(1/(f_s*1000));
  f_init=0;
  f_bin=(t_bin/60)*1000*f_s;
    
  m_sigma=noise;
  m_outf=outfile;
  m_mass1=mass1;
  m_mass2=mass2;
  m_dist=dist;
  m_theta=theta;
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
    
    double fi = 40;        // Interferometer low sensitivity (in Hz)
    double ff = 2000;      // Interferometer high sensitivity (in Hz)
    double duration = 60;  // Signal duration, in seconds
           
    // Use it to get the number of sampling points over measurement time
    int  n=static_cast<int>(duration*f_s*1000);
    
    //Allocate an array big enough to hold the transform output
    //Transform output in 1d contains, for a transform of size N,
    //N/2+1 complex numbers, i.e. 2*(N/2+1) real numbers
    //our transform is of size n+1, because the histogram has n+1 bins
    
    double re,im;
    double *in = new double[2*((n+1)/2+1)];
    double *noise = new double[2*((n+1)/2+1)];
    int n_size = n+1;
    
    Signal->clear();
    T->clear();
    H->clear();
    Tf->clear();
    Hfr->clear();
    Hfi->clear();
    N->clear();
    Nfr->clear();
    Nfi->clear();
    
    // Random gaussian nois distribution generation
    std::normal_distribution<double> distribution(0.,m_sigma);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    
    // The FFT is handled via a dedicated ROOT tool
    // based on the example available here:
    // https://root.cern.ch/doc/v608/FFT_8C.html
    //
    // !!SV note: this is pretty inefficient, in the future this code should be directly
    // based on FFTW class (which is used by TVirtualFFT anyway.
    
    TVirtualFFT *fft_own = TVirtualFFT::FFT(1, &n_size, "MAG R2C ES K");
    
    // Everything is setup we can now generate the signal
    
    // First we create a chirp object via the chirp class (see corresponding doc)
    // with the corresponding properties
    
    chirp *mychirp=new chirp();
    mychirp->init(m_mass1,m_mass2,m_theta*4*atan(1.)/180.,m_dist);

    // Define the time range when the signal frequency will
    // be within the detector sensitivity
    // Divide by 2 because binary orbital rotation frequency is
    // half the frequency of the wave
    // We will define the chirp signal within this range only
    
    double ti = mychirp->get_time(fi/2);
    double tf = mychirp->get_time(ff/2);
    
    // The object has tchirp=0 by construction, we have to move it
    // randomly somewhere between t=0 and t=duration/2
    
    tchirp = (rand()%(static_cast<int>(1000*duration/2.)))/1000.;
 
    cout << "--> Signal produced:" << endl;
    cout << endl;
    cout << "m1 = " << m_mass1 << " solar masses" <<endl;
    cout << "m2 = " << m_mass2 << " solar masses" <<endl;
    cout << "Dist bet m1 and m2 is 10r_s at t0 = " << mychirp->get_t0() << " s " <<endl;
    cout << "At ti = " << ti << " s the wave frequency is " << fi << " Hz" <<endl;
    cout << "At tf = " << tf << " s the wave frequency is " << ff << " Hz" <<endl;
    cout << "Coalescence at tc = " << mychirp->get_tc() << " s " <<endl;
    cout << "Moved in signal at tc = " << tchirp << " s " <<endl;
    
    ti = std::min(ti,mychirp->get_t0());
    ti = std::max(ti,-30.);
    
    cout << "Start sampling at t = " << ti << " s " <<endl;
    
    int i = 0;
    
    // A loop to produce the full signal between t_init and t_init+duration (so 60 sec)
    
    for(double t=t_init ; t<t_init+duration ; t=t+t_bin)
    {
        i++ ;
        in[i]=distribution(generator); // Noise
        noise[i]=in[i];
        N->push_back(noise[i]);
        if (t>ti+tchirp && t<tf+tchirp) in[i] += mychirp->get_h(t-tchirp); // Add signal in the relevant range
        
        T->push_back(t);
        H->push_back(in[i]); // The total signal
        if (t>ti && t<tf)
        {
            Signal->push_back(mychirp->get_h(t)); // The original chirp
        }
        else
        {
            Signal->push_back(0);
        }
        
    }
    
    if (!fft_own) return;
    
    // Now perform a Fourier transform of the signal
    
    fft_own->SetPoints(in);
    fft_own->Transform();
        
    //Copy all the output points:
    fft_own->GetPoints(in);

    // Retrieve all the FFT output info for the
    // rootuple
    
    for (Int_t j=0; j<=T->size()/2-1; j++)
    {
        re = in[2*j];
        im = in[2*j+1];
            
        Tf->push_back(float(j)/T->size()*f_s*1000);
        Hfr->push_back(re);
        Hfi->push_back(im);
    }

    // Then do the Fourier transform of the noise
    //
    // !!SV note: this is hacked a bit for the moment
    // in a future version the noise PSD should be evaluated
    // using an appropriate method
    
    fft_own = TVirtualFFT::FFT(1, &n_size, "MAG R2C ES K");

    fft_own->SetPoints(noise);
    fft_own->Transform();
        
    //Copy all the output points:
    fft_own->GetPoints(noise);

    for (Int_t j=0; j<=T->size()/2-1; j++)
    {
        re = noise[2*j];
        im = noise[2*j+1];

        /*  // !!! CODE NEEDS TO BE IMPROVED HERE !!!
        // To do this properly you need to compute the noise PSD averaging
        // a serie of noise FFT done on short noise time series
        // (Welch method)
        Nfr->push_back(re);
        Nfi->push_back(im);
        */
        
        // For the moment, as we are using gaussian noise sigma, PSD is sigma^2
        Nfr->push_back(m_sigma*sqrt(T->size()/2));
        Nfi->push_back(m_sigma*sqrt(T->size()/2));
        
    }
    // Fill the ROOT tree
    Chirparams -> Fill();
}

void chirpgen::reset()
{
  m_mass1=0;
  m_mass2=0;
  m_dist=0;
  m_theta=0;
  
  Signal->clear();
  T->clear();
  H->clear();
  N->clear();
  Tf->clear();
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
    Chirparams->Branch("Nfr",&Nfr);
    Chirparams->Branch("Nfi",&Nfi);
    Chirparams->Branch("t_init",&t_init);
    Chirparams->Branch("t_bin",&t_bin);
    Chirparams->Branch("f_init",&f_init);
    Chirparams->Branch("f_bin",&f_bin);
}


