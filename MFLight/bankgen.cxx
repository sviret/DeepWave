///////////////////////////////////
//
// GW bank generation
//
//  Authors: S.Viret, based on initial code from B.Branchu and E.Pont
//  Date   : 21/01/2022
//
//  Input parameters:
//
//  mass1/mass2 : mass range within which the templates are generated
//                given in solar masses units
//  outfile     : name of the ROOT file containing the bank
//
//  The templates are generated for all mass pairs between m1 and m2, with
//  a mass step of 1
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "bankgen.h"
#include <fftw3.h>

// Main constructor
// Bank initialization

bankgen::bankgen(double mass_min,double mass_max,double noise , std::string outfile):
T(new std::vector<double>),
H(new std::vector<double>),
Tf(new std::vector<double>),
Hfr(new std::vector<double>),
Hfi(new std::vector<double>)
{
    
  bankgen::reset();
    
  f_s=2.048;
    

  t_init=-30;
  t_bin=(1/(f_s*1000));
  f_init=0;
  f_bin=(t_bin/60)*1000*f_s;
    
  m_outf=outfile;
  m_mass1=mass_min;
  m_mass2=mass_max;
  m_sigma=noise*1e-23; // Noise value is rescaled to strain unit

  bankgen::initTuple();    // Create the bank ROOT file
  bankgen::create_bank();  // Bank creation
     
   // Write the output root file
   m_outfile->Write();
   m_outfile->Close();
}


//////////////////////////////////////////////
//
// 
//////////////////////////////////////////////
void bankgen::create_bank()
{
    double fi = 15;      // Interferometer low sensitivity (in Hz)
    double ff = 1000;    // Interferometer high sensitivity (in Hz)

    // We define the chirp over a maximum period of 30s here
    double duration = 30;  // Max signal duration, in seconds
    
    // Use it to get the number of sampling points over measurement time
    int  n_size=static_cast<int>(duration*f_s*1000)+1;
    
    //Allocate an array big enough to hold the transform output
    //Transform output in 1d contains, for a transform of size N,
    //N/2+1 complex numbers, i.e. 2*(N/2+1) real numbers
    //our transform is of size n+1, because the histogram has n+1 bins
    
    // Important note: Here we are creating the bank, we will therefore
    // just keep the useful signal part
    
    double re,im;
    double *in = new double[2*(n_size/2+1)];

    // We will need it for whitening
    m_psd=2*m_sigma*m_sigma*n_size;
    
    // The FFT is handled via the FFTW algorithm
    //
    // https://www.fftw.org
    //
 
    fftw_complex *input;
    fftw_complex *output;
    fftw_plan p;
    
    chirp *mychirp=new chirp();
    double scaling=1e-21;
    double eta;
    
    // Create the bank of chirp signals
    
    for(double m1=m_mass1 ; m1<=m_mass2 ; m1++)
    {
        for(double m2=m1 ; m2<=m_mass2 ; m2++)  // Pair are just done once (m1,m2) and not (m2,m1)
        {
            mychirp->init(m1,m2,1);
            T->clear();
            H->clear();
            Tf->clear();
            Hfr->clear();
            Hfi->clear();
            
            // Define the time range when the signal frequency will
            // be within the detector sensitivity
            // Divide by 2 because binary orbital rotation frequency is
            // half the frequency of the wave
    
            t_i = mychirp->get_time(0.8*fi/2);
            t_f = mychirp->get_time(ff/2);
    
            // Here we resize the Fourier transform as we perform it
            // only on the useful part of the signal
            
            int npts= std::min(n_size,static_cast<int>((-t_i)/t_bin)+1);

            input = (fftw_complex*) fftw_malloc(npts*2 * sizeof(fftw_complex));
            output = (fftw_complex*) fftw_malloc(npts*2 * sizeof(fftw_complex));

            for(int k=0;k<n_size;++k) in[k]=0;
            
            for (int i=0; i<=npts; i++)
            {
                input[i][0]=0.;
                input[i][1]=0.;
                output[i][0]=0.;
                output[i][1]=0.;
            }
                
            cout <<endl;
            cout << "Feeding the bank with the following template" <<endl;
            cout << "m1 = " << m1 << " solar masses" <<endl;
            cout << "m2 = " << m2 << " solar masses" <<endl;
            cout << "At ti = " << t_i << " s the wave frequency is " << fi << " Hz" <<endl;
            cout << "At tf = " << t_f << " s the wave frequency is " << ff << " Hz" <<endl;
            cout << "Coalescence at tc = " << mychirp->get_tc() << " s " <<endl;
            
            mass1=m1;
            mass2=m2;
            
            eta=(m1*m2)/(pow(m1+m2,2.));
            mc=pow(eta,3./5.)*(m1+m2);
            
            int i = 0;
            for(double t=0. ; t<=-t_i ; t=t+(1/(f_s*1000)))
            {
                ++i;
                in[i] = mychirp->get_h(t+t_i);

                input[i][0]=in[i]/scaling;
                input[i][1]=0.;
                
                T->push_back(t);
            }
                
            // Signal is created, we FFT it
            p = fftw_plan_dft_1d(npts, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(p);
            fftw_destroy_plan(p);
            
            // Important to store the frequency bin width as it
            // will depend on the template here.
            
            //f_bin=1./T->size()*f_s*1000;
            f_bin=1./(std::min(duration,-t_i));
     
            SNRmax=0.;

            for (int i=0; i<=npts/2-1; i++)
            {
                // Non-whitened FFT
                
                re = output[i][0]*scaling/sqrt(T->size());
                im = output[i][1]*scaling/sqrt(T->size());
                
                // Whitened
                re /= sqrt(m_psd/2);
                im /= sqrt(m_psd/2);
                                
                Tf->push_back(float(i)*f_bin);
                Hfr->push_back(re);
                Hfi->push_back(im);
                
                //if (re*re+im*im>10000)
                //    cout << i*f_bin << " /" << re*re+im*im << endl;
                
                if (i*f_bin<15) continue;
                if (i*f_bin>1000) continue;

                // Scaling factor to get the correct SNRmax
                // in the detector freq domain
                SNRmax+=(re*re+im*im);
            }
                        
            SNRmax=sqrt(2*SNRmax); // Final norm factor
            std::cout<< "SNRmax compute in the frequency range [15,1000]: " << SNRmax << std::endl;
            std::cout<< "Whitened signal with this value will have an SNR of 1 " << std::endl;
            
            
            input=0;
            output=0;
            bankparams -> Fill();
        }
    }
}



void bankgen::reset()
{
  m_mass1=0;
  m_mass2=0;
  m_psd=0;
    
  T->clear();
  H->clear();
  Tf->clear();
  Hfr->clear();
  Hfi->clear();

  tchirp = 0;
  mass1=0;
  mass2=0;
  mc=0;
  t_i=0;
  t_f=0;
  t_init=0;
  t_bin=0;
  f_init=0;
  f_bin=0;
  SNRmax=0;
}


//
// ROOTuple creation
//

void bankgen::initTuple()
{
    m_outfile  = new TFile(m_outf.c_str(),"recreate");

    bankparams  = new TTree("Bank","");
      
    bankparams->Branch("T",&T);            // Sampling time of the template
    bankparams->Branch("H",&H);            // Corresponding h value
    bankparams->Branch("mass1",&mass1);    // Ma (in solar masses)
    bankparams->Branch("mass2",&mass2);    // Mb (in solar masses)
    bankparams->Branch("mchirp",&mc);      // Mchirp (in solar masses)
    bankparams->Branch("t_i",&t_i);        // Time at which the template enters the frequency
                                           // range of the interferometer (f_low)
    bankparams->Branch("t_f",&t_f);        // Time at which the template exits the frequency
                                           // range of the interferometer (f_high)
    bankparams->Branch("Tf",&Tf);          // Sampling frequency for the template
    bankparams->Branch("Hfr",&Hfr);        // Fourier transform real part
    bankparams->Branch("Hfi",&Hfi);        // Fourier transform imaginary part
    bankparams->Branch("t_init",&t_init);  // Starting time
    bankparams->Branch("t_bin",&t_bin);    // Time bin
    bankparams->Branch("f_init",&f_init);  // Starting frequency
    bankparams->Branch("f_bin",&f_bin);    // Frequency bin
    bankparams->Branch("SNRmax",&SNRmax);  // The normalisation factor to be used with the filter
}


