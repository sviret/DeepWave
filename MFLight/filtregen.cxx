///////////////////////////////////
//
// GW signal detection with matched filtering
//
//  Authors : S.Viret, based on initial code from B.Branchu and E.Pont
//  Date    : 21/01/2022
//  Maj.Rev.: 13/01/2023
//
//  Input parameters:
//
//  bank        : name of the rootfile containing the template bank
//  signal      : name of the rootfile containing the signal to analyze
//  outfile     : name of the rootfile containing the final data
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "filtregen.h"

// Main constructor
filtregen::filtregen(std::string bank, std::string signal, std::string outfile):
Htfr(new std::vector<double>),
Htfi(new std::vector<double>),
Tfin(new std::vector<double>),
Hfin(new std::vector<double>),
Hfinr(new std::vector<double>),
Hfini(new std::vector<double>)
{
  filtregen::reset();
    
  m_outs=signal;
  m_outb=bank;
  m_outf=outfile;
  filtregen::initTuple(); // Create the output ROOT file
    
  filtregen::do_MF();
     
   // Write the output root file
  m_outfile->Write();
  m_outfile->Close();}


//
// Here we perform the matched filtering over the bank of templates
//

void filtregen::do_MF()
{
     
  // First of all one has to retrieve all the data
    
  TChain *Bank = new TChain("Bank");
  TChain *Signalinfo = new TChain("Chirpinfo");
  Bank->Add(m_outb.c_str());
  Signalinfo->Add(m_outs.c_str());


  std::vector<double>  mH;
  std::vector<double>  mHs;
  std::vector<double>  mHfr;
  std::vector<double>  mHfi;
  std::vector<double>  mNfr;
  std::vector<double>  mNfi;
  std::vector<double>  mHsfr;
  std::vector<double>  mHsfi;

  std::vector<double>  *H = new std::vector<double>;
  std::vector<double>  *Hs = new std::vector<double>;
  std::vector<double>  *Hfr = new std::vector<double>;
  std::vector<double>  *Hfi = new std::vector<double>;
  std::vector<double>  *Nsfr = new std::vector<double>;
  std::vector<double>  *Nsfi = new std::vector<double>;
  std::vector<double>  *Hsfr = new std::vector<double>;
  std::vector<double>  *Hsfi = new std::vector<double>;

  double s_mass1;
  double s_mass2;
  double b_mass1;
  double b_mass2;
  double f_m1,f_m2,f_mc,s_mc;
  double f_binbank;
  double tchirp;
  double norm,SNRmax;
    
  H=&mH;
  Hs=&mHs;
  Hfr=&mHfr;
  Hfi=&mHfi;
  Nsfr=&mNfr;
  Nsfi=&mNfi;
  Hsfr=&mHsfr;
  Hsfi=&mHsfi;

  Bank->SetBranchAddress("H",&H);
  Bank->SetBranchAddress("Hfr",&Hfr);
  Bank->SetBranchAddress("Hfi",&Hfi);
  Bank->SetBranchAddress("mass1",&b_mass1);
  Bank->SetBranchAddress("mass2",&b_mass2);
  Bank->SetBranchAddress("f_bin",&f_binbank);
  Bank->SetBranchAddress("SNRmax",&SNRmax);
    
  Signalinfo->SetBranchAddress("H",&Hs);
  Signalinfo->SetBranchAddress("Hfr",&Hsfr);
  Signalinfo->SetBranchAddress("Hfi",&Hsfi);
  Signalinfo->SetBranchAddress("Nfr",&Nsfr);
  Signalinfo->SetBranchAddress("Nfi",&Nsfi);
  Signalinfo->SetBranchAddress("tchirp",&tchirp);
  Signalinfo->SetBranchAddress("mass1",&s_mass1);
  Signalinfo->SetBranchAddress("mass2",&s_mass2);
  Signalinfo->SetBranchAddress("t_init",&t_init);
  Signalinfo->SetBranchAddress("t_bin",&t_bin);
  Signalinfo->SetBranchAddress("f_init",&f_init);
  Signalinfo->SetBranchAddress("f_bin",&f_bin);
    
  int bestentry = 0;
  double maxTtot= 0.;
  double maxHtot = 0.;
  double maxT= 0.;
  double maxH = 0.;
  int i_bk;
    
  f_m1=0;
  f_m2=0;
  f_mc=0;
    
  // OK we have everything, we can start
    
  // Get the signal
  Signalinfo->GetEntry(0);

  int  n=Hsfr->size();

  double eta,m1,m2;
    
  m1=s_mass1;
  m2=s_mass2;
  eta=(m1*m2)/(pow(m1+m2,2.));
    
  s_mc=pow(eta,3./5.)*(m1+m2);
  
  // Allocate sufficient space in the TF plan elements
  input = (fftw_complex*) fftw_malloc(n*2 * sizeof(fftw_complex));
  output = (fftw_complex*) fftw_malloc(n*2 * sizeof(fftw_complex));
    
  // Run the matched filtering over all the bank parameters
  int n_entries = Bank->GetEntries();
    
  for (int j=0;j<n_entries;++j)
  {
    if (j%10==0) cout << "Processing template " << j << "/" << n_entries << endl;
    Bank->GetEntry(j);
      
    m1=b_mass1;
    m2=b_mass2;
    eta=(m1*m2)/(pow(m1+m2,2.));
    
    c_mass2=b_mass2;
    c_mass1=b_mass1;
    c_mc=pow(eta,3./5.)*(m1+m2);
      
    Htfr->clear();
    Htfi->clear();
    Hfin->clear();
    Hfinr->clear();
    Hfini->clear();
    Tfin->clear();
      
    // We run over all the signal FFT bins
    // to compute the normalization factor of the template
    double sigval=SNRmax;
     // std::cout << f_bin << " / " << f_binbank << endl;
    for (int i=0;i<mHsfr.size();i++)
    {
      input[i][0]= 0.;
      input[i][1]= 0.;
        
      Htfr->push_back( 0.);
      Htfi->push_back( 0.);
      
      // We do that only within the detector sensitivity
      // (== infinite noise ponderation elsewhere)
      //
      if (f_init+i*f_bin<15) continue;
      if (f_init+i*f_bin>1000) continue;
  
      // The template fourier transform is not
      // computed on the same number of point.
      // One has to find the right index
      // and normalize correctly the bin size
        
      i_bk=static_cast<int>(i*f_bin/f_binbank);
      norm=sqrt(f_bin/f_binbank);
    
      // Power spectral density of the noise at frequency f
      // is not needed because signals are already whitened

        
      // Matched filter value for bin i
      // Complete this part
      Htfr->at(i)=0.;
      Htfi->at(i)=0.;
               
      input[i][0]= Htfr->at(i);
      input[i][1]= Htfi->at(i);
    }
          
    // Now compute the inverse FFT
    // It should peak at T=Tchirp for the good template
      
    p = fftw_plan_dft_1d(n, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
      
    // Normalisation factor, we have n sampling points but sigval is a sqrt, so n/sqrt(n)
    // The factor 2 wandering around comes from the fact that size of the backward FFT is
    // half of the initial one (only account for positive frequencies)
    // Bin size at the output is therefore 2*t_bin
 
    // Filtered signal along time
    for (int binx = 1; binx<=n; binx++)
    {
        //cout << output[binx-1][0] << "//" << output[binx-1][1] << endl;
        Hfin->push_back( 2*sqrt(output[binx-1][0]*output[binx-1][0]+output[binx-1][1]*output[binx-1][1])/sigval);
        Hfinr->push_back( 2*output[binx-1][0]/sigval);
        Hfini->push_back( 2*output[binx-1][1]/sigval);
        Tfin->push_back( t_init+2*binx*t_bin );
    }
      

   
    maxH = 0.;
    maxT = 0.;
      
    // Now look for the time where the filter output is maximum (TChirp)
      
    for (int k = 0; k<Hfin->size(); ++k)
    {
        if (abs(Hfin->at(k)) > maxH)
        {
            maxH = abs(Hfin->at(k));
            maxT = Tfin->at(k);
        }
    }

    maxrho=maxH;
    Filtreparams->Fill();
      
    // Check if this is the largest obtained so far
    if (maxH > maxHtot)
    {
        maxHtot = maxH;
        maxTtot = maxT;
        bestentry = j;
        f_m1=c_mass1;
        f_m2=c_mass2;
        f_mc=c_mc;
    }
 }  
   
 // Finally plot some results
    
 cout << endl;
 cout << " End of the match filtering loop  " << bestentry << endl;
 cout << "    Best match found is mc - m1/m2  = " << f_mc << " - " << f_m1 << "/" << f_m2 << endl;
 cout << "    *Real signal is mc - m1/m2      = " << s_mc << " - " << s_mass1 << "/" << s_mass2 << endl;
 cout << "    Coalescence time found is  = " << maxTtot << " s " << endl;
 cout << "    *Real Tc time is           = " << tchirp << " s" << endl;
 cout << "    Peak magnitude is  " << maxHtot << endl;

 input=0;
 output=0;
}

void filtregen::reset()
{
  c_mass1=0;
  c_mass2=0;
  c_mc=0;
  maxrho=0;
    
  Htfr->clear();
  Htfi->clear();
  Hfin->clear();
  Hfinr->clear();
  Hfini->clear();
  Tfin->clear();
}


void filtregen::initTuple()
{
    m_outfile  = new TFile(m_outf.c_str(),"update");

    Filtreparams  = new TTree("Filtreinfo","");
      
    Filtreparams->Branch("Htfr",&Htfr);
    Filtreparams->Branch("Htfi",&Htfi);
    Filtreparams->Branch("c_mass1",&c_mass1);
    Filtreparams->Branch("c_mass2",&c_mass2);
    Filtreparams->Branch("m_chirp",&c_mc);
    Filtreparams->Branch("Hfin",&Hfin);
    Filtreparams->Branch("Hfinr",&Hfinr);
    Filtreparams->Branch("Hfini",&Hfini);
    Filtreparams->Branch("Tfin",&Tfin);
    Filtreparams->Branch("peakrho",&maxrho);
}

