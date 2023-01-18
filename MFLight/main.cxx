#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>

// Internal includes

#include "testfft.h"
#include "chirpgen.h"
#include "bankgen.h"
#include "filtregen.h"
#include "jobparams.h"
#include "TROOT.h"

using namespace std;

///////////////////////////////////
//
// Toy code to play with chirps and matched filtering
//
//  Authors: S.Viret, based on initial code from B.Branchu and E.Pont
//  Date   : 21/01/2022
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

int main(int argc, char** argv)
{
  jobparams params(argc,argv);

  // Depending on the option chosen, process the information

    
  // Option 1: create a temporal chirp function and add some gaussian noise to it
  if (params.option()=="getchirp")
  {
    chirpgen* my_chirp = new chirpgen(params.mass1(),params.mass2(),params.dist(),params.sigma(),params.outfile());
    delete my_chirp;
  }

  // Option 2: filters a chirp noisy signal with a bank of template chirps
  if (params.option()=="domatch")
  {
    filtregen* my_signal = new filtregen(params.bkfile(),params.dtfile(),params.outfile());
    delete my_signal;
  }

  // Option 3: create a bank of templates
  if (params.option()=="getbank")
  {
    bankgen* my_bank = new bankgen(params.mass1(),params.mass2(),params.sigma(),params.outfile());
    delete my_bank;
  }

  // Option 4: test FFTW
  if (params.option()=="test")
  {
     testfft* my_fft = new testfft(params.freq(),params.sfreq(),params.duration(),params.outfile());
     delete my_fft;
  }
    
  // Option 5: generate a noise
  if (params.option()=="getnoise")
  {
    chirpgen* my_chirp = new chirpgen(0,0,0,params.sigma(),params.outfile());
    delete my_chirp;
  }
    
  return 0;
}
