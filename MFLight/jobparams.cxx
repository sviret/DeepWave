#include "jobparams.h"

//tclap
#include <tclap/CmdLine.h>
using namespace TCLAP;

jobparams::jobparams(int argc, char** argv){
  


   try {
     // command line parser
     CmdLine cmd("Command option", ' ', "0.9");

     ValueArg<std::string> option("c","case","type of analysis (getchirp/...)",
				false, "getchirp", "string");
     cmd.add(option);
       
     ValueArg<double> dist("d","dist","distance bet. the 2 sources (in Mpsec)",
				   false, 1., "float");
     cmd.add(dist);

     ValueArg<double> theta("t","theta","Azimuthal angle bet. observer and the normal to the binary system plane (in deg)",
				    false, 10., "float");
     cmd.add(theta);

     ValueArg<std::string> outfile("o","outfile","name of the output ROOT file",
				false, "data.root", "string");
     cmd.add(outfile);
       
     ValueArg<std::string> bkfile("b","bankfile","name of the bank ROOT file",
                 false, "bank.root", "string");
     cmd.add(bkfile);
       
     ValueArg<std::string> dtfile("p","datafile","name of the input data ROOT file",
                false, "data.root", "string");
     cmd.add(dtfile);
       
     ValueArg<double> mass1("m","mass1","mass of object 1 (in solar masses)",
			   false, 1., "float");
     cmd.add(mass1);

    ValueArg<double> mass2("n","mass2","mass of object 2 (in solar masses)",
               false, 1., "float");
    cmd.add(mass2);
       
    ValueArg<double> te("e","te","chirp starting time (in s)",
                  false, 1., "float");
    cmd.add(te);

    ValueArg<double> sigma("s","sigma","standard deviation of the normal distribution",
                  false, 1., "float");
    cmd.add(sigma);
     // parse
     cmd.parse(argc, argv);
     
    m_outfile      = outfile.getValue();
    m_bkfile       = bkfile.getValue();
    m_dtfile       = dtfile.getValue();
    m_theta        = theta.getValue();
    m_dist         = dist.getValue();
    m_opt          = option.getValue();
    m_mass1        = mass1.getValue();
    m_mass2        = mass2.getValue();
    m_te           = te.getValue();
    m_sigma        = sigma.getValue();
   }
    
   catch (ArgException &e){ // catch exception from parse
     std::cerr << "ERROR: " << e.error() << " for arg " << e.argId()  << std::endl;
     abort();
   }
}
