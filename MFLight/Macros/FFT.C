///////////////////////////////////
//
//  Example of ROOT macro showing how to open
//  a root tree and to retrieve variables
//
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////


#include <string>
#include <vector>
#include <iostream>

#include "TStyle.h"
#include "TSystem.h"
#include "TFile.h"
#include "TChain.h"
#include "TBranch.h"
#include "TH2F.h"
#include "TCanvas.h"


//
// Here we open the file produce by the chirpgen command
//
// Filename is the name of the root file produced
//


void opener(std::string filename)
{
  // First of all one has to retrieve all the data
    
  TChain *mytree = new TChain("FFT");
  mytree->Add(filename.c_str());

  // You need to define the variable you will retrieve
  // when you will retrieve the data from the ROOT file they
  // will be linked to the adress of the corresponding variable
    
  std::vector<double>  *aVector = new std::vector<double>;
  double aDouble;
   
  // Link the ROOT branches with the macro's variable
    
  mytree->SetBranchAddress("Signal",&aVector);  // & because you link the branch with the variable address
  mytree->SetBranchAddress("t_bin",&aDouble);
    
  // Get the content of the branches (here there is just one entry)
  mytree->GetEntry(0);
  
  // Now the branch content is linked to the variable address, so logically you have access
  // to the corresponding values
        
  std::cout << "Value of branch time bin is " << aDouble << std::endl;
    
  int length=static_cast<int>(aVector->size());
  std::cout << "The vector contains " << length << " data points" << std::endl;
    
  // Let's loop over the vector
    
  for (int i=0;i<length;++i)
  {
    if (i%10000==0) std::cout << "aVector[" << i << "] = " << aVector->at(i) << std::endl;
  }    
}
