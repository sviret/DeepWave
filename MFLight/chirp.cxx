///////////////////////////////////
//
// GW chirp base class
//
//  Authors: S.Viret
//  Date   : 21/01/2022
//
//  This class contains a serie of tools to compute the
//  strain signal w.r.t. the binary system parameters
//
//  Reminder:
//
//  h(t)=a(t)*(angular params)*sin(2*Phi(t))
//
//  For the moment the values used here are very simplified
//  No post newtonian approximation, not the exact angular dependance
//  observed in the VIRGO/LIGO system
//  We put ourselves in the approximation of (angular params)=1
//  Those details are anyway not necessary for a good comprehension
//  of the matched filtering process
//
//  Code is described on the following page:
//  http://sviret.web.cern.ch/sviret/Welcome.php?n=Virgo.Ana
//
//  The calculations are described in the following document:
//  http://sviret.web.cern.ch/sviret/docs/Chirps.pdf
//
//  Questions/Comments: s.viret_at_ip2i.in2p3.fr
//
///////////////////////////////////

#include "chirp.h"

//
// Base constructor
//

chirp::chirp()
{
    m_mass1=0.;
    m_mass2=0.;
    m_dist=0.;
    
    m_M=0;
    m_eta=0;
    m_M=m_mass1+m_mass2;
    m_M*=m_sol;                     // Express the chirp mass in kg
    m_Mc=pow(m_eta,3./5.)*m_M;      // Chirp mass
    m_rs=2*G*m_M/pow(c,2.);         // Shwarzshild radius
    m_rsc=2*G*m_Mc/pow(c,2.);       // Shwarzshild radius of the chirp
    m_tsc=m_rsc/c;                  // Shwarzshild time of the chirp
    m_A=0;
    
    m_r0=10;                        // Initial distance bet. the binaries, in rs units
    m_tc=0;                         // Coalescence time is at 0
}

//
// Normal object constructor (Default)
//

chirp::chirp(double mass1,double mass2,double dist)
{
    m_mass1=mass1;
    m_mass2=mass2;
    m_dist=dist*Mpsec;
    
    m_M=m_mass1+m_mass2;
    m_eta=(m_mass1*m_mass2)/(pow(m_M,2.));
    m_M*=m_sol;
    m_Mc=pow(m_eta,3./5.)*m_M;          // Chirp mass
    m_rs=2*G*m_M/pow(c,2.);             // Shwarzshild radius
    m_rsc=2*G*m_Mc/pow(c,2.);           // Shwarzshild radius of the chirp
    m_tsc=m_rsc/c;                      // Shwarzshild time of the chirp
    m_A=pow(2./5.,-1./4.)*(m_rsc/(4.*m_dist));
    
    m_r0=10;                            // Initial distance bet. the binaries, in rs units
    m_tc=0;                             // Coalescence time is at 0
    
    // Frequency of the signal when the dist between the 2 objects is
    // equal to m_r0*rs
    
    double f0 = 1./(4.*atan(1.))*sqrt(G*m_M/pow(m_r0*m_rs,3.));
    
    m_t0=chirp::get_time(f0);           // Initial time (when r=r0)
}

chirp::~chirp()
{}

//
// This method can be used to change the parameters of the object
// useful for the bank creation
//

void chirp::init(double mass1,double mass2, double dist)
{
    m_mass1=mass1;
    m_mass2=mass2;
    m_dist=dist*Mpsec;
    
    m_M=(m_mass1+m_mass2);
    m_eta=(m_mass1*m_mass2)/(pow(m_M,2.));
    m_M*=m_sol;
    m_Mc=pow(m_eta,3./5.)*m_M;            // Chirp mass
    m_rs=2*G*m_M/pow(c,2.);               // Shwarzshild radius
    m_rsc=2*G*m_Mc/pow(c,2.);             // Shwarzshild radius of the chirp
    m_tsc=m_rsc/c;                        // Shwarzshild time of the chirp
    m_A=pow(2./5.,-1./4.)*(m_rsc/(4.*m_dist));
    
    m_r0=10;                              // Initial distance bet. the binaries, in rs units
    m_tc=0;                               // Coalescence time is at 0
    
    double f0 = 1./(4.*atan(1.))*sqrt(G*m_M/pow(m_r0*m_rs,3.));
    
    m_t0=chirp::get_time(f0);
}


//
// Method returning the value of Phi(t) (one takes Phi(t=tc)=0 for simplification)
//
// !!SV note: this is the 0PN approximation, should be improved possibly in the
// future (but not mandatory to understand MF)

double chirp::get_Phi(double t)
{
    return -pow(2.*(m_tc-t)/(5.*m_tsc),5./8.);
}

//
// Method returning the value of a(t)
//

double chirp::get_a(double t)
{
    return m_A*pow((m_tc-t)/(m_tsc),-1./4.);
}

//
// Method returning the value of the strain for a given time.
//
// !!SV note: angular dependencies are not accounted for here, we assume maximal amplitude
//

double chirp::get_h(double t)
{
    return chirp::get_a(t)*cos(2.*chirp::get_Phi(t));
}

//
// Method returning rotation frequency of the binary system at a given time
//

double chirp::get_time(double freq)
{
    return m_tc-5./2.*pow(0.25,8./3.)*pow(m_tsc,-5./3.)*pow(8.*atan(1.)*freq,-8./3.);
}


double chirp::get_t0()
{
    return m_t0;
}

double chirp::get_tc()
{
    return m_tc;
}
