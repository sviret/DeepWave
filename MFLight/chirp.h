//
//  chirp.h
//  
//
//  Created by Sebastien Viret on 17/09/2021.
//

#ifndef chirp_h
#define chirp_h

#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>

class chirp
{
public:
    chirp();
    chirp(double mass1,double mass2,double theta, double dist);
    ~chirp();
 
    void init(double mass1,double mass2,double theta, double dist);
    double get_Phi(double t);
    double get_a(double t);
    double get_h(double t);
    double get_time(double freq);

    double get_t0();
    double get_tc();

    
private:
    
    const double G     = 6.67*pow(10.,-11.);  // Gravitational constant
    const double c     = 3.*pow(10.,8.);      // Speed of light
    const double m_sol = 1.988*pow(10.,30.);  // Solar mass
    const double Mpsec = pow(10.,6.)*3.08*pow(10.,16.); // 1Mpc, 1pc=3.26al=3,08*10^16m
    double m_mass1;    // Mass of object 1, in solar masses
    double m_mass2;    // Mass of object 2, in solar masses
    double m_theta;    // Angle between the observer and a vector normal to the system plane
    double m_dist;     // Distance between the observer and the system in Mpc

    double m_M;        // Total mass (m1+m2)
    double m_eta;      // Reduced mass (m1*m2)/M^2
    double m_Mc;       // Chirp mass eta*M
    double m_rs;       // Schwarzshild radius of M
    double m_rsc;      // Schwarzshild radius of Mc
    double m_tsc;      // Schwarzshild time of Mc

    double m_A;
    double m_r0;
    double m_tc;
    double m_t0;
};

#endif
