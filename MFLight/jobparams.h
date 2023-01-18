#ifndef jobparams_H
#define jobparams_H

#include <string>
#include <cstdio>
#include <tclap/CmdLine.h>
using namespace TCLAP;

class jobparams{

 public:

  /** constructer reading from standard arg */
  jobparams(int argc, char** argv);

  /** default constructer */
  jobparams(){}

  /** copy constructer */
  jobparams(const jobparams& tparams); 

  /** destructer */
  ~jobparams(){}

  /** return value */
    
  std::string option() const;
  std::string outfile() const;
  std::string bkfile() const;
  std::string dtfile() const;
  double       dist() const;
  double       theta() const;
  double       mass1() const;
  double       mass2() const;
  double       te() const;
  double       sigma() const;
  double       sfreq() const;
  double       freq() const;
  double       duration() const;

    
 private:

  std::string  m_opt;
  std::string  m_outfile;
  std::string  m_bkfile;
  std::string  m_dtfile;
  double        m_theta;
  double        m_dist;
  double        m_mass1;
  double        m_mass2;
  double        m_te;
  double        m_sigma;
  double        m_freq;
  double        m_sfreq;
  double        m_duration;
};

inline std::string jobparams::option() const{
  return m_opt;
}

inline std::string jobparams::outfile() const{
  return m_outfile;
}

inline std::string jobparams::dtfile() const{
  return m_dtfile;
}

inline std::string jobparams::bkfile() const{
  return m_bkfile;
}

inline double jobparams::mass1() const{
  return m_mass1;
}

inline double jobparams::mass2() const{
  return m_mass2;
}

inline double jobparams::theta() const{
    return m_theta;
}

inline double jobparams::dist() const{
    return m_dist;
}

inline double jobparams::te() const{
    return m_te;
}

inline double jobparams::sigma() const{
    return m_sigma;
}

inline double jobparams::freq() const{
    return m_freq;
}

inline double jobparams::sfreq() const{
    return m_sfreq;
}

inline double jobparams::duration() const{
    return m_duration;
}
#endif
