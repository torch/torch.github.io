#ifndef INT_REAL_INC
#define INT_REAL_INC

#include "general.h"

namespace Torch {

/** This simple structure is used to sort data.
    #the_int# contains generally the index of a structured object,
    while #the_real# contains the value by which we wish
    to sort the objects.

    @author Samy Bengio (bengio@idiap.ch)
*/

struct Int_real {
  int the_int;
  real the_real;
};

struct real_real {
  real real1;
  real real2;
  real* p_real;
};

struct Int_char {
  int the_int;
  char *the_char;
};

/// this function returns 1 if p1->the_real > p2->the_real
extern "C" int compar_int_real(const void *p1, const void *p2);

/// this function returns 1 if p1 > p2
extern "C" int compar_real(const void *p1, const void *p2);

/// this function returns 1 if p1 > p2
extern "C" int compar_real_real(const void *p1, const void *p2);

/// this function returns 1 if p1->the_char > p2->the_char
extern "C" int compar_int_char(const void *p1, const void *p2);

}

#endif
