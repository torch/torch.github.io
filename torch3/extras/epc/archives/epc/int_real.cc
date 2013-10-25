#include "int_real.h"

namespace Torch {

extern "C" int compar_int_real(const void *p1, const void *p2)
{
  Int_real v1 = *((Int_real*)p1);
  Int_real v2 = *((Int_real*)p2);
  if (v1.the_real > v2.the_real)
    return 1;
  if (v1.the_real < v2.the_real)
    return -1;
  return 0;
}

extern "C" int compar_real(const void *p1, const void *p2)
{
	real v1 = *((real*)p1);
	real v2 = *((real*)p2);
  if (v1 > v2)
    return 1;
  if (v1 < v2)
    return -1;
  return 0;
}

extern "C" int compar_real_real(const void *p1, const void *p2)
{
  real_real v1 = *((real_real*)p1);
  real_real v2 = *((real_real*)p2);
  if (v1.real1 > v2.real1)
    return 1;
  if (v1.real1 < v2.real1)
    return -1;
  return 0;
}

extern "C" int compar_int_char(const void *p1, const void *p2)
{
  Int_char v1 = *((Int_char*)p1);
  Int_char v2 = *((Int_char*)p2);
  return (strcmp(v1.the_char, v2.the_char));
}

}

