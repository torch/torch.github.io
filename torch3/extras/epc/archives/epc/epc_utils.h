#ifndef EPC_UTILS_INC
#define EPC_UTILS_INC

#include "general.h"
#include "TwoClassFormat.h"
#include "int_real.h"
#include "DiskXFile.h"

namespace Torch {
struct err {
  int fp;
  int fn;
  int tp;
  int tn;
  int n_p;
  int n_n;
  int n;
};

struct real_3 {
  real real1;
  real real2;
  real real3;
};

	//general functions
	int compar_real3(const void *p1, const void *p2);
	void sort_scores(Int_real* to_sort, err* e, bool sort);
	void compute4values(real thrd, Int_real* to_sort, err* e,bool sort,bool equal);

	//precision and recall
	real computePR_DCF(Int_real* to_sort, err* e,real ratio_of_precision, bool sort);
	real computeThGivenRecall(Int_real* to_sort, err* e,real given_recall, bool sort);
	real computeThGivenPrecision(Int_real* to_sort, err* e,real given_precision, bool sort);
	real epcPrecision(real incr,real bound,Int_real* train_scores,Int_real* test_scores,err* etr, err* ete, DiskXFile* f,real_3* prec_scores,bool F1,bool aurocc);
	real computeBEP(Int_real* to_sort, err* e,bool sort);
	
	// sensitivity specificity
	real computeSS_DCF(Int_real* to_sort, err* e, real ratio_of_sensitivity, bool sort);
	real computeSS_BEP(Int_real* to_sort, err* e,bool sort);
	real computeThGivenSensitivity(Int_real* to_sort, err* e,real given_sensitivity, bool sort);
	real computeThGivenSpecificity(Int_real* to_sort, err* e,real given_specificity, bool sort);
	
	// far frr
	real computeDCF(Int_real* to_sort, err* e,real ratio_of_far, bool sort);
	real computeNISTDCF(Int_real* to_sort, err* e,real ratio_of_far, bool sort);
	real computeThGivenFAR(Int_real* to_sort, err* e,real given_far, bool sort);
	real computeThGivenFRR(Int_real* to_sort, err* e,real given_frr, bool sort);
	real ppndf(real p);
	real computeEER(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_ = -1, bool sort=true);
	real computeHTER(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_ = -1, bool sort=true);
	void computeFaFr(real thrd, Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_ = -1,bool sort = true);
	real computeClassERR(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_ = -1, bool sort=true);

}
#endif
