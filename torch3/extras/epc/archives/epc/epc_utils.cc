#include "epc_utils.h"

namespace Torch{
	real computeEER(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_,bool sort){

		//sort the scores
		if(sort)
			qsort(to_sort, n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int number_of_clients = 0;
		if(number_of_clients_ < 0){
			for(int i=0;i<n;i++)
				if(to_sort[i].the_int == 1)
					number_of_clients++;
		}else
			number_of_clients = number_of_clients_;
		int number_of_impostors = n - number_of_clients;	

		int n_i = number_of_impostors;
		int n_c = 0;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=0;i<n;i++){
			if (to_sort[i].the_int){
				n_c++;
			}else{
				n_i--;
			}
			real current_thrd = to_sort[i].the_real;
			real current_far=n_i/(real)number_of_impostors;
			real current_frr=n_c/(real)number_of_clients;
			cost = fabs(current_far-current_frr);
			if(cost < cost_min){
				cost_min = cost;
				*frr = current_frr;
				*far = current_far;
				if(i != n -1)
					thrd_min = (current_thrd + to_sort[i+1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}  
		return (thrd_min);
	}

	real computeHTER(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_,bool sort){

		//sort the scores
		if(sort)
			qsort(to_sort, n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int number_of_clients = 0;
		if(number_of_clients_ < 0){
			for(int i=0;i<n;i++)
				if(to_sort[i].the_int == 1)
					number_of_clients++;
		}else
			number_of_clients = number_of_clients_;
		int number_of_impostors = n - number_of_clients;	

		int n_i = number_of_impostors;
		int n_c = 0;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=0;i<n;i++){
			if (to_sort[i].the_int){
				n_c++;
			}else{
				n_i--;
			}
			real current_thrd = to_sort[i].the_real;
			real current_far=n_i/(real)number_of_impostors;
			real current_frr=n_c/(real)number_of_clients;
			cost = current_far+current_frr;
			if(cost < cost_min){
				cost_min = cost;
				*frr = current_frr;
				*far = current_far;
				if(i != n -1)
					thrd_min = (current_thrd + to_sort[i+1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}  
		return (thrd_min);
	}



	void computeFaFr(real thrd, Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_,bool sort){

		//sort the scores
		if(sort)
			qsort(to_sort, n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int number_of_clients = 0;
		if(number_of_clients_ < 0){
			for(int i=0;i<n;i++)
				if(to_sort[i].the_int == 1)
					number_of_clients++;
		}else
			number_of_clients = number_of_clients_;
		int number_of_impostors = n - number_of_clients;	

		int n_i = number_of_impostors;
		int n_c = 0;
		real thrd_min = to_sort[0].the_real;

		int i = 0;
		while((i < n - 1) && (thrd>thrd_min)){
			if (to_sort[i].the_int){
				n_c++;
			}else{
				n_i--;
			}
			thrd_min = to_sort[i+1].the_real;
			i++;
		}
		*far=n_i/(real)number_of_impostors;
		*frr=n_c/(real)number_of_clients;
	}

	real computeClassERR(Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_,bool sort){

		//sort the scores
		if(sort)
			qsort(to_sort, n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int number_of_clients = 0;
		if(number_of_clients_ < 0){
			for(int i=0;i<n;i++)
				if(to_sort[i].the_int == 1)
					number_of_clients++;
		}else
			number_of_clients = number_of_clients_;
		int number_of_impostors = n - number_of_clients;	

		int n_i = number_of_impostors;
		int n_c = 0;
		real cost_min = INF;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=0;i<n;i++){
			if (to_sort[i].the_int){
				n_c++;
			}else{
				n_i--;
			}
			real current_thrd = to_sort[i].the_real;
			real current_far=n_i/(real)n;
			real current_frr=n_c/(real)n;
			cost = fabs(n_c + n_i);
			if(cost < cost_min){
				cost_min = cost;
				*frr = current_frr;
				*far = current_far;
				if(i != n -1)
					thrd_min = (current_thrd + to_sort[i+1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}  
		return (thrd_min);
	}

	int compar_real3(const void *p1, const void *p2)
	{
		real_3 v1 = *((real_3*)p1);
		real_3 v2 = *((real_3*)p2);
		if (v1.real1 > v2.real1)
			return 1;
		if (v1.real1 < v2.real1)
			return -1;
		return 0;
	}

	void sort_scores(Int_real* to_sort, err* e, bool sort)
	{
		//sort the scores
		if(sort)
			qsort(to_sort, e->n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int n_p = 0;
		if(e->n_p < 0) {
			for(int i=0;i<e->n;i++)
				if(to_sort[i].the_int == 1)
					n_p++;
			e->n_p = n_p;
		}
		e->n_n = e->n - e->n_p;
	}

	real computeSS_DCF(Int_real* to_sort, err* e, real ratio_of_sensitivity, bool sort)
	{
		real ratio_of_specificity = 1. - ratio_of_sensitivity;

		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;
		real cost_max = -100;
		real cost = cost_max + 1;
		real thrd_max = to_sort[0].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real specificity=e->tn/(real)(e->tn + e->fp);
			real sensitivity=e->tp/(real)(e->tp + e->fn);
			cost = ratio_of_sensitivity * sensitivity + ratio_of_specificity * specificity;
			if(cost > cost_max){
				cost_max = cost;
				if(i != 0)
					thrd_max = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_max = current_thrd;
			}
		}
		return (thrd_max);
	}

	real computePR_DCF(Int_real* to_sort, err* e,real ratio_of_precision, bool sort)
	{

		sort_scores(to_sort,e,sort);
		real ratio_of_recall = 1. - ratio_of_precision;

		e->fp = 0;
		e->fn = e->n_p;
		real cost_max = -100;
		real cost = cost_max + 1;
		real thrd_max = to_sort[0].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real current_precision=e->tp/(real)(e->tp + e->fp);
			real current_recall=e->tp/(real)(e->tp + e->fn);
			cost = ratio_of_precision * current_precision + ratio_of_recall * current_recall;
			if(cost > cost_max){
				cost_max = cost;
				if(i != 0)
					thrd_max = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_max = current_thrd;
			}
		}
		return (thrd_max);
	}

	real computeDCF(Int_real* to_sort, err* e,real ratio_of_far, bool sort)
	{
		sort_scores(to_sort,e,sort);

		real ratio_of_frr = 1. - ratio_of_far;

		e->fp = 0;
		e->fn = e->n_p;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real current_far=e->fp/(real)(e->fp + e->tn);
			real current_frr=e->fn/(real)(e->fn + e->tp);
			cost = ratio_of_far * current_far + ratio_of_frr * current_frr;
			if(cost < cost_min){
				cost_min = cost;
				if(i != 0)
					thrd_min = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}
		return (thrd_min);
	}

	real computeNISTDCF(Int_real* to_sort, err* e,real ratio_of_far, bool sort)
	{
		sort_scores(to_sort,e,sort);

		//real ratio_of_far = 1. - ratio_of_frr;
		real ratio_of_frr = 1.;

		e->fp = 0;
		e->fn = e->n_p;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real current_far=e->fp/(real)(e->fp + e->tn);
			real current_frr=e->fn/(real)(e->fn + e->tp);
			cost = ratio_of_far * current_far + ratio_of_frr * current_frr;
			if(cost < cost_min){
				cost_min = cost;
				if(i != 0)
					thrd_min = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}
		return (thrd_min);
	}

	real computeBEP(Int_real* to_sort, err* e,bool sort)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[e->n-1].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real current_precision=e->tp/(real)(e->tp + e->fp);
			real current_recall=e->tp/(real)(e->tp + e->fn);
			//printf("P %f   R %f   (P+R)/2 %f \n",current_precision,current_recall,(current_precision+current_recall)/2);
			cost = fabs(current_precision-current_recall);
			if(cost <= cost_min){
				cost_min = cost;
				if(i != 0)
					thrd_min = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}
		return (thrd_min);
	}

	real computeSS_BEP(Int_real* to_sort, err* e,bool sort)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;
		real cost_min = 100;
		real cost = cost_min - 1;
		real thrd_min = to_sort[0].the_real;

		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			real current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			real current_specificity=e->tn/(real)(e->tn + e->fp);
			real current_sensitivity=e->tp/(real)(e->tp + e->fn);
			cost = fabs(current_sensitivity-current_specificity);
			if(cost < cost_min){
				cost_min = cost;
				if(i != 0)
					thrd_min = (current_thrd + to_sort[i-1].the_real)/2.0;
				else
					thrd_min = current_thrd;
			}
		}
		return (thrd_min);
	}

	real computeThGivenSensitivity(Int_real* to_sort, err* e,real given_sensitivity, bool sort)
	{

		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_sensitivity = -1;
		real previous_specificity = -1;
		real previous_thrd = -1;
		real current_sensitivity = -1;
		real current_specificity = -1;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_specificity=e->tn/(real)(e->tn + e->fp);
			current_sensitivity=e->tp/(real)(e->tp + e->fn);
			if (current_sensitivity > given_sensitivity)
				break;
			previous_sensitivity = current_sensitivity;
			previous_specificity = current_specificity;
			previous_thrd = current_thrd;
		}
		if (current_sensitivity - given_sensitivity > given_sensitivity - previous_sensitivity) {
			current_thrd = previous_thrd;
		}
		return (current_thrd);
	}

	real computeThGivenSpecificity(Int_real* to_sort, err* e,real given_specificity, bool sort)
	{

		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_sensitivity = -1;
		real previous_specificity = -1;
		real previous_thrd = -1;
		real current_specificity = -1;
		real current_sensitivity = -1;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_specificity=e->tn/(real)(e->tn + e->fp);
			current_sensitivity=e->tp/(real)(e->tp + e->fn);
			if (current_specificity < given_specificity)
				break;
			previous_sensitivity = current_sensitivity;
			previous_specificity = current_specificity;
			previous_thrd = current_thrd;
		}
		if (given_specificity - current_specificity > previous_specificity - given_specificity) {
			current_thrd = previous_thrd;
		}
		return (current_thrd);
	}

	real computeThGivenRecall(Int_real* to_sort, err* e,real given_recall, bool sort)
	{

		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_precision = -1;
		real previous_recall = -1;
		real previous_thrd = -1;
		real current_recall = -1;
		real current_precision = -1;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_precision = e->tp/(real)(e->tp + e->fp);
			current_recall=e->tp/(real)(e->tp + e->fn);
			if (current_recall > given_recall)
				break;
			previous_precision = current_precision;
			previous_recall = current_recall;
			previous_thrd = current_thrd;
		}
		if (current_recall - given_recall > given_recall - previous_recall) {
			current_thrd = previous_thrd;
		}
		//printf("given recall %f current_recall %f previous_recall %f current_thresh %f\n",given_recall,current_recall,previous_recall,current_thrd);
		return (current_thrd);
	}

	real computeThGivenPrecision(Int_real* to_sort, err* e,real given_precision, bool sort)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_precision = -1;
		real previous_recall = -1;
		real previous_thrd = -1;
		real current_precision = 0;
		real current_recall = 0;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_precision=e->tp/(real)(e->tp + e->fp);
			current_recall=e->tp/(real)(e->tp + e->fn);
			if (current_precision < given_precision)
				break;
			previous_precision = current_precision;
			previous_recall = current_recall;
			previous_thrd = current_thrd;
		}
		if (given_precision - current_precision > previous_precision - given_precision) {
			current_thrd = previous_thrd;
		}
		return (current_thrd);
	}

	real computeThGivenFAR(Int_real* to_sort, err* e,real given_far, bool sort)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_far = -1;
		real previous_frr = -1;
		real previous_thrd = -1;
		real current_far = 0;
		real current_frr = 0;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_far=e->fp/(real)(e->tn + e->fp);
			current_frr=e->fn/(real)(e->tp + e->fn);
			if (current_far > given_far)
				break;
			previous_far = current_far;
			previous_frr = current_frr;
			previous_thrd = current_thrd;
		}
		if (current_far - given_far > given_far - previous_far) {
			current_thrd = previous_thrd;
		} 
		return (current_thrd);
	}

	real computeThGivenFRR(Int_real* to_sort, err* e,real given_frr, bool sort)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;

		real previous_far = -1;
		real previous_frr = -1;
		real previous_thrd = -1;
		real current_frr = 0;
		real current_far = 0;
		real current_thrd = -1;
		for(int i=e->n-1;i>=0;i--){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			current_thrd = to_sort[i].the_real;
			e->tp = e->n_p - e->fn;
			e->tn = e->n_n - e->fp;
			current_far=e->fp/(real)(e->tn + e->fp);
			current_frr=e->fn/(real)(e->tp + e->fn);
			if (current_frr < given_frr)
				break;
			previous_far = current_far;
			previous_frr = current_frr;
			previous_thrd = current_thrd;
		}
		if (given_frr - current_frr > previous_frr - given_frr) {
			current_thrd = previous_thrd;
		}
		return (current_thrd);
	}

	void compute4values(real thrd, Int_real* to_sort, err* e,bool sort,bool equal)
	{
		sort_scores(to_sort,e,sort);

		e->fp = 0;
		e->fn = e->n_p;
		real thrd_min = to_sort[e->n-1].the_real;

		int i = e->n-1;
		while((i >=0 ) && (equal? thrd<=thrd_min : thrd<thrd_min || e->fn == e->n_p)){
			if (to_sort[i].the_int == 1){
				e->fn--;
			}else{
				e->fp++;
			}
			thrd_min = to_sort[i-1].the_real;
			i--;
		}
		e->tp = e->n_p - e->fn;
		e->tn = e->n_n - e->fp;
	}

	real epcPrecision(real incr,real bound,Int_real* train_scores,Int_real* test_scores,err* etr, err* ete, DiskXFile* f,real_3* prec_scores,bool F1,bool aurocc)
	{
		real sum = 0;
		for (int i=0;i<etr->n;i++) {
			compute4values(train_scores[i].the_real,train_scores,etr,false,false);
			//precision
			prec_scores[i].real1 = etr->tp/(real)(etr->tp + etr->fp);
			//recall
			prec_scores[i].real3 = etr->tp/(real)(etr->tp + etr->fn);
			prec_scores[i].real2 = train_scores[i].the_real;
		}
		qsort(prec_scores, etr->n, sizeof(real_real), compar_real_real);
		real r = incr;
		for (int i=0;i<etr->n;i++) {
			if (prec_scores[i].real1 > r) {
				compute4values(prec_scores[i].real2,test_scores,ete,false,false);
				real precision = ete->tp/(real)(ete->tp + ete->fp);
				real recall = ete->tp/(real)(ete->tp + ete->fn);
				if(!aurocc){
					if(!F1)
						f->printf("%f   %f   %f   %f   %f   %f   %f\n",prec_scores[i].real1,prec_scores[i].real3,0.5*prec_scores[i].real1+0.5*prec_scores[i].real3,precision,recall,0.5*precision+0.5*recall,r);
					else
						f->printf("%f   %f   %f   %f   %f   %f   %f\n",prec_scores[i].real1,prec_scores[i].real3,2*prec_scores[i].real1*prec_scores[i].real3/(prec_scores[i].real1+prec_scores[i].real3),precision,recall,2*precision*recall/(precision+recall),r);
				}
				else sum += recall + precision;
				r += incr;
				if(prec_scores[i].real1 >= bound) break;
			}
		}
		return sum;
	}

	real ppndf(real p)
	{
		real SPLIT = 0.42;
		real A0 = 2.5066282388;
		real A1 =  -18.6150006252;
		real A2 = 41.3911977353;
		real A3 = -25.4410604963;
		real B1 = -8.4735109309;
		real B2 = 23.0833674374;
		real B3 = -21.0622410182;
		real B4 = 3.1308290983;
		real C0 = -2.7871893113;
		real C1 = -2.2979647913;
		real C2 =  4.8501412713;
		real C3 =  2.3212127685;
		real D1 =  3.5438892476;
		real D2 =  1.6370678189;

		//real eps = 2.2204e-16;
		real eps = 2.2204e-7;
		real ret_val;
		real r;

		if (p >= 1.0) p = 1 - eps;
		if (p<= 0.0) p = eps;
		real q = p - 0.5;
		if (fabs(q) <= SPLIT) {
			r = q*q;
			ret_val = q * (((A3 * r + A2) * r + A1) * r + A0) /((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
		} else {
			r = q > 0.0 ? 1.0 - p : p;
			if (r <= 0)
				error("Found r = %f\n",r);
			else
				r = sqrt( - log(r));
			ret_val = (((C3 * r + C2) * r + C1) * r + C0)/((D2 * r + D1) * r + 1.0);
			if (q < 0)
				ret_val *= -1.0;
		}
		return ret_val;
	}

}
