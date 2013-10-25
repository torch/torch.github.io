const char *help = "\
(c) Trebolloc & Co 2001\n\
\n\
measure_verif\n";

#include "CmdLine.h"
#include "IOAscii.h"
#include "DiskXFile.h"
#include "epc_utils.h"


/* NOTATION
  n      = number of examples (n = n_p + n_n)
  n_p    = number of positives
  n_n    = number of negatives
  n_fp   = number of false positives
  n_fn   = number of false negatives
  n_tp   = number of true positives
  n_tn   = number of true negatives

  n_p_tr = number of positives in the training set
  n_p_te = number of positives in the test set
*/

using namespace Torch;

int main(int argc, char **argv)
{
  char *train_filename;
  char *test_filename;
  int column;
  bool dcf_criterion = false;
  bool far_criterion = false;
  bool frr_criterion = false;
	char* criterion;
  bool no_stats;
	char* det_file;
	char* epc_file;
	bool roc;
	char* eer_file;
	char* hter_file;
  real bound, step;
  
  Allocator *allocator = new Allocator;

  CmdLine cmd;
  cmd.info(help);
  cmd.addText("\nArguments:");
  cmd.addSCmdArg("training score file", &train_filename, "ascii training matrix file");
  cmd.addSCmdArg("test score file", &test_filename, "ascii test matrix file");

  cmd.addText("\nOptions:");
  cmd.addICmdOption("-c", &column, 0, "column where the score is");
  cmd.addRCmdOption("-s", &step, 1e-3, "step between points for the curves");

  cmd.addText("\nCurves:");
  cmd.addSCmdOption("-det_file", &det_file, "", "compute FAR/FRR DET curves");
  cmd.addBCmdOption("-roc", &roc, false, "ROC instead of DET");
  cmd.addSCmdOption("-epc_file", &epc_file, "", "use the FAR/FRR DCF criterion");
  cmd.addSCmdOption("-criterion", &criterion, "dcf", "frr, far, dcf values");
  cmd.addRCmdOption("-bound", &bound,1.,"bound for the criterion values");
	
	cmd.addText("\nPoints:");
  cmd.addSCmdOption("-eer_file", &eer_file, "", "compute EER points");
  cmd.addSCmdOption("-hter_file", &hter_file, "", "compute HTER points");

	cmd.addText("\nStats:");
  cmd.addBCmdOption("-no_stats", &no_stats, false, "compute far/frr statistics");  

  cmd.read(argc, argv);

  IOAscii train_io(train_filename);
  IOAscii test_io(test_filename);

  Sequence train_sequence(1, train_io.frame_size);
  Sequence test_sequence(1, test_io.frame_size);

  if(train_io.frame_size<2)
    error("train frame size should be greather than 2: %d",train_io.frame_size);
  if(test_io.frame_size<2)
    error("test frame size should be greather than 2: %d",test_io.frame_size);
  if(column > train_io.frame_size - 1 || column > test_io.frame_size - 1)
    error("bad column number");

  Int_real* train_scores = (Int_real*)allocator->alloc(train_io.n_sequences*sizeof(Int_real));
  Int_real* test_scores = (Int_real*)allocator->alloc(test_io.n_sequences*sizeof(Int_real));

  int n_tr = train_io.n_sequences;
  int n_p_tr = 0;
  for(int t = 0; t < n_tr; t++) {
    train_io.getSequence(t, &train_sequence);
    train_scores[t].the_real = train_sequence.frames[0][column];
    int target = (int)train_sequence.frames[0][train_io.frame_size-1];
    if(target == 1)
      n_p_tr++;
    train_scores[t].the_int = target;
  }
  qsort(train_scores, n_tr, sizeof(Int_real), compar_int_real);
  err etr;
  etr.n = n_tr;
  etr.n_p = n_p_tr;
  etr.n_n = n_tr - n_p_tr;

  int n_te = test_io.n_sequences;
  int n_p_te = 0;
  for(int t = 0; t < n_te; t++) {
    test_io.getSequence(t, &test_sequence);
    test_scores[t].the_real = test_sequence.frames[0][column];
    int target = (int)test_sequence.frames[0][test_io.frame_size-1];
    if(target == 1)
      n_p_te++;
    test_scores[t].the_int = target;
  }
  qsort(test_scores, n_te, sizeof(Int_real), compar_int_real);
  err ete;
  ete.n = n_te;
  ete.n_p = n_p_te;
  ete.n_n = n_te - n_p_te;

  // compute curve

  real train_far,train_frr;
  real test_far,test_frr;
  real th = 0;
	if(!strcmp(criterion,"dcf")){
		dcf_criterion = true;
	}else
		if(!strcmp(criterion,"frr")){
			frr_criterion = true;
		}else
			if(!strcmp(criterion,"far")){
				far_criterion = true;
			}else{
				error("Option -criterion set with bad value. Take default value");
			}	

		
			if(strcmp(epc_file, "")){
				DiskXFile x_epc(epc_file,"w");
				if (dcf_criterion) {
					real the_bound = 1.;
               if (bound < 0.5) the_bound = bound;          
					else the_bound = 1-bound;
					for (real r = the_bound + step; r < 1 - the_bound; r += step) {
						th = computeDCF(train_scores,&etr,r,false);
						train_far = etr.fp/(real)(etr.fp + etr.tn);
						train_frr = etr.fn/(real)(etr.fn + etr.tp);
						compute4values(th,test_scores,&ete,false,true);
						test_far = ete.fp/(real)(ete.fp + ete.tn);
						test_frr = ete.fn/(real)(ete.fn + ete.tp);
						x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_far,train_frr,0.5*train_far+0.5*train_frr,test_far,test_frr,0.5*test_far+0.5*test_frr,r);
					}
				} else if (far_criterion) {
					for (real r = step;r<bound;r += step) {
						th = computeThGivenFAR(train_scores,&etr,r,false);
						train_far = etr.fp/(real)(etr.fp + etr.tn);
						train_frr = etr.fn/(real)(etr.fn + etr.tp);
						compute4values(th,test_scores,&ete,false,true);
						test_far = ete.fp/(real)(ete.fp + ete.tn);
						test_frr = ete.fn/(real)(ete.fn + ete.tp);
						x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_far,train_frr,0.5*train_far+0.5*train_frr,test_far,test_frr,0.5*test_far+0.5*test_frr,r);
					}
				} else if (frr_criterion) {
					for (real r = step;r<bound;r += step) {
						th = computeThGivenFRR(train_scores,&etr,r,false);
						train_far = etr.fp/(real)(etr.fp + etr.tn);
						train_frr = etr.fn/(real)(etr.fn + etr.tp);
						compute4values(th,test_scores,&ete,false,true);
						test_far = ete.fp/(real)(ete.fp + ete.tn);
						test_frr = ete.fn/(real)(ete.fn + ete.tp);
						x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_far,train_frr,0.5*train_far+0.5*train_frr,test_far,test_frr,0.5*test_far+0.5*test_frr,r);
					}
				} 
			}

			/*************** DET an ROC (Warning a posteriori measure) ************/
			DiskXFile* x_det = NULL;	
			if(strcmp(det_file, ""))
				x_det = new(allocator)DiskXFile(det_file,"w");

			if(strcmp(det_file, "")){
				DiskXFile x_det(det_file,"w");
				warning("You are using a posteriorir measure!!!");
				for (real r = step;r<bound;r += step) {
					th = computeThGivenFAR(train_scores,&etr,r,false);
					train_far = etr.fp/(real)(etr.fp + etr.tn);
					train_frr = etr.fn/(real)(etr.fn + etr.tp);
					th = computeThGivenFAR(test_scores,&ete,r,false);
					compute4values(th,test_scores,&ete,false,true);
					test_far = ete.fp/(real)(ete.fp + ete.tn);
					test_frr = ete.fn/(real)(ete.fn + ete.tp);
					if (!roc) {
						train_frr = ppndf(train_frr);
						train_far = ppndf(train_far);
						test_frr = ppndf(test_frr);
						test_far = ppndf(test_far);
					}
						x_det.printf("%f   %f   %f   %f   %f   %f   %f\n",train_far,train_frr,0.5*train_far+0.5*train_frr,test_far,test_frr,0.5*test_far+0.5*test_frr,r);
				}
			}


  // hter_point_min ?
	//
			if(strcmp(eer_file, "")){
				DiskXFile x_eer(eer_file,"w");
				th = computeHTER(train_scores,n_tr,&train_frr,&train_far,n_p_tr,false);
				computeFaFr(th,test_scores,n_te,&test_frr,&test_far,n_p_te,false);
				if (!roc) {
					test_frr = ppndf(test_frr);
					test_far = ppndf(test_far);
				}
				x_eer.printf("%f %f",test_far,test_frr);
			}

			// eer_point_min ?
			if (strcmp(hter_file,"")) {
				DiskXFile x_hter(eer_file,"w");
				th = computeEER(train_scores,n_tr,&train_frr,&train_far,n_p_tr,false);
				computeFaFr(th,test_scores,n_te,&test_frr,&test_far,n_p_te,false);
				if (!roc) {
					test_frr = ppndf(test_frr);
					test_far = ppndf(test_far);
				}
				x_hter.printf("%f %f",test_far,test_frr);
			}
  
	
	// compute stats

  
	if (!no_stats) {
		th = computeEER(test_scores,n_te,&test_frr,&test_far,n_p_te,false);
		printf("HTER test when EER test        FAR %f   FRR %f   HTER %f\n", test_far,test_frr,(test_far+test_frr)/2.);
		//int n_impostors = n_te - n_p_te;
		//  printf("test: n_clients %d n_impostors %d classif_err %f\n",n_p_te,n_impostors,(n_impostors*test_far + n_p_te*test_frr)/n_te);
		th = computeEER(train_scores,n_tr,&train_frr,&train_far,n_p_tr,false);
		computeFaFr(th,test_scores,n_te,&test_frr,&test_far,n_p_te,false);
		printf("HTER test when EER train       FAR %f   FRR %f   HTER %f\n", test_far,test_frr,(test_far+test_frr)/2.);
		//  printf("test: n_clients %d n_impostors %d classif_err %f\n",n_p_te,n_impostors,(n_impostors*test_far + n_p_te*test_frr)/n_te);
		th = computeHTER(train_scores,n_tr,&train_frr,&train_far,n_p_tr,false);
		computeFaFr(th,test_scores,n_te,&test_frr,&test_far,n_p_te,false);
		printf("HTER test when HTER min train  FAR %f   FRR %f   HTER %f\n", test_far,test_frr,(test_far+test_frr)/2.);
		//  printf("test: n_clients %d n_impostors %d classif_err %f\n",n_p_te,n_impostors,(n_impostors*test_far + n_p_te*test_frr)/n_te);
			real prior_average_far = 0;
			real prior_average_frr = 0;
			real prior_average_dcf = 0;
			real post_average = 0;
			real prior_class_err = 0;
			real post_class_err = 0;
			for (real r = step;r<bound;r += step) {
				th = computeThGivenFAR(test_scores,&ete,r,false);
				test_frr = ete.fn/(real)(ete.fn + ete.tp);
				post_average += test_frr*step;
				post_class_err += (ete.fn + ete.fp)*step;
				th = computeThGivenFRR(test_scores,&ete,r,false);
				post_class_err += (ete.fn + ete.fp)*step;
				th = computeThGivenFAR(train_scores,&etr,r,false);
				train_frr = etr.fn/(real)(etr.fn + etr.tp);
				compute4values(th,test_scores,&ete,false,true);
				test_far = ete.fp/(real)(ete.fp + ete.tn);
				test_frr = ete.fn/(real)(ete.fn + ete.tp);
				prior_average_far += 0.5*(test_frr + test_far)*step;
				prior_class_err += 0.5*(ete.fn + ete.fp)*step;
				th = computeThGivenFRR(train_scores,&etr,r,false);
				train_far = etr.fp/(real)(etr.fp + etr.tn);
				compute4values(th,test_scores,&ete,false,true);
				test_far = ete.fp/(real)(ete.fp + ete.tn);
				test_frr = ete.fn/(real)(ete.fn + ete.tp);
				prior_average_frr += 0.5*(test_frr + test_far)*step;
				prior_class_err += 0.5*(ete.fn + ete.fp)*step;
			}
			real the_bound = 1.;
			if (bound < 0.5) the_bound = bound;          
			else the_bound = 1 - bound;
			for (real r = the_bound + step; r < 1 - the_bound; r += step) {
				th = computeDCF(train_scores,&etr,r,false);
				compute4values(th,test_scores,&ete,false,true);
				test_far = ete.fp/(real)(ete.fp + ete.tn);
				test_frr = ete.fn/(real)(ete.fn + ete.tp);
				prior_average_dcf += 0.5*(test_frr + test_far)*step;
			}
			printf("---------------------------------------------------------------\n");
			post_average = 0.5 + post_average;
			prior_class_err /= ete.n;
			post_class_err /= ete.n;
			printf("(AUROCC + 1/2)/2 test  : %f\n",post_average*0.5);
			printf("E(HTER|FAR) train  : %f\n",prior_average_far);
			printf("E(HTER|FRR) train  : %f  \n", prior_average_frr);  
			printf("E(HTER|DCF) train  : %f  \n", prior_average_dcf);  
	}

  delete allocator;
  return(0);
}
