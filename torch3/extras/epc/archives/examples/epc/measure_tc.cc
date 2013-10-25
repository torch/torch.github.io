const char *help = "\
(c) Trebolloc & Co 2001\n\
\n\
measure_tc\n";

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
  bool precision_criterion = false;
  bool recall_criterion = false;
  char* criterion;
  bool no_stats;
  char* roc_file;
  char* epc_file;
  bool F1;
  char* bep_file;
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
  cmd.addBCmdOption("-F1", &F1, false, "compute the F1 measure instead of (P+R)/2");
  cmd.addSCmdOption("-roc_file", &roc_file, "", "compute PRECISION/RECALL ROC curves");
  cmd.addSCmdOption("-epc_file", &epc_file, "", "use the PRECISION/RECALL DCF criterion");
  cmd.addSCmdOption("-criterion", &criterion, "dcf", "recall, precision, dcf values");
  cmd.addRCmdOption("-bound", &bound,1.,"bound for the criterion values");	

	cmd.addText("\nPoints:");
  cmd.addSCmdOption("-bep_file", &bep_file, "", "compute BEP points");


	cmd.addText("\nStats:");
  cmd.addBCmdOption("-no_stats", &no_stats, false, "compute precision/recall statistics");
  
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

  real train_precision,train_recall;
  real test_precision,test_recall;  
  real th = 0;
	if(!strcmp(criterion,"dcf")){
		dcf_criterion = true;
	}else
		if(!strcmp(criterion,"recall")){
			recall_criterion = true;
		}else
			if(!strcmp(criterion,"precision")){
				precision_criterion = true;
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
						th = computePR_DCF(train_scores,&etr,r,false);
						train_precision=etr.tp/(real)(etr.tp + etr.fp);
						train_recall=etr.tp/(real)(etr.tp + etr.fn);
						compute4values(th,test_scores,&ete,false,true);
						test_precision=ete.tp/(real)(ete.tp + ete.fp);
						test_recall=ete.tp/(real)(ete.tp + ete.fn);
						if(!F1){
							x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,0.5*train_precision+0.5*train_recall,test_precision,test_recall,0.5*test_precision+0.5*test_recall,r);
						}else{
							x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,2*train_precision*train_recall/(train_precision+train_recall),test_precision,test_recall,2*test_precision*test_recall/(test_precision+test_recall),r);
						}
					}
				} else if (precision_criterion) {				
					real_3* prec_scores = (real_3*)allocator->alloc(n_tr*sizeof(real_3));
					epcPrecision(step,bound,train_scores,test_scores,&etr,&ete,&x_epc,prec_scores,F1,false);
				} else if (recall_criterion) {
					for (real r = step;r<bound;r += step) {
						th = computeThGivenRecall(train_scores,&etr,r,false);
						train_precision= etr.tp/(real)(etr.tp + etr.fp);
						train_recall=etr.tp/(real)(etr.tp + etr.fn);
						compute4values(th,test_scores,&ete,false,false);
						test_precision=ete.tp/(real)(ete.tp + ete.fp);
						test_recall=ete.tp/(real)(ete.tp + ete.fn);
						if(!F1){
							x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,0.5*train_precision+0.5*train_recall,test_precision,test_recall,0.5*test_precision+0.5*test_recall,r);
						}else{
							x_epc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,2*train_precision*train_recall/(train_precision+train_recall),test_precision,test_recall,2*test_precision*test_recall/(test_precision+test_recall),r);
						}
					}
				} 
			}

			/*************** DET an ROC (Warning a posteriori measure) ************/
			DiskXFile* x_roc = NULL;	
			if(strcmp(roc_file, ""))
				x_roc = new(allocator)DiskXFile(roc_file,"w");

			if(strcmp(roc_file, "")){
				DiskXFile x_roc(roc_file,"w");
				warning("You are using an a posteriori measure!!!");
				bound = 0.5;
				for (real r = step;r<bound;r += step) {
					th = computeThGivenRecall(train_scores,&etr,r,false);
					train_precision=etr.tp/(real)(etr.tp + etr.fp);
					train_recall=etr.tp/(real)(etr.tp + etr.fn);
					th = computeThGivenRecall(test_scores,&ete,r,false);
					test_precision=ete.tp/(real)(ete.tp + ete.fp);
					test_recall=ete.tp/(real)(ete.tp + ete.fn);
					if(!F1)
						x_roc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,0.5*train_precision+0.5*train_recall,test_precision,test_recall,0.5*test_precision+0.5*test_recall,r);
					else
						x_roc.printf("%f   %f   %f   %f   %f   %f   %f\n",train_precision,train_recall,2*train_precision*train_recall/(train_precision+train_recall),test_precision,test_recall,2*test_precision*test_recall/(test_precision+test_recall),r);
				}
			}


  // breakeven point ?
	//
			if(strcmp(bep_file, "")){
				DiskXFile x_bep(bep_file,"w");
				th = computeBEP(train_scores,&etr,false);
				compute4values(th,test_scores,&ete,false,true);
				test_precision = ete.tp/(real)(ete.tp + ete.fp);
				test_recall = ete.tp/(real)(ete.tp + ete.fn);
				x_bep.printf("%f %f",test_precision,test_recall);
			}
  
	
	// compute stats

  
	if (!no_stats) {
    th = computeBEP(test_scores,&ete,false);
    compute4values(th,test_scores,&ete,false,true);
    test_precision = ete.tp/(real)(ete.tp + ete.fp);
    test_recall = ete.tp/(real)(ete.tp + ete.fn);
	 if(!F1)
		 printf("Perf test when BEP test    Precision %f   Recall %f   P+R %f\n", test_precision,test_recall,test_precision+test_recall);
	 else
		 printf("Perf test when BEP test    Precision %f   Recall %f   F1 %f\n", test_precision,test_recall,2*test_precision*test_recall/(test_precision+test_recall));
    th = computeBEP(train_scores,&etr,false);
    compute4values(th,test_scores,&ete,false,true);
    test_precision = ete.tp/(real)(ete.tp + ete.fp);
    test_recall = ete.tp/(real)(ete.tp + ete.fn);
	 if(!F1)
		 printf("Perf test when BEP train   Precision %f   Recall %f   P+R %f\n", test_precision,test_recall,test_precision+test_recall);
	 else
		 printf("Perf test when BEP train   Precision %f   Recall %f   F1 %f\n", test_precision,test_recall,2*test_precision*test_recall/(test_precision+test_recall));
			
	 real prior_average_P = 0; 
	 real prior_average_R = 0;
	 real prior_average_PR_DCF = 0;
	 real post_average = 0;
	 for (real r = step;r<bound;r += step) {
		 th = computeThGivenRecall(test_scores,&ete,r,false);
		 test_precision = ete.tp/(real)(ete.tp + ete.fp);
		 post_average += test_precision*step; 
		 th = computeThGivenRecall(train_scores,&etr,r,false);
		 compute4values(th,test_scores,&ete,false,false);
		 test_precision = ete.tp/(real)(ete.tp + ete.fp);
		 test_recall = ete.tp/(real)(ete.tp + ete.fn);
		 prior_average_R += 0.5*(test_precision + test_recall)*step; 
	 }
	 real the_bound = 1.;
	 if (bound < 0.5) the_bound = bound;          
	 else the_bound = 1 - bound;
	 for (real r = the_bound + step; r < 1 - the_bound; r += step) {			 
		 th = computePR_DCF(train_scores,&etr,r,false);
		 compute4values(th,test_scores,&ete,false,true);
		 test_precision=ete.tp/(real)(ete.tp + ete.fp);
		 test_recall=ete.tp/(real)(ete.tp + ete.fn);
		 prior_average_PR_DCF += test_precision + test_recall;
	 }
	 real_3* prec_scores = (real_3*)allocator->alloc(n_tr*sizeof(real_3));
	 real sum = epcPrecision(step,bound,train_scores,test_scores,&etr,&ete,NULL,prec_scores,F1,true);
	 prior_average_P += sum;
	 printf("Perf test :(AUROCC + 1/2)/2 test  : %f\n", (post_average + 0.5)*0.5);
	 printf("Perf test : E((P+R)/2|P)  : %f\n", prior_average_P*step*0.5);
	 printf("Perf test : E((P+R)/2|R)  : %f\n", prior_average_R);
	 printf("Perf test : E((P+R)/2|DCF)  : %f\n", prior_average_PR_DCF*step*0.5);  
	}

  delete allocator;
  return(0);
}
