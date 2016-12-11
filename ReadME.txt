Project Submisssion - Image Classification 

 By : Yanyu Liang (yanyul@andrew.cmu.edu) , Priyanka Raja (priyankj@andrew.cmu.edu)

The report for the project is "project_report.pdf"

In total we have three base learners. We adopted the SVM function from the libSVM library(https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and used it as one of our base learner. We have also implemented two other algorithms as shown below:
1) sparse Logistic Regression
2) sparse Neural Networks


The code for these three algorithms is available in the Code/.. folder, please run the following code files to generate results:

1) svm_simple.m , svm_moderate.m and svm_difficult.m are the matlab files to run the one-vs-one SVM classifier.
2) sparseLogit_simple.m, sparseLogit_moderate.m, sparseLogit_difficult.m are the matlab files to run the sparse Logistic Regression       classifier 
3) sparseNN_simple.m , sparseNN_moderate.m, and sparseNN_difficult.m are the matlab files for the sparseNN algorithm.  

We implemented the uncertainity sampling algorithm as our active learning algorithm and the code is also available at Code/uncertainity_based_learner.m .

The blinded predictions obtained by our algorithms can be found in the Output/.. folder and are as follows:
1)   svm_simple.csv , svm_moderate.csv and svm_difficult.csv
2)   sparseLogit_simple.csv, sparseLogit_moderate.csv, sparseLogit_difficult.csv
3)   sparseNN_simple.csv , sparseNN_moderate.csv, and sparseNN_difficult.csv
