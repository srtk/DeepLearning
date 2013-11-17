#!/bin/bash

OUTPUT="svm_cv_output_single_30"
SCALED="../data/scaled_single_30.txt"
PNG="${OUTPUT}/out.png"
LOG="${OUTPUT}/gr_result.txt"

PYTHON="/Users/hiroki/Virtualenvs/default-2/bin/python"
GNUPLOT="/usr/local/bin/gnuplot"
SVMTRAIN="/usr/local/bin/svm-train"
GRID="grid.py"

#LOG2C="-1,6,1"
#LOG2G="0,-8,-1"
#LOG2P="-8, -1, 1"
#LOG2C="-0.4,0.2,0.05"
#LOG2G="-9.2,-9.4,-0.05"
#LOG2P="-8,-2,3"
#LOG2C="0.4,1,0.2"
#LOG2G="-9.1,-9.7,-0.2"
LOG2C="0.6,1.0,0.2"
LOG2G="-9.3,-9.7,-0.2"

#svm-train -v 2 $SCALED

${PYTHON} $GRID -log2c ${LOG2C} -log2g ${LOG2G} -png $PNG -svmtrain $SVMTRAIN -gnuplot $GNUPLOT -v 3 -t 2 ${SCALED} |tee $LOG
#echo $PYTHON $GRID -log2c "$LOG2C" -log2g "$LOG2G" -svmtrain $SVMTRAIN -gnuplot $GNUPLOT -png $PNG -v 3 -t 2 ${SCALED}
