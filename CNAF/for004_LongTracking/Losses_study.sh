#!/bin/bash

#scl enable devtoolset-7 bash
uname -r 

export MYPYTHON=/home/HPC/kparasch/20190715_sixtracklib/miniconda3
source $MYPYTHON/bin/activate

which python

cd ../../004_LongTracking
python 000_longterm_tracking.py /home/HPC/kparasch/20190715_sixtracklib/sixtracklib_tracking_lhc/CNAF/for004_LongTracking/data/losses_sixtracklib.$2.$1 20000000 0 $2$1 opencl:0.$1
cd ../CNAF/for004_LongTracking
