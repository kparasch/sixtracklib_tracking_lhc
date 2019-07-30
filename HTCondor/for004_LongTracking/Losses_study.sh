#!/bin/bash

uname -r 

export MYPYTHON=/afs/cern.ch/work/k/kparasch/public/sixtracklib-tracking/miniconda3

source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4cuda9/latest/x86_64-centos7-gcc62-opt/setup.sh
unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

which python

cd 004_LongTracking
python 000_longterm_tracking.py losses_sixtracklib.$2.$1 10000 $1
cd ..
