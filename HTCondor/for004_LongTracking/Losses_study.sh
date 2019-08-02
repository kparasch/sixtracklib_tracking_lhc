#!/bin/bash

echo "=========start simulation_parameters.py========="
tee 004_LongTracking/simulation_parameters.py <<EOF
device = 'opencl:0.0'

output_filename = 'losses_sixtracklib.$2.$1.h5'
line_filename = 'line_from_mad_with_bbCO.pkl'
CO_filename = 'particle_on_CO_mad_line.pkl'
optics_filename = 'optics_mad.pkl' 

seed = $2$1

n_turns = 20000000

n_particles = 10000
n_sigmas = 4 
init_delta_wrt_CO = 9.7e-5

disable_BB = False
simplify_line = True
EOF
echo "==========end simulation_parameters.py=========="


echo "uname -r:" `uname -r`

export MYPYTHON=/afs/cern.ch/work/k/kparasch/public/sixtracklib-tracking/miniconda3

source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4cuda9/latest/x86_64-centos7-gcc62-opt/setup.sh
unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate ""
export PATH=$MYPYTHON/bin:$PATH

echo "which python:" `which python`

cd 004_LongTracking
python 000_longterm_tracking.py
cd ..
