#!/bin/bash

echo "=========start simulation_parameters.py========="
tee ../../004_LongTracking/simulation_parameters.py <<EOF
device = 'opencl:0.$1'

output_filename = '/home/HPC/kparasch/20190715_sixtracklib/sixtracklib_tracking_lhc/CNAF/for004_LongTracking/data/losses_sixtracklib.$2.$1.h5'
line_filename = 'line_from_mad_with_bbCO.pkl'
CO_filename = 'particle_on_CO_mad_line.pkl'
optics_filename = 'optics_mad.pkl' 

seed = $2$1

n_turns = 20000000

n_particles = 10000
n_sigmas = 4 
init_delta_wrt_CO = 1.84e-4

disable_BB = False
simplify_line = True
EOF
echo "==========end simulation_parameters.py=========="

echo "uname -r:" `uname -r`

export MYPYTHON=/home/HPC/kparasch/20190715_sixtracklib/miniconda3
source $MYPYTHON/bin/activate

echo "which python:" `which python`

cd ../../004_LongTracking
python 000_longterm_tracking.py 
cd ../CNAF/for004_LongTracking
