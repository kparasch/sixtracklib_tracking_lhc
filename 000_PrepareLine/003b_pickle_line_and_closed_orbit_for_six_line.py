import sixtracktools
from cpymad.madx import Madx
import pysixtrack
import pickle
import os

os.system('(cd sixtrack; ./runsix)')

import numpy as np

##############
# Build line #
##############

# Read sixtrack input
sixinput = sixtracktools.SixInput('./sixtrack')
p0c_eV = sixinput.initialconditions[-3]*1e6

# Build pysixtrack line from sixtrack input
line, other_data = pysixtrack.Line.from_sixinput(sixinput)


# Info on sixtrack->pyblep conversion 
iconv = other_data['iconv']


########################################################
#                  Search closed orbit                 #
# (for comparison purposes we the orbit from sixtrack) #
########################################################

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101('sixtrack/res/dump3.dat')
# Assume first particle to be on the closed orbit
Nele_st = len(iconv)
sixdump_CO = sixdump_all[::2][:Nele_st]

# Disable BB elements
line.disable_beambeam()

# Find closed orbit
guess_from_sixtrack = [getattr(sixdump_CO, att)[0]
         for att in 'x px y py sigma delta'.split()]
part_on_CO = line.find_closed_orbit(
        guess=guess_from_sixtrack, method='get_guess', p0c=p0c_eV)

print('Closed orbit at start machine:')
print('x px y py sigma delta:')
print(part_on_CO)

#######################################################
#  Store closed orbit and dipole kicks at BB elements #
#######################################################

line.beambeam_store_closed_orbit_and_dipolar_kicks(
        part_on_CO,
        separation_given_wrt_closed_orbit_4D = True,
        separation_given_wrt_closed_orbit_6D = True)


#################################
# Save machine in pyblep format #
#################################

with open('line_from_six_with_bbCO.pkl', 'wb') as fid:
    pickle.dump(line.to_dict(keepextra=True), fid)

##########################################
# Compute linear map around closed orbit #
##########################################

part = pysixtrack.Particles(p0c = p0c_eV)
pysixtrack_CO_bb, M = normalization.get_CO_and_linear_map(part, line, 1e-10, 1.e-10)

Ms = normalization.healy_symplectify(M)

W, invW, R = normalization.linear_normal_form(Ms)

#########################################
# Save particle on closed orbit as dict #
#########################################

with open('particle_on_CO_six_line.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

#########################################
# Save sixtrack->pyblep conversion info #
#########################################

with open('iconv.pkl', 'wb') as fid:
    pickle.dump(iconv, fid)

#########################################
# Save some optics as dict              #
#########################################
seq = 'lhcb1'

mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False

mad.call('lhcwbb_fortracking.seq')
mad.use(seq)
twiss_table = mad.twiss()

optics_dict = {'betx'      : twiss_table.betx[0],
               'bety'      : twiss_table.bety[0],
               'alfx'      : twiss_table.alfx[0],
               'alfy'      : twiss_table.alfy[0],
               'qx'        : mad.table['summ']['q1'][0],
               'qy'        : mad.table['summ']['q2'][0],
               'dqx'       : mad.table['summ']['dq1'][0],
               'dqy'       : mad.table['summ']['dq2'][0],
               'rf_volt'   : sum(twiss_table.volt),
               'rf_freq'   : max(twiss_table.freq),
               'rf_harmon' : max(twiss_table.harmon),
               'rf_lag'    : twiss_table.lag[twiss_table.harmon != 0][0],
               'length'    : twiss_table.s[-1],
               'alfa'      : twiss_table.alfa[0],
               'beta0'     : mad.sequence[seq].beam.beta,
               'gamma0'    : mad.sequence[seq].beam.gamma,
               'p0c_eV'    : mad.sequence[seq].beam.pc*1.e9,
               'M'         : M,
               'Ms'        : Ms,
               'W'         : W,
               'invW'      : invW,
               'R'         : R
              }

with open('optics_mad.pkl', 'wb') as fid:
    pickle.dump(optics_dict , fid)

