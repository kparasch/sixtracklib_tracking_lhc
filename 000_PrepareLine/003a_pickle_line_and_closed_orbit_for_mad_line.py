from cpymad.madx import Madx
import pysixtrack
import pickle
import sys
sys.path.append('..')

import normalization

########################################################
#                  Search closed orbit                 #
########################################################
seq = 'lhcb1'

mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False

mad.call('lhcwbb_fortracking.seq')
mad.use(seq)

beta0 = mad.sequence[seq].beam.beta
gamma0 = mad.sequence[seq].beam.gamma
p0c_eV = mad.sequence[seq].beam.pc*1.e9

twiss_table = mad.twiss()
x_CO  = twiss_table.x[0]
px_CO = twiss_table.px[0]
y_CO  = twiss_table.y[0]
py_CO = twiss_table.py[0]
t_CO  = twiss_table.t[0]
pt_CO = twiss_table.pt[0]


#convert tau, pt to sigma,delta
sigma_CO = beta0 * t_CO
delta_CO = ((pt_CO**2 + 2*pt_CO/beta0) + 1)**0.5 - 1

mad_CO = [x_CO, px_CO, y_CO, py_CO, sigma_CO, delta_CO]

# Load line
with open('line_from_mad.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

# Disable BB elements
line.disable_beambeam()

#Put closed orbit
part_on_CO = line.find_closed_orbit(
        guess=mad_CO, method='get_guess', p0c=p0c_eV)

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

with open('line_from_mad_with_bbCO.pkl', 'wb') as fid:
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

with open('particle_on_CO_mad_line.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

#########################################
# Save some optics as dict              #
#########################################

optics_dict = {'betx'      : twiss_table.betx[0],
               'bety'      : twiss_table.bety[0],
               'alfx'      : twiss_table.alfx[0],
               'alfy'      : twiss_table.alfy[0],
               'disp_x'    : twiss_table.dx[0]/mad.sequence[seq].beam.beta,
               'disp_px'   : twiss_table.dpx[0]/mad.sequence[seq].beam.beta,
               'disp_y'    : twiss_table.dy[0]/mad.sequence[seq].beam.beta,
               'disp_py'   : twiss_table.dpy[0]/mad.sequence[seq].beam.beta,
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

