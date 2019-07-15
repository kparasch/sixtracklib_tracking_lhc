from cpymad.madx import Madx
import pysixtrack
import pickle

########################################################
#                  Search closed orbit                 #
########################################################


mad = Madx()
#mad.call('lhc_as-built.seq')
mad.options.echo = False
mad.options.warn = False
mad.options.info = False

mad.call('lhcwbb_fortracking.seq')
mad.use('lhcb1')

beta0 = mad.sequence['lhcb1'].beam.beta
gamma0 = mad.sequence['lhcb1'].beam.gamma
p0c_eV = mad.sequence['lhcb1'].beam.pc*1.e9

twiss_table = mad.twiss()
x_CO  = twiss_table.x[0]
px_CO = twiss_table.px[0]
y_CO  = twiss_table.y[0]
py_CO = twiss_table.py[0]
t_CO  = twiss_table.t[0]
pt_CO = twiss_table.pt[0]

#mad.input('ptc_create_universe;')
#mad.input('ptc_create_layout, time=true;')
#mad.input('ptc_twiss, icase=6, closed_orbit=true, summary_file=ptc_twiss;')
#
#f = open('ptc_twiss')
#lines = f.readlines()
#f.close()
#twiss_summary = lines[-1].strip().split()
#x_CO  = float(twiss_summary[43])
#px_CO = float(twiss_summary[44])
#y_CO  = float(twiss_summary[45])
#py_CO = float(twiss_summary[46])
#pt_CO  = float(twiss_summary[47])
#t_CO = float(twiss_summary[48])

#convert tau, pt to sigma,delta
sigma_CO = beta0 * t_CO
delta_CO = ((pt_CO**2 + 2*pt_CO/beta0) + 1)**0.5 - 1

mad_PTC_CO = [x_CO, px_CO, y_CO, py_CO, sigma_CO, delta_CO]

# Load line
with open('line_from_mad.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

# Disable BB elements
line.disable_beambeam()

#Put closed orbit
part_on_CO = line.find_closed_orbit(
        guess=mad_PTC_CO, method='get_guess', p0c=p0c_eV)

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
  
#########################################
# Save particle on closed orbit as dict #
#########################################

with open('particle_on_CO_mad_line.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

