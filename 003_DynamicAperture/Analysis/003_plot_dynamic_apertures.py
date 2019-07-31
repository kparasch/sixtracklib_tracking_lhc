import analysis_dynamic_aperture as ada
import matplotlib.pyplot as plt

############## boundary x vs. y ##############

f1 = plt.figure(1)

x_da1, y_da1 = ada.dynamic_aperture_contour('dynap_sixtracklib_turns_1000000_delta_0.00000.h5')
x_da2, y_da2 = ada.dynamic_aperture_contour('dynap_sixtracklib_turns_1000000_delta_0.00009.h5')

plt.plot(x_da1,y_da1,'r', label = r'$\delta_0 =$ %r' %delta)
plt.plot(x_da2,y_da2,'r', label = r'$\delta_0 =$ %r' %delta)

plt.xlabel(r'$x [\sigma_x]$')
plt.ylabel(r'$y [\sigma_y]$')

plt.title('Dynamic Aperture Analysis, $\delta_{0} =$ %r' % delta)

plt.legend(loc='upper right')

plt.show()

############## boundary r vs. theta #############

f2 = plt.figure(2)

r_da1 = np.sqrt( np.multiply(x_da1, x_da1) + np.multiply(y_da1, y_da1) )
theta_da1 = np.arctan(np.divide(y_da1, x_da1))

############## which particles are lost x vs. y #############
#
#f3 = plt.figure(3)
#x_lost = mydict['xy_norm'][i_lost, j_lost, 0]
#y_lost = mydict['xy_norm'][i_lost, j_lost, 1]
#
#x_kept = mydict['xy_norm'][i_kept, j_kept, 0]
#y_kept = mydict['xy_norm'][i_kept, j_kept, 1]
#
#
#plt.plot(Ax, Ay, 'o', color='blue', label = 'tracked particles')
#plt.plot(x_lost, y_lost, '.', color='red', label = 'lost particles')
#
#plt.xlabel(r'$x [\sigma_x]$')
#plt.ylabel(r'$y [\sigma_y]$')
#
#plt.title('Lost particles after %r  turns' % nturns)
#
#plt.legend(loc='upper right')

plt.style.use('kostas')

plt.show()

