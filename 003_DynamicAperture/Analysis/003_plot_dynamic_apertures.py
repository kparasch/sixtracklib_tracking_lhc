import analysis_dynamic_aperture as ada
import matplotlib.pyplot as plt

x_da1, y_da1 = ada.dynamic_aperture_contour('dynap_sixtracklib_turns_1000000_delta_0.00000.h5')
x_da2, y_da2 = ada.dynamic_aperture_contour('dynap_sixtracklib_turns_1000000_delta_0.00009.h5')

plt.plot(x_da1,y_da1,'r')
plt.plot(x_da2,y_da2,'r')

plt.show()
