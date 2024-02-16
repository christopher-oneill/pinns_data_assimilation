import numpy as np
import matplotlib.pyplot as plot
import h5py
import sys
sys.path.append('C:/projects/pinns_local/code/') # add the library
from pinns_data_assimilation.lib.dft import dft


input_data_dir = 'I:/projects/fixed_cylinder/unstructured/data/'


velocity_file = h5py.File(input_data_dir+'rawField.mat','r')
#pressure_file = h5py.File(input_data_dir+'pressureField.mat','r')

velocityField = np.array(velocity_file['velocityField']).transpose()
#pressureField = np.array(pressure_file['pressureField']).transpose()

meanVelocity = np.mean(velocityField,1)
#meanPressure = np.mean(pressureField,1)

fluctuatingVelocity = velocityField - np.reshape(meanVelocity,[meanVelocity.shape[0],1,meanVelocity.shape[1]])
#fluctuatingPressure = pressureField - meanPressure


print('fluctuatingVelocity.shape: ',fluctuatingVelocity.shape)
velocityModes, f = dft(fluctuatingVelocity[70000:70002,:,:])
print('velocityModes.shape: ',velocityModes.shape)

half_index = int(velocityModes.shape[1]/2)

L = np.arange(4000,4100)
print('L: ',L.shape)
max_vals = np.zeros(L.shape)
for i in range(L.shape[0]):
    velocityModes, f = dft(fluctuatingVelocity[50000:50002,0:L[i],:])
    max_vals[i]= np.max(np.abs(velocityModes[0,0:half_index,0]))

ind_max = np.argmax(max_vals)
print('Max: ',max_vals[ind_max])
print('Length: ',L[ind_max])

# length to maximize the concentration of energy content: 4082

plot.figure(1)
plot.plot(f[0:half_index],np.abs(velocityModes[0,0:half_index,0])/np.sum(np.abs(velocityModes[0,0:half_index,0])))
plot.plot(f[0:half_index],np.abs(velocityModes[0,0:half_index,1])/np.sum(np.abs(velocityModes[0,0:half_index,1])))

plot.figure(2)
plot.plot(L,max_vals,'or')

plot.show()