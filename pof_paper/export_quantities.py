
import numpy as np
import h5py
import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/') # add the library
from pinns_data_assimilation.lib.dft import dft

fs=10 # needed for the frequency vector of the fourier modes
L_dft = 4082 # from fourier_length_analysis.py

input_data_dir = 'I:/projects/fixed_cylinder/unstructured/data/'
output_data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed/'

config_file = h5py.File(input_data_dir+'configuration.mat','r')




velocity_file = h5py.File(input_data_dir+'rawField.mat','r')
pressure_file = h5py.File(input_data_dir+'pressureField.mat','r')

velocityField = np.array(velocity_file['velocityField']).transpose()
pressureField = np.array(pressure_file['pressureField']).transpose()

# truncate per L_dft
velocityField = velocityField[:,0:L_dft,:]
pressureField = pressureField[:,0:L_dft]


meanVelocity = np.mean(velocityField,1)
h5f = h5py.File(output_data_dir+'meanVelocity.mat','w')
h5f.create_dataset('meanVelocity',data=meanVelocity)
h5f.close()

meanPressure = np.mean(pressureField,1)
h5f = h5py.File(output_data_dir+'meanPressure.mat','w')
h5f.create_dataset('meanPressure',data=meanPressure)
h5f.close()
print('Exported mean velocity and pressure fields. Computing stresses...')

fluctuatingVelocity = velocityField - np.reshape(meanVelocity,[meanVelocity.shape[0],1,meanVelocity.shape[1]])
fluctuatingPressure = pressureField - np.reshape(meanPressure,[meanPressure.shape[0],1])

print(np.max(np.abs(fluctuatingVelocity[:,:,0]),(0,1)))
print(np.max(np.abs(fluctuatingPressure),(0,1)))

# compute stresses
fluctuatingStress = np.zeros([fluctuatingVelocity.shape[0],fluctuatingVelocity.shape[1],3])
fluctuatingStress[:,:,0] = fluctuatingVelocity[:,:,0]*fluctuatingVelocity[:,:,0]
fluctuatingStress[:,:,1] = fluctuatingVelocity[:,:,0]*fluctuatingVelocity[:,:,1]
fluctuatingStress[:,:,2] = fluctuatingVelocity[:,:,1]*fluctuatingVelocity[:,:,1]

reynoldsStress = np.mean(fluctuatingStress,1)
h5f = h5py.File(output_data_dir+'reynoldsStress.mat','w')
h5f.create_dataset('reynoldsStress',data=reynoldsStress)
h5f.close()

print('Computing fourier modes....')
# compute fourier modes

velocityModes,modeFrequencies = dft(fluctuatingVelocity)
pressureModes,f_pressure = dft(fluctuatingPressure)
stressModes,f_stress = dft(fluctuatingStress)

print(np.max(np.abs(velocityModes[:,:,0]),(0,1)))
print(np.max(np.abs(pressureModes),(0,1)))

# keep single sided spectrum, divide by L_DFT to normalize
half_index = int(L_dft/2)
modeFrequencies = modeFrequencies[0:half_index]
velocityModes = velocityModes[:,0:half_index,:]/L_dft
pressureModes = pressureModes[:,0:half_index]/L_dft
stressModes = stressModes[:,0:half_index,:]/L_dft

# scale the non-dimensional frequncies by the sampling rate
modeFrequencies = modeFrequencies*fs

plot.figure(1)
plot.plot(np.abs(velocityModes[70000,:,0]))
plot.yscale('log')
plot.show()

# extract the 4 most energetic modes for saving
inds_max = np.argsort(np.abs(velocityModes[70000,:,0]))[::-1]
inds_max = inds_max[0:8]
print('Energetic modes: ',inds_max)
print("Mode peaks:")
print(np.abs(velocityModes[70000,inds_max,0])/np.sum(np.abs(velocityModes[70000,:,0])))

modeFrequencies = modeFrequencies[inds_max]
print("Mode frequencies: ",modeFrequencies)

velocityModes = velocityModes[:,inds_max,:]
pressureModes = pressureModes[:,inds_max]
stressModes = stressModes[:,inds_max,:]

h5f = h5py.File(output_data_dir+'fourierModes.mat','w')
h5f.create_dataset('modeFrequencies',data=modeFrequencies)
h5f.create_dataset('velocityModes',data=velocityModes)
h5f.create_dataset('pressureModes',data=pressureModes)
h5f.create_dataset('stressModes',data=stressModes)
h5f.close()


















