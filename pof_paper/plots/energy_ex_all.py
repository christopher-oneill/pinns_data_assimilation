
import numpy as np
import h5py
import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/') # add the library
from pinns_data_assimilation.lib.dft import dft

fs=10 # needed for the frequency vector of the fourier modes
L_dft = 4082 # from fourier_length_analysis.py

input_data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'

configFile = h5py.File(input_data_dir+'configuration.mat','r')

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

inputMeanVelocityFile = h5py.File(input_data_dir+'meanVelocity.mat','r')
velocity_file = h5py.File(input_data_dir+'fluctuatingVelocity.mat','r')


inputMeanVelocityField = np.array(inputMeanVelocityFile['meanVelocity']).transpose()
inputFluctuatingVelocityField = np.array(velocity_file['fluctuatingVelocity']).transpose()

velocityField = inputFluctuatingVelocityField + np.reshape(inputMeanVelocityField,[inputFluctuatingVelocityField.shape[0],1,inputFluctuatingVelocityField.shape[2]])
# truncate per L_dft
velocityField = velocityField[:,0:L_dft,:]

meanVelocity = np.mean(velocityField,1)
print('Exported mean velocity and pressure fields. Computing stresses...')

fluctuatingVelocity = velocityField - np.reshape(meanVelocity,[meanVelocity.shape[0],1,meanVelocity.shape[1]])

print('Computing fourier modes....')
# compute fourier modes

velocityModes,modeFrequencies = dft(fluctuatingVelocity)

print(np.max(np.abs(velocityModes[:,:,0]),(0,1)))

# keep single sided spectrum, divide by L_DFT to normalize
half_index = int(L_dft/2)
modeFrequencies = modeFrequencies[0:half_index]
phi = velocityModes[:,0:half_index,:]/L_dft

# scale the non-dimensional frequncies by the sampling rate
modeFrequencies = modeFrequencies*fs

# add back the mean field to the DFT modes
phi[:,0,0] = meanVelocity[:,0]+0j
phi[:,0,1] = meanVelocity[:,1]+0j

phi = np.reshape(phi,[X_grid.shape[0],X_grid.shape[1],half_index,2])

E_ij_all = np.zeros([L_dft,L_dft],dtype=np.complex128)

for i in range(-half_index+1,half_index):
    for j in range(-half_index+1,half_index):
            if (i-j)>=half_index or (i-j)<=-half_index:
                # the transfer mode is not in the nyquist
                E_ij_all[i+half_index,j+half_index]= np.NaN
            else:
                if i>0:
                    phi_i = np.conj(phi[:,:,i,:]) # use phi^* on mode 1
                else:
                    phi_i = phi[:,:,abs(i),:] # use phi^* ^* = phi

                if (i-j)<0:
                    phi_ij = np.conj(phi[:,:,abs(i-j),:])
                else:
                    phi_ij = phi[:,:,i-j,:]

                if j<0:
                    phi_j_x = np.gradient(np.conj(phi[:,:,abs(j),:]),X_grid[:,0],axis=0) # if less than zero use the conj
                    phi_j_y = np.gradient(np.conj(phi[:,:,abs(j),:]),Y_grid[0,:],axis=1)
                else:
                    phi_j_x = np.gradient(phi[:,:,j,:],X_grid[:,0],axis=0)
                    phi_j_y = np.gradient(phi[:,:,j,:],Y_grid[0,:],axis=1)

                e_ij_1 = phi_i[:,:,0] * phi_ij[:,:,0] * phi_j_x[:,:,0]
                e_ij_2 = phi_i[:,:,0] * phi_ij[:,:,1] * phi_j_y[:,:,0]
                e_ij_3 = phi_i[:,:,1] * phi_ij[:,:,0] * phi_j_x[:,:,1]
                e_ij_4 = phi_i[:,:,1] * phi_ij[:,:,1] * phi_j_y[:,:,1]

                e_ij = e_ij_1 + e_ij_2 + e_ij_3 + e_ij_4

                e_ij[cylinder_mask] = 0+0j

                t1r = np.trapz(np.real(e_ij),X_grid[:,0],axis=0)
                t1i = np.trapz(np.imag(e_ij),X_grid[:,0],axis=0)
                t2r = np.trapz(t1r,Y_grid[0,:],axis=0)
                t2i = np.trapz(t1i,Y_grid[0,:],axis=0)

                E_ij_all[i+half_index,j+half_index] = t2r+t2i*1j

plot_frequencies = np.concatenate(modeFrequencies[1::-1],modeFrequencies)
plot_frequencies_grid_i,plot_frequencies_grid_j = np.meshgrid(plot_frequencies,plot_frequencies)


figures_dir = 'C:/projects/paper_figures/energy_exchange/'
plot.figure(1)
plot.contourf(plot_frequencies_grid_i,plot_frequencies_grid_j,np.real(E_ij_all))
filename = 'energy_exchange_real_all'
plot.savefig(figures_dir+filename+'.pdf')
plot.savefig(figures_dir+filename+'.png',dpi=300)
plot.close(1)

plot.figure(1)
plot.contourf(plot_frequencies_grid_i,plot_frequencies_grid_j,np.imag(E_ij_all))
filename = 'energy_exchange_imag_all'
plot.savefig(figures_dir+filename+'.pdf')
plot.savefig(figures_dir+filename+'.png',dpi=300)