import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')


figures_dir = 'C:/projects/paper_figures/reconstruction/'
rec_dir = 'C:/projects/paper_figures/data/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'

base_dir=data_dir
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')



x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

X_grid_plot = X_grid
Y_grid_plot = Y_grid
X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)


ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

uxi = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uyi = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()

uxi = uxi+np.reshape(ux,[ux.shape[0],1])
uyi = uyi+np.reshape(uy,[uy.shape[0],1])

uxi = np.reshape(uxi,[X_grid.shape[0],X_grid.shape[1],uxi.shape[1]])
L_dft = 4082

reconstructedVelocityFile =  h5py.File(rec_dir+'reconstruction_S0_c0.h5','r')
uxr = np.reshape(np.array(reconstructedVelocityFile['ux']),[uxi.shape[0],uxi.shape[1],L_dft])

plot.figure(1)
plot.subplot(2,1,1)
plot.contourf(X_grid,Y_grid,uxi[:,:,2000],cmap= matplotlib.colormaps['bwr'])
plot.subplot(2,1,2)
plot.contourf(X_grid,Y_grid,uxr[:,:,2000],cmap= matplotlib.colormaps['bwr'])
plot.show()