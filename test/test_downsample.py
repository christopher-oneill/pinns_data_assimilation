
import sys
import numpy as np
import h5py 
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.downsample import compute_downsample_inds_irregular

from pinns_data_assimilation.lib.downsample import boundary_inds_irregular

import matplotlib.pyplot as plot


base_dir = 'C:/projects/pinns_narval/sync/data/'
configFile_grid = h5py.File(base_dir+'/mazi_fixed_grid/configuration.mat','r')
configFile_irregular = h5py.File(base_dir+'/mazi_fixed/configuration.mat','r')


x = np.array(configFile_grid['X_vec'][0,:])
X_grid = np.array(configFile_grid['X_grid'])
y = np.array(configFile_grid['X_vec'][1,:])
Y_grid = np.array(configFile_grid['Y_grid'])
global d
d = np.array(configFile_grid['cylinderDiameter'])

X_irregular = np.array(configFile_irregular['X']).transpose()


dx = np.mean(np.diff(X_grid[:,0]))
dy = np.mean(np.diff(Y_grid[0,:]))

inds8 = compute_downsample_inds_irregular(8,X_irregular,d)
inds16 = compute_downsample_inds_irregular(16,X_irregular,d)
inds32 = compute_downsample_inds_irregular(32,X_irregular,d)
inds64 = compute_downsample_inds_irregular(64,X_irregular,d)

plot.figure(1)
plot.subplot(4,1,1)
plot.scatter(X_irregular[inds8,0],X_irregular[inds8,1],6,'k','o')
plot.subplot(4,1,2)
plot.scatter(X_irregular[inds16,0],X_irregular[inds16,1],6,'r','o')
plot.subplot(4,1,3)
plot.scatter(X_irregular[inds32,0],X_irregular[inds32,1],6,'g','o')
plot.subplot(4,1,4)
plot.scatter(X_irregular[inds64,0],X_irregular[inds64,1],6,'b','o')

Nx = 600
Ny = 200
x_x = np.linspace(-2.0,10.0,Nx)
y_xt = 2.0*np.ones([Nx,])
y_xb = -2.0*np.ones([Nx,])

# left side
x_y = -2.0*np.ones([Ny,])
y_y = np.linspace(-2.0,2.0,Ny)

# concatenate them all
X_t = np.concatenate((x_x,x_y,x_x),axis=0)
Y_t = np.concatenate((y_xb,y_y,y_xt),axis=0)

inds_boundary = boundary_inds_irregular(Nx,Ny,X_irregular)
print(inds_boundary.shape)

plot.figure(2)

plot.scatter(X_irregular[inds_boundary,0],X_irregular[inds_boundary,1],6,'k','o')
plot.scatter(X_t,Y_t,6,'r','o')
plot.show()