
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plot
import matplotlib

HOMEDIR = 'C:/projects/pinns_narval/sync/'
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.decomposition import POD
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

base_dir = HOMEDIR+'/data/mazi_fixed_grid/'


meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')


x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')

um_x = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
um_y = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
pm = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

p_t  = np.array(fluctuatingPressureFile['fluctuatingPressure']).transpose()
p_t = p_t + np.reshape(pm,[p_t.shape[0],1])
t = np.array(configFile['time']).transpose()

u_t = np.array(fluctuatingVelocityFile['fluctuatingVelocity']).transpose()

supersample_factors = np.array([0,2,4,8,16,32])
supersample_labels = ['S*=1','S*=2','S*=4','S*=8','S*=16','S*=32']

Phi_list = []
Lambda_list = []
Ak_list = []
x_ds_list = []
y_ds_list = []
um_x_ds_list = []
um_y_ds_list = []
ndx_list = []
ndy_list =[]
X_grid_ds_list = []
Y_grid_ds_list = []



for supersample_factor in supersample_factors:
    if supersample_factor>0:
        downsample_inds, ndx,ndy = compute_downsample_inds_center(supersample_factor,X_grid[:,0],Y_grid[0,:].transpose())
        x_ds = x[downsample_inds]
        y_ds = y[downsample_inds]
        X_grid_ds = np.reshape(x_ds,[ndy,ndx]).transpose()
        Y_grid_ds = np.reshape(y_ds,[ndy,ndx]).transpose()

        um_x_ds = um_x[downsample_inds]
        um_y_ds = um_y[downsample_inds]
        u_t_ds = u_t[downsample_inds,:]
    else:
        ndx = X_grid.shape[0]
        ndy = X_grid.shape[1]
        X_grid_ds = np.reshape(x,[ndx,ndy])
        Y_grid_ds = np.reshape(y,[ndx,ndy])

        x_ds = x
        y_ds = y
        um_x_ds = um_x
        um_y_ds = um_y
        u_t_ds = u_t 

    # do the POD
    Phi, Lambda, Ak = POD(u_t_ds)
    
    Phi_list.append(Phi[:,0:10])
    Lambda_list.append(Lambda[0:10])
    Ak_list.append(Ak[:,0:10])
    ndx_list.append(ndx)
    ndy_list.append(ndy)
    x_ds_list.append(x_ds)
    y_ds_list.append(y_ds)
    X_grid_ds_list.append(X_grid_ds)
    Y_grid_ds_list.append(Y_grid_ds)

fig_save_loc = 'C:/projects/paper_figures/POD/'
# plot
plot_modes = 3

dx0 =  X_grid_ds_list[0][1,0] - X_grid_ds_list[0][0,0]
dy0 =  Y_grid_ds_list[0][0,1] - Y_grid_ds_list[0][0,0]
Nx0 = ndx_list[0]*ndy_list[0]
A0 = Nx0*dx0*dy0
print('A0:',A0)
for m in range(0,2*plot_modes,2):
    plot.figure(1)
    for s in range(len(supersample_factors)):
        dx =  X_grid_ds_list[s][1,0] - X_grid_ds_list[s][0,0]
        dy =  Y_grid_ds_list[s][0,1] - Y_grid_ds_list[s][0,0]
        #print('S=',s)
        print(X_grid_ds_list[s][0:2,0:2])
        print('dx',dx)
        print(Y_grid_ds_list[s][0:2,0:2])
        print('dy',dy)
        Nx = ndx_list[s]*ndy_list[s]
        Ai = (Nx*dx*dy) # /(Lambda_list[s][m])
        print('Ai',Ai)
        plot.plot(t-t[0],Ak_list[s][:,m]*np.sqrt((Nx0/Nx))*(A0/Ai),linewidth=2)
    plot.xlim(0,5)
    plot.legend(supersample_labels)
    plot.title('Ak='+str(m))
    plot.savefig(fig_save_loc+'POD_aliasing_Ak'+str(m)+'.png',dpi=600)
    plot.close(1)

    plot.figure(1)
    for s in range(len(supersample_factors)):
        ndx = ndx_list[s]
        ndy = ndy_list[s]
        plot.subplot(len(supersample_factors),1,1+s)
        if s>0:
            plot.contourf(X_grid_ds_list[s],Y_grid_ds_list[s],np.reshape(Phi_list[s][0:ndx*ndy,m],[ndy,ndx]).transpose(),levels=21,norm=matplotlib.colors.CenteredNorm(),cmap= matplotlib.colormaps['bwr'])
        else:
            plot.contourf(X_grid_ds_list[s],Y_grid_ds_list[s],np.reshape(Phi_list[s][0:ndx*ndy,m],[ndx,ndy]),levels=21,norm=matplotlib.colors.CenteredNorm(),cmap= matplotlib.colormaps['bwr'])
        plot.colorbar()
    plot.savefig(fig_save_loc+'POD_spatial_modes'+str(m)+'.png',dpi=600)
    plot.close(1)
    