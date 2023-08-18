

import numpy as np
import scipy.io
from scipy import interpolate
from scipy.interpolate import griddata
import h5py
import os
import glob
import sys
import re
import smt
import h5py
from smt.sampling_methods import LHS
from pyDOE import lhs
from datetime import datetime
from datetime import timedelta
import platform

node_name = platform.node()

LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    import matplotlib.colors as mplcolors
    useGPU=False    
    HOMEDIR = 'C:/projects/pinns_narval/sync/'
    sys.path.append('C:/projects/pinns_local/code/')

from pinns_galerkin_viv.lib import galerkin
from pinns_galerkin_viv.lib import file_util

# read the data

base_dir = HOMEDIR+'/data/mazi_fixed_grid/'
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
POD_dataFile = h5py.File(base_dir+'POD_data.mat','r')

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()
n_x = int(np.array(POD_dataFile['n_x']))

phi_x = np.array(POD_dataFile['Phi_ext'][:,0:n_x]).transpose()
phi_y = np.array(POD_dataFile['Phi_ext'][:,n_x:2*n_x]).transpose()
psi = np.array(POD_dataFile['Phi_ext'][:,2*n_x:3*n_x]).transpose()
n_trunc = int(np.array(POD_dataFile['Phi']).shape[0])

Ak = np.array(POD_dataFile['Ak']).transpose()

print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
y = np.array(configFile['X_vec'][1,:])
d = np.array(configFile['cylinderDiameter'])
X_grid = np.array(configFile['X_grid'])
Y_grid = np.array(configFile['Y_grid'])

nu_mol = 0.0066667

# send quantities to (nx,ny) grid
ux_grid = np.reshape(ux,X_grid.shape)
uy_grid = np.reshape(uy,X_grid.shape)
p_grid = np.reshape(p,X_grid.shape)
phi_x_grid = np.reshape(phi_x,np.array([X_grid.shape[0],X_grid.shape[1],n_trunc]))
phi_y_grid = np.reshape(phi_y,np.array([X_grid.shape[0],X_grid.shape[1],n_trunc]))
psi_grid = np.reshape(psi,np.array([X_grid.shape[0],X_grid.shape[1],n_trunc]))

# compute numerical derivatives on the grid
ux_x_grid,ux_y_grid,ux_xx_grid,ux_yy_grid,uy_x_grid,uy_y_grid,uy_xx_grid,uy_yy_grid,p_x_grid,p_y_grid,phi_x_x_grid,phi_x_y_grid,phi_x_xx_grid,phi_x_yy_grid,phi_y_x_grid,phi_y_y_grid,phi_y_xx_grid,phi_y_yy_grid,psi_x_grid,psi_y_grid = galerkin.galerkin_gradients(X_grid[:,0],Y_grid[0,:],ux_grid,uy_grid,p_grid,phi_x_grid,phi_y_grid,psi_grid)

# send derivatives to (nx*ny) vector
ux_x = np.reshape(ux_x_grid,ux.shape)
ux_y = np.reshape(ux_y_grid,ux.shape)
ux_xx = np.reshape(ux_xx_grid,ux.shape)
ux_yy = np.reshape(ux_yy_grid,ux.shape)

uy_x = np.reshape(uy_x_grid,ux.shape)
uy_y = np.reshape(uy_y_grid,ux.shape)
uy_xx = np.reshape(uy_xx_grid,ux.shape)
uy_yy = np.reshape(uy_yy_grid,ux.shape)

p_x = np.reshape(p_x_grid,ux.shape)
p_y = np.reshape(p_y_grid,ux.shape)

phi_x_x = np.reshape(phi_x_x_grid,phi_x.shape)
phi_x_y = np.reshape(phi_x_x_grid,phi_x.shape)
phi_x_xx = np.reshape(phi_x_xx_grid,phi_x.shape)
phi_x_yy = np.reshape(phi_x_yy_grid,phi_x.shape)

phi_y_x = np.reshape(phi_y_x_grid,phi_x.shape)
phi_y_y = np.reshape(phi_y_y_grid,phi_x.shape)
phi_y_xx = np.reshape(phi_y_xx_grid,phi_x.shape)
phi_y_yy = np.reshape(phi_y_yy_grid,phi_x.shape)

psi_x = np.reshape(psi_x_grid,phi_x.shape)
psi_y = np.reshape(psi_y_grid,phi_x.shape)

# compute galerkin projection

Fk, Lkl, Qklm = galerkin.galerkin_projection(nu_mol,ux,ux_x,ux_y,ux_xx,ux_yy,uy,uy_x,uy_y,uy_xx,uy_yy,p,p_x,p_y,phi_x,phi_x_x,phi_x_y,phi_x_xx,phi_x_yy,phi_y,phi_y_x,phi_y_y,phi_y_xx,phi_y_yy,psi,psi_x,psi_y)

dX = X_grid[1,0]-X_grid[0,0]
dY = Y_grid[0,1]-Y_grid[0,0]

iFk = np.sum(Fk,axis=0)
iLkl = np.sum(Lkl,axis=0)
iQklm = np.sum(Qklm,axis=0)

n_LOM = 4

print(iFk[0:n_LOM]*dX*dY)
print(iLkl[0:n_LOM,0:n_LOM]*dX*dY)
print(iQklm[0:n_LOM,0:n_LOM,0]*dX*dY)
print(iQklm[0:n_LOM,0:n_LOM,1]*dX*dY)

galerkin_folder = HOMEDIR+'output/mfg_galerkin/'
file_util.create_directory_if_not_exists(galerkin_folder)
figures_folder = galerkin_folder + 'figures/'
file_util.create_directory_if_not_exists(figures_folder)
figure_prefix = figures_folder + 'galerkin_constants'

cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

for k in range(n_LOM):

    Fk_i_grid = np.reshape(Fk[:,k],X_grid.shape)
    Fk_i_grid[cylinder_mask] = np.NaN
    MAX_Fk_i = np.nanmax(np.abs(Fk_i_grid))

    x_lim_vec = [-2,10.0]
    y_lim_vec = [-2.0,2.0]
    f1_levels = np.linspace(-MAX_Fk_i,MAX_Fk_i,21)
    fig = plot.figure(1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,Fk_i_grid,levels=f1_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    plot.savefig(figure_prefix+'_F_'+str(k)+'.png',dpi=300)
    plot.close()

    for l in range(n_LOM):

        Lkl_i_grid = np.reshape(Lkl[:,k,l],X_grid.shape)
        Lkl_i_grid[cylinder_mask] = np.NaN
        MAX_Lkl_i = np.nanmax(np.abs(Lkl_i_grid))

        x_lim_vec = [-2,10.0]
        y_lim_vec = [-2.0,2.0]
        f1_levels = np.linspace(-MAX_Lkl_i,MAX_Lkl_i,21)
        fig = plot.figure(1)
        plot.axis('equal')
        plot.contourf(X_grid,Y_grid,Lkl_i_grid,levels=f1_levels)
        plot.set_cmap('bwr')
        plot.colorbar()
        ax=plot.gca()
        ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
        ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
        plot.ylabel('y/D')
        plot.savefig(figure_prefix+'_L_'+str(k)+'_'+str(l)+'.png',dpi=300)
        plot.close()

        for m in range(n_LOM):

            Qklm_i_grid = np.reshape(Qklm[:,k,l,m],X_grid.shape)
            Qklm_i_grid[cylinder_mask] = np.NaN
            MAX_Qklm_i = np.nanmax(np.abs(Qklm_i_grid))

            x_lim_vec = [-2,10.0]
            y_lim_vec = [-2.0,2.0]
            f1_levels = np.linspace(-MAX_Qklm_i,MAX_Qklm_i,21)
            fig = plot.figure(1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,Qklm_i_grid,levels=f1_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            plot.savefig(figure_prefix+'_Q_'+str(k)+'_'+str(l)+'_'+str(m)+'.png',dpi=300)
            plot.close()


