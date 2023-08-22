

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
from pinns_galerkin_viv.lib import downsample

# read the data

base_dir = HOMEDIR+'/data/mazi_fixed_grid/'
supersample_factors = np.array([1,8,16,32,64])

for nS in range(supersample_factors.shape[0]):
    supersample_factor = supersample_factors[nS]

    meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
    meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
    configFile = h5py.File(base_dir+'configuration.mat','r')
    POD_dataFile = h5py.File(base_dir+'POD_data.mat','r')

    ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
    uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
    p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()
    t = np.array(configFile['time']).transpose()
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

    X_grid_lin = X_grid.ravel()
    Y_grid_lin = Y_grid.ravel()

    noise_magnitude = 0.00

    if supersample_factor==1:
        noise_ux = noise_magnitude*np.max(ux,0)*np.random.standard_normal(ux.shape)
        noise_uy = noise_magnitude*np.max(uy,0)*np.random.standard_normal(uy.shape)
        noise_p = noise_magnitude*np.max(p,0)*np.random.standard_normal(p.shape)
        noise_phi_x = noise_magnitude*np.max(phi_x,0)*np.random.standard_normal(phi_x.shape)
        noise_phi_y = noise_magnitude*np.max(phi_y,0)*np.random.standard_normal(phi_y.shape)
        noise_psi = noise_magnitude*np.max(psi,0)*np.random.standard_normal(psi.shape)


    if supersample_factor>1:
        ux = ux+noise_ux
        uy = uy+noise_uy
        p = p +noise_p
        phi_x = phi_x+noise_phi_x
        phi_y = phi_y + noise_phi_y
        psi = psi+noise_psi


        n_x = np.array(configFile['x_grid']).size
        n_y = np.array(configFile['y_grid']).size
        downsample_inds, n_x_d, n_y_d = downsample.compute_downsample_inds_even(supersample_factor,n_x,n_y)
        
        x = x[downsample_inds]
        y = y[downsample_inds]
        ux = ux[downsample_inds]
        ux = ux[:,0]
        uy = uy[downsample_inds]
        uy = uy[:,0]
        p = p[downsample_inds]
        p = p[:,0]
        phi_x = phi_x[downsample_inds,:]
        phi_x = phi_x[:,0,:]
        phi_y = phi_y[downsample_inds,:]
        phi_y = phi_y[:,0,:]
        psi = psi[downsample_inds,:]
        psi = psi[:,0,:]
        X_grid = np.reshape(X_grid_lin[downsample_inds],[n_x_d,n_y_d])
        Y_grid = np.reshape(Y_grid_lin[downsample_inds],[n_x_d,n_y_d])

    print(X_grid.shape)
    print(Y_grid.shape)
    print(phi_x.shape)
    print(psi.shape)

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

    if supersample_factor==1:
        phi_x_grid_s1 = phi_x_grid
        phi_x_x_grid_s1 = phi_x_x_grid
        phi_y_x_grid_s1 = phi_y_x_grid
        X_grid_s1 = X_grid
        Y_grid_s1 = Y_grid

    ind_test_line_x = np.int32(Y_grid.shape[1]/3)
    ind_line_x_s1 = np.argwhere(Y_grid_s1[0,:]==Y_grid[0,ind_test_line_x])

    #plot.figure(100+nS)
    #plot.plot(X_grid_s1[:,ind_line_x_s1].ravel(),phi_x_grid_s1[:,ind_line_x_s1,0].ravel())
    #plot.plot(X_grid_s1[:,ind_line_x_s1].ravel(),phi_x_x_grid_s1[:,ind_line_x_s1,0].ravel())
    #plot.scatter(X_grid[:,ind_test_line_x].ravel(),phi_x_grid[:,ind_test_line_x,0].ravel())
    #plot.scatter(X_grid[:,ind_test_line_x].ravel(),phi_x_x_grid[:,ind_test_line_x,0].ravel())


    # compute galerkin projection

    Fk, Lkl, Qklm = galerkin.galerkin_projection(nu_mol,ux,ux_x,ux_y,ux_xx,ux_yy,uy,uy_x,uy_y,uy_xx,uy_yy,p,p_x,p_y,phi_x,phi_x_x,phi_x_y,phi_x_xx,phi_x_yy,phi_y,phi_y_x,phi_y_y,phi_y_xx,phi_y_yy,psi,psi_x,psi_y)

    dX = np.gradient(X_grid,axis=0)
    dY = np.gradient(Y_grid,axis=1)

    if supersample_factor ==1:
        dA_s1 = np.mean(dX.ravel()*dY.ravel())

    #print('dX: ',dX,'  dY: ',dY)
    dA  = dX.ravel()*dY.ravel()
    print('dA: ',np.mean(dA))
    print('area: ',np.sum(dA))
    iFk = np.sum(np.swapaxes(np.swapaxes(Fk,0,1)*dA,1,0),axis=0)
    iLkl = np.sum(np.swapaxes(np.swapaxes(Lkl,0,2)*dA,2,0),axis=0)
    iQklm = np.sum(np.swapaxes(np.swapaxes(Qklm,0,3)*dA,3,0),axis=0)

    print(iQklm.shape)

    n_LOM = 6

    #print(iFk[0:n_LOM])
    #print(iLkl[0:n_LOM,0:n_LOM])
    #print(iQklm[0:n_LOM,0:n_LOM,0])
    #print(iQklm[0:n_LOM,0:n_LOM,1])

    galerkin_folder = HOMEDIR+'output/mfg_galerkin/'
    file_util.create_directory_if_not_exists(galerkin_folder)
    figures_folder = galerkin_folder + 'figures/'
    file_util.create_directory_if_not_exists(figures_folder)
    figure_prefix = figures_folder + 'galerkin_constants'

    cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

    if False:
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

    # numerical derivative
    dAk_n = np.gradient(Ak*dA_s1,t[:,0],axis=0)
    amp_Ak = 0.5*np.max(dAk_n,axis=0)-0.5*np.min(dAk_n,axis=0)


    # compute the derivative using the galerkin equations
    dAk_g = np.zeros([Ak.shape[0],n_LOM])

    for k in range(n_LOM):
        dAk_g[:,k] = dAk_g[:,k]+iFk[k]
        for l in range(n_LOM):
            dAk_g[:,k] = dAk_g[:,k]+iLkl[k,l]*Ak[:,l]
            for m in range(n_LOM):
                dAk_g[:,k] = dAk_g[:,k]+iQklm[k,l,m]*Ak[:,l]*Ak[:,m]

    ks  =[3]
    for k in ks:
        plot.figure(k+nS*n_LOM)
        plot.plot(t,(dAk_n[:,k])/amp_Ak[k])
        plot.plot(t,dAk_g[:,k]/amp_Ak[k])
        plot.xlim([460,480])

    #for k in range(n_LOM):
    #    plot.figure(k+n_LOM)
    #    plot.plot(t,(dAk_n[:,k]-dAk_g[:,k])/amp_Ak[k])
        #plot.plot(t,dAk_g[:,k]/amp_Ak[k])

    
plot.show()