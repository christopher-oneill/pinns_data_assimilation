

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
time_data_dir = 'I:/projects/fixed_cylinder/grid/data/'
supersample_factors = np.array([1,4,8,16,32,64])
#supersample_factors = np.array([1])
n_LOM = 10
err_dAk = np.zeros([supersample_factors.shape[0],n_LOM+1])

for nS in range(supersample_factors.shape[0]):
    supersample_factor = supersample_factors[nS]

    meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
    meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
    configFile = h5py.File(base_dir+'configuration.mat','r')
    POD_dataFile = h5py.File(base_dir+'POD_data.mat','r')

    fluctuatingPressureFile = h5py.File(time_data_dir+'fluctuatingPressure.mat','r')
    
    ux_temp = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
    uy_temp = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
    p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

    p_t  = np.array(fluctuatingPressureFile['fluctuatingPressure']).transpose()
    p_t = p_t + np.reshape(p,[p_t.shape[0],1])
    t = np.array(configFile['time']).transpose()
    n_x = int(np.array(POD_dataFile['n_x']))

    phi_x_temp = np.array(POD_dataFile['Phi_ext'][:,0:n_x]).transpose()
    phi_y_temp = np.array(POD_dataFile['Phi_ext'][:,n_x:2*n_x]).transpose()
    n_trunc = int(np.array(POD_dataFile['Phi']).shape[0])
    Ak_temp = np.array(POD_dataFile['Ak']).transpose()
    

    # combine the u and phi arrays according to the notation of noack

    phi_x = np.zeros([phi_x_temp.shape[0],phi_x_temp.shape[1]+1])
    phi_x[:,0] = ux_temp
    phi_x[:,1:phi_x.shape[1]] = phi_x_temp

    phi_y = np.zeros([phi_y_temp.shape[0],phi_y_temp.shape[1]+1])
    phi_y[:,0] = uy_temp
    phi_y[:,1:phi_y.shape[1]] = phi_y_temp

    Ak = np.ones([Ak_temp.shape[0],Ak_temp.shape[1]+1])
    Ak[:,1:Ak.shape[1]] = Ak_temp

    

    #coords
    print(configFile['X_vec'].shape)
    x = np.array(configFile['X_vec'][0,:])
    y = np.array(configFile['X_vec'][1,:])
    d = np.array(configFile['cylinderDiameter'])
    X_grid = np.array(configFile['X_grid'])
    Y_grid = np.array(configFile['Y_grid'])
    nu_mol = 0.0066667

    grad_X_grid = np.gradient(X_grid,axis=0)
    grad_Y_grid = np.gradient(Y_grid,axis=1)

    X_grid_lin = X_grid.ravel()
    Y_grid_lin = Y_grid.ravel()

    # compute the reference area of the POD
    A0 = np.sum((grad_X_grid.ravel())*(grad_Y_grid.ravel()),axis=0)
    M0 = X_grid.shape[0]*X_grid.shape[1]

    if supersample_factor>1:
        n_x = np.array(configFile['x_grid']).size
        n_y = np.array(configFile['y_grid']).size
        downsample_inds, n_x_d, n_y_d = downsample.compute_downsample_inds_even(supersample_factor,n_x,n_y)
        
        x = x[downsample_inds]
        y = y[downsample_inds]
        p_t = p_t[downsample_inds,:]
        p_t = p_t[:,0,:]
        phi_x = phi_x[downsample_inds,:]
        phi_x = phi_x[:,0,:]
        phi_y = phi_y[downsample_inds,:]
        phi_y = phi_y[:,0,:]
        p = p[downsample_inds]
        X_grid = np.reshape(X_grid_lin[downsample_inds],[n_x_d,n_y_d])
        Y_grid = np.reshape(Y_grid_lin[downsample_inds],[n_x_d,n_y_d])

    print(X_grid.shape)
    print(Y_grid.shape)
    print(phi_x.shape)

    cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

    # send quantities to (nx,ny) grid
    p_grid = np.reshape(p,X_grid.shape)
    phi_x_grid = np.reshape(phi_x,np.array([X_grid.shape[0],X_grid.shape[1],1+n_trunc]))
    phi_y_grid = np.reshape(phi_y,np.array([X_grid.shape[0],X_grid.shape[1],1+n_trunc]))
    p_t_grid = np.reshape(p_t,np.array([X_grid.shape[0],X_grid.shape[1],p_t.shape[1]]))

    # compute derivatives
    phi_x_x_grid = np.gradient(phi_x_grid,X_grid[:,0],axis=0)
    phi_x_xx_grid = np.gradient(phi_x_x_grid,X_grid[:,0],axis=0)
    phi_x_y_grid = np.gradient(phi_x_grid,Y_grid[0,:],axis=1)
    phi_x_yy_grid = np.gradient(phi_x_y_grid,Y_grid[0,:],axis=1)

    phi_y_x_grid = np.gradient(phi_y_grid,X_grid[:,0],axis=0)
    phi_y_xx_grid = np.gradient(phi_y_x_grid,X_grid[:,0],axis=0)
    phi_y_y_grid = np.gradient(phi_y_grid,Y_grid[0,:],axis=1)
    phi_y_yy_grid = np.gradient(phi_y_y_grid,Y_grid[0,:],axis=1)

    p_t_x_grid = np.gradient(p_t_grid,X_grid[:,0],axis=0)
    p_t_y_grid = np.gradient(p_t_grid,Y_grid[0,:],axis=1)


    # set elements in cylinder to zero
    phi_x_x_grid[cylinder_mask]==0
    phi_x_xx_grid[cylinder_mask]==0
    phi_y_x_grid[cylinder_mask]==0
    phi_y_xx_grid[cylinder_mask]==0
    phi_x_y_grid[cylinder_mask]==0
    phi_x_yy_grid[cylinder_mask]==0
    phi_y_y_grid[cylinder_mask]==0
    phi_y_yy_grid[cylinder_mask]==0


    # send derivatives to (nx*ny) vector
    phi_x_x = np.reshape(phi_x_x_grid,phi_x.shape)
    phi_x_y = np.reshape(phi_x_x_grid,phi_x.shape)
    phi_x_xx = np.reshape(phi_x_xx_grid,phi_x.shape)
    phi_x_yy = np.reshape(phi_x_yy_grid,phi_x.shape)

    phi_y_x = np.reshape(phi_y_x_grid,phi_x.shape)

    phi_y_y = np.reshape(phi_y_y_grid,phi_x.shape)
    phi_y_xx = np.reshape(phi_y_xx_grid,phi_x.shape)
    phi_y_yy = np.reshape(phi_y_yy_grid,phi_x.shape)

    p_t_x = np.reshape(p_t_x_grid,p_t.shape)
    p_t_y = np.reshape(p_t_y_grid,p_t.shape)



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

    amp_Ak = 0.5*np.max(Ak[:,0:n_LOM],axis=0)-0.5*np.min(Ak[:,0:n_LOM],axis=0)

    dX = np.gradient(X_grid,axis=0)
    dY = np.gradient(Y_grid,axis=1)

    dA = dX*dY # we need to normalize this by the A of the S=1 case so that we get back the right magnitude of TC's

    # compute galerkin projection
    D_ij, C_ijk = galerkin.galerkin_noack(phi_x_grid,phi_x_x_grid,phi_x_y_grid,phi_x_xx_grid,phi_x_yy_grid,phi_y_grid,phi_y_x_grid,phi_y_y_grid,phi_y_xx_grid,phi_y_yy_grid)

    print(D_ij.shape)

    i_Dij = np.zeros([D_ij.shape[2],D_ij.shape[3]])
    i_Cijk = np.zeros([C_ijk.shape[2],C_ijk.shape[3],C_ijk.shape[4]])
   
    for i in range(D_ij.shape[2]):
        for j in range(D_ij.shape[3]):
            i_Dij[i,j] = np.sum((M0*dA/A0)*D_ij[:,:,i,j],(0,1))
            for k in range(C_ijk.shape[4]):
                i_Cijk[i,j,k] = np.sum((M0*dA/A0)*C_ijk[:,:,i,j,k],(0,1))
    
    n_LOM_p = n_LOM+1
    # calibrate the pressure
    Phi_p_t_k = np.zeros([n_LOM_p,p_t.shape[1]])
    L_pi_kl = np.zeros([n_LOM_p,n_LOM_p+1])

    ## compute the inner product of the pressure gradient with the velocity modes
    for k in range(n_LOM_p):
        Phi_p_t_k[k,:] = np.sum((-np.reshape(phi_x[:,k],[phi_x.shape[0],1])*p_t_x-np.reshape(phi_y[:,k],[phi_y.shape[0],1])*p_t_y),axis=0)
    amp_Phi_p_t_k = 0.5*np.max(Phi_p_t_k,axis=0)-0.5*np.min(Phi_p_t_k,axis=0)

    ## compute the linear coefficients based on the inverse of the temporal coefficients and the pressure contribution
    Ak_inv = np.linalg.pinv(Ak[:,0:n_LOM_p+1]).transpose()
    for k in range(n_LOM_p):
        L_pi_kl[k,:]=np.matmul(np.reshape(Phi_p_t_k[k,:],[1,Phi_p_t_k.shape[1]]),Ak_inv)

    # test if the linear coefficients accurately reconstruct the pressure contribution
    Phi_p_t_r = np.zeros(Phi_p_t_k.shape)
    print(L_pi_kl.shape)
    print(Ak.shape)
    for k in range(n_LOM):
        Phi_p_t_r[k,:] = np.matmul(L_pi_kl[k,:],Ak[:,0:n_LOM_p+1].transpose())

    if False:
        # plot the error in the reconstruction of the pressure term
        for k in range(7):
            plot.figure(100+k)
            plot.subplot(2,1,1)
            plot.plot(t,Phi_p_t_k[k,:])
            plot.plot(t,Phi_p_t_r[k,:])
            plot.subplot(2,1,2)
            plot.plot(t,(Phi_p_t_k[k,:]-Phi_p_t_r[k,:])/amp_Phi_p_t_k[k])
        plot.show()
    
    


    galerkin_folder = HOMEDIR+'output/mfg_galerkin_noack/'
    file_util.create_directory_if_not_exists(galerkin_folder)
    figures_folder = galerkin_folder + 'figures/'
    file_util.create_directory_if_not_exists(figures_folder)
    figure_prefix = figures_folder + 'galerkin_constants'

    # numerical derivative
    dAk_n = np.gradient(Ak,t[:,0],axis=0)
    amp_dAk = 0.5*np.max(dAk_n,axis=0)-0.5*np.min(dAk_n,axis=0)
    amp_dAk[0]=1 # we need to set the amplitude of the constant term to not zero

    # compute the derivative using the galerkin equations
    dAk_g = np.zeros([Ak.shape[0],n_LOM+1])

    for k in range(n_LOM+1):
        for l in range(n_LOM+1):
            dAk_g[:,k] = dAk_g[:,k]+nu_mol*i_Dij[k,l]*Ak[:,l]+L_pi_kl[k,l]*Ak[:,l]
            for m in range(n_LOM+1):
                dAk_g[:,k] = dAk_g[:,k]-i_Cijk[k,l,m]*Ak[:,l]*Ak[:,m]

    
    err_dAk[nS,:] = np.power(np.mean(np.power(dAk_n[np.int32(t.shape[0]/6):np.int32(5*t.shape[0]/6),0:n_LOM+1]-dAk_g[np.int32(t.shape[0]/6):np.int32(5*t.shape[0]/6),:],2.0),axis=0),0.5)/amp_dAk[0:n_LOM+1]


    ks  =[0,1,2,3,4,5,6]
    for k in ks:
        plot.figure(k+nS*n_LOM)
        plot.subplot(2,1,1)
        plot.plot(t,dAk_n[:,k])
        plot.plot(t,dAk_g[:,k])

        plot.subplot(2,1,2)        
        plot.plot(t,(dAk_n[:,k]-dAk_g[:,k])/amp_dAk[k])

        plot.ylabel('da_i/(0.5*(max(da_i)-min(da_i))')
        plot.savefig(figure_prefix+'_dAk'+str(k)+'_S'+str(supersample_factors[nS])+'_full.png',dpi=300)
        plot.xlim([460,480])
        plot.savefig(figure_prefix+'_dAk'+str(k)+'_S'+str(supersample_factors[nS])+'_close.png',dpi=300)
        plot.close()




legend_arr = []
plot.figure(100)
for k in range(1,6):
    plot.plot(supersample_factors,100*err_dAk[:,k],marker='o')
    legend_arr.append('Mode '+str(k+1))
plot.xlabel('Supersample Factor')
plot.ylabel('Percent Error')
plot.legend(legend_arr)
plot.xscale('log')
plot.yscale('log')
plot.savefig(figure_prefix+'galerkin_error.png',dpi=300)
plot.close()

plot.show()