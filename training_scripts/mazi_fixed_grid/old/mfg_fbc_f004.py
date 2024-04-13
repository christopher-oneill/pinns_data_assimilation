#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import scipy.io
from scipy import interpolate
from scipy.interpolate import griddata
import tensorflow as tf
import tensorflow.keras as keras
import h5py
import os
import glob
import re
import smt
import h5py
from smt.sampling_methods import LHS
from pyDOE import lhs
from datetime import datetime
from datetime import timedelta
import platform
import sys

# saving loading functions


def load_custom():
    # get the model from the model file
    model_file,model_training_steps = find_highest_numbered_file(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep','[0-9]*','_model.h5')
    model_FANS = keras.models.load_model(model_file,custom_objects={'QuadraticInputPassthroughLayer':QuadraticInputPassthroughLayer,'FourierPassthroughEmbeddingLayer':FourierPassthroughEmbeddingLayer,'FourierPassthroughReductionLayer':FourierPassthroughReductionLayer,'InputPassthroughLayer':InputPassthroughLayer})
    
    # get the most recent set of weights
    
    checkpoint_filename,weights_training_steps = find_highest_numbered_file(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep','[0-9]*','.weights.h5')

    if model_training_steps>=weights_training_steps:
        # just set the training steps
        training_steps = model_training_steps
    else:
        training_steps = weights_training_steps
        # load from weights
        if checkpoint_filename is not None:
            #pred = model_RANS.predict(np.zeros([10,2])) # needed to enable weight loading
            weights_file = h5py.File(checkpoint_filename)          
            #print(weights_file.keys())
            #print(weights_file['_layer_checkpoint_dependencies']['quadratic_input_passthrough_layer']['vars'].keys())
            w_keys =  list(weights_file.keys())
            if platform.system()=='Windows':
                # check if the file is windows weight style or linux weight style
                if w_keys[0]=='_layer_checkpoint_dependencies':
                    # if the file was saved on linux we need to copy it back to windows format 
                    new_weights_file = h5py.File(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep'+str(training_steps+1)+'.weights.h5','w')
                    # copy all other keys
                    for nkey in range(1,len(w_keys)):
                        weights_file.copy(weights_file[w_keys[nkey]],new_weights_file)
                    # copy the layer keys
                    layer_keys = list(weights_file['_layer_checkpoint_dependencies'].keys())
                    print(weights_file['_layer_checkpoint_dependencies']['fourier_passthrough_reduction_layer']['dense_reduction_layer']['vars'].keys())
                    for nkey in range(len(layer_keys)):
                        new_weights_file.create_group('_layer_checkpoint_dependencies\\'+layer_keys[nkey])
                        weights_file.copy(weights_file['_layer_checkpoint_dependencies'][layer_keys[nkey]]['vars'],new_weights_file['_layer_checkpoint_dependencies\\'+layer_keys[nkey]]) # ['_layer_checkpoint_dependencies\\'+layer_keys[nkey]] 
                        if layer_keys[nkey] == 'fourier_passthrough_reduction_layer':
                            dense_key = list(weights_file['_layer_checkpoint_dependencies'][layer_keys[nkey]])[0]
                            print(dense_key)
                            new_weights_file.create_group('_layer_checkpoint_dependencies\\'+layer_keys[nkey]+'\\'+dense_key)
                            weights_file.copy(weights_file['_layer_checkpoint_dependencies'][layer_keys[nkey]][dense_key]['vars'],new_weights_file['_layer_checkpoint_dependencies\\'+layer_keys[nkey]+'\\'+dense_key]) # ['_layer_checkpoint_dependencies\\'+layer_keys[nkey]] 
                            print(new_weights_file['_layer_checkpoint_dependencies\\'+layer_keys[nkey]+'\\'+dense_key].keys())
                    new_weights_file.close()
                    model_FANS.load_weights(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep'+str(training_steps+1)+'.weights.h5')
                else:
                    # windows style loading
                    model_FANS.load_weights(checkpoint_filename)
            else:
                model_FANS.load_weights(checkpoint_filename)
                
    model_FANS.summary()
    print('Model Loaded. Epoch',str(training_steps))
    return model_FANS, training_steps

def save_pred():
    pred = model_FANS.predict(X_train,batch_size=32)
    h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
    t_mxr,t_mxi,t_myr,t_myi,t_massr,t_massi = net_f_fourier_cartesian(f_colloc_train,mean_data)
    h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_error.mat','w')
    h5f.create_dataset('mxr',data=t_mxr)
    h5f.create_dataset('mxi',data=t_mxi)
    h5f.create_dataset('myr',data=t_myr)
    h5f.create_dataset('myi',data=t_myi)
    h5f.create_dataset('massr',data=t_massr)
    h5f.create_dataset('massi',data=t_massi)
    h5f.close()

####### plotting functions

def plot_NS_residual():
    # NS residual
    global X_plot
    global ScalingParameters
    cylinder_mask = (np.power(X_grid_plot,2.0)+np.power(Y_grid_plot,2.0))<=np.power(d/2.0,2.0)
    mxr,mxi,myr,myi,massr,massi = batch_FANS_cartesian(model_FANS,X_plot/ScalingParameters.MAX_x,mean_data_plot,ScalingParameters,1000)
    mxr = 1.0*np.reshape(mxr,X_grid_plot.shape)
    mxr[cylinder_mask] = np.NaN
    mxi = 1.0*np.reshape(mxi,X_grid_plot.shape)
    mxi[cylinder_mask] = np.NaN
    myr = 1.0*np.reshape(myr,X_grid_plot.shape)
    myr[cylinder_mask] = np.NaN
    myi = 1.0*np.reshape(myi,X_grid_plot.shape)
    myi[cylinder_mask] = np.NaN
    massr = 1.0*np.reshape(massr,X_grid_plot.shape)
    massr[cylinder_mask] = np.NaN
    massi = 1.0*np.reshape(massi,X_grid_plot.shape)
    massi[cylinder_mask] = np.NaN


    plot.figure(1)
    plot.subplot(3,1,1)
    mxr_min = np.nanpercentile(mxr.ravel(),0.1)
    mxr_max = np.nanpercentile(mxr.ravel(),99.9)
    mxr_level = np.max([abs(mxr_min),abs(mxr_max)])
    mxr_levels = np.linspace(-mxr_level,mxr_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,mxr,levels=mxr_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    myr_min = np.nanpercentile(myr.ravel(),0.1)
    myr_max = np.nanpercentile(myr.ravel(),99.9)
    myr_level = np.max([abs(myr_min),abs(myr_max)])
    myr_levels = np.linspace(-myr_level,myr_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,myr,levels=myr_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    massr_min = np.nanpercentile(massr.ravel(),0.1)
    massr_max = np.nanpercentile(massr.ravel(),99.9)
    massr_level = np.max([abs(massr_min),abs(massr_max)])
    massr_levels = np.linspace(-massr_level,massr_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,massr,levels=massr_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_FANS_residual_r.png',dpi=300)
    plot.close(1)

    plot.figure(1)
    plot.subplot(3,1,1)
    mxi_min = np.nanpercentile(mxi.ravel(),0.1)
    mxi_max = np.nanpercentile(mxi.ravel(),99.9)
    mxi_level = np.max([abs(mxi_min),abs(mxi_max)])
    mxi_levels = np.linspace(-mxi_level,mxi_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,mxi,levels=mxi_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    myi_min = np.nanpercentile(myi.ravel(),0.1)
    myi_max = np.nanpercentile(myi.ravel(),99.9)
    myi_level = np.max([abs(myi_min),abs(myi_max)])
    myi_levels = np.linspace(-myi_level,myi_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,myi,levels=myi_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    massi_min = np.nanpercentile(massi.ravel(),0.1)
    massi_max = np.nanpercentile(massi.ravel(),99.9)
    massi_level = np.max([abs(massi_min),abs(massi_max)])
    massi_levels = np.linspace(-massi_level,massi_level,21)
    plot.contourf(X_grid_plot,Y_grid_plot,massi,levels=massi_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_FANS_residual_i.png',dpi=300)
    plot.close(1)


def plot_err():
    global X_plot  
    global d
    global training_steps
    global ScalingParameters

    cylinder_mask = (np.power(X_grid_plot,2.0)+np.power(Y_grid_plot,2.0))<=np.power(d/2.0,2.0)

    F_test_grid_temp = 1.0*F_test_grid
    F_test_grid_temp[cylinder_mask,:] = np.NaN

    pred_test = model_FANS(X_plot/ScalingParameters.MAX_x,training=False)
    
    pred_test_grid = np.copy(np.reshape(pred_test,[X_grid_plot.shape[0],X_grid_plot.shape[1],12]))
    pred_test_grid[cylinder_mask,:] = np.NaN


    # interpolate the error
    err_test_grid = F_test_grid - pred_test_grid


    plot.close('all')

    X_train_plot = X_train_LBFGS*ScalingParameters.MAX_x

    plot_save_exts = ['_phi_xr.png','_phi_xi.png','_phi_yr.png','_phi_yi.png','_tau_xx_r.png','_tau_xx_i.png','_tau_xy_r.png','_tau_xy_i.png','_tau_yy_r.png','_tau_yy_i.png','_psi_r.png','_psi_i.png']
    # quantities
    for i in range(12):
        plot.figure(1)
        plot.subplot(3,1,1)
        plot.contourf(X_grid_plot,Y_grid_plot,F_test_grid_temp[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        if supersample_factor>1:
            plot.scatter(X_train_plot[:,0],X_train_plot[:,1],2,'k','.')
        
        plot.subplot(3,1,2)
        plot.contourf(X_grid_plot,Y_grid_plot,pred_test_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        e_test_min = np.nanpercentile(err_test_grid[:,:,i].ravel(),0.1)
        e_test_max = np.nanpercentile(err_test_grid[:,:,i].ravel(),99.9)
        e_test_level = np.max([abs(e_test_min),abs(e_test_max)])
        e_test_levels = np.linspace(-e_test_level,e_test_level,21)
        plot.contourf(X_grid_plot,Y_grid_plot,err_test_grid[:,:,i],levels=e_test_levels,extend='both')
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts[i],dpi=300)
        plot.close(1)

    # also plot the profiles
    profile_locations = np.linspace(-1.5,9.5,22) # spacing of 0.5
    X_line_locations = X_grid_plot[:,0]
    X_distance_matrix = np.power(np.power(np.reshape(X_line_locations,[X_line_locations.size,1])-np.reshape(profile_locations,[1,profile_locations.size]),2.0),0.5)
    # find index of closest data line
    line_inds = np.argmin(X_distance_matrix,axis=0)
    profile_locations = X_grid_plot[line_inds,:]

    #point_locations = np.linspace(-2,2,40)
    #Y_line_locations = Y_plot[:,0]
    #Y_distance_matrix = np.power(np.power(np.reshape(Y_line_locations,[Y_line_locations.size,1])-np.reshape(point_locations,[1,point_locations.size]),2.0),0.5)
    #point_inds = np.argmin(Y_distance_matrix,axis=0)
    point_locations = Y_grid_plot[0,:]

    plot_save_exts2 = ['_phi_xr_profile.png','_phi_xi_profile.png','_phi_yr_profile.png','_phi_yi_profile.png','_tau_xx_r_profile.png','_tau_xx_i_profile.png','_tau_xy_r_profile.png','_tau_xy_i_profile.png','_tau_yy_r_profile.png','_tau_yy_i_profile.png','_psi_r_profile.png','_psi_i_profile.png']
    x_offset = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    x_scale = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    err_scale = np.nanmax(np.abs(err_test_grid),axis=(0,1))
    
    for i in range(12):
        plot.figure(1)
        plot.subplot(3,1,1)
        plot.contourf(X_grid_plot,Y_grid_plot,F_test_grid_temp[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        
        plot.set_cmap('bwr')
        plot.xlim(-2,10)
        
        plot.subplot(3,1,2)
        for k in range(profile_locations.shape[0]):
            plot.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
            plot.plot((F_test_grid_temp[line_inds[k],:,i]+x_offset[i])*x_scale[i]+profile_locations[k],point_locations,'-k',linewidth=0.5)
            plot.plot((pred_test_grid[line_inds[k],:,i]+x_offset[i])*x_scale[i]+profile_locations[k],point_locations,'-r',linewidth=0.5)
        plot.xlim(-2,10)
        plot.subplot(3,1,3)
        for k in range(profile_locations.shape[0]):
            plot.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
            plot.plot((0.5/err_scale[i])*err_test_grid[line_inds[k],:,i]+profile_locations[k],point_locations,'-r',linewidth=0.5)
        plot.text(4.5,1.7,"x-Scaled by (0.5/MaxErr). MaxErr={txterr:.4f}".format(txterr=err_scale[i]),fontsize=7.0)
        plot.xlim(-2,10)
        plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts2[i],dpi=300)
        plot.close(1)

def plot_frequencies():
    temp_F_test_grid = np.copy(F_test_grid)
    temp_X_grid_plot = X_grid_plot
    temp_Y_grid_plot = Y_grid_plot

    line_plot = int((X_grid_plot.shape[1])/3.0)

    x_plot = X_grid_plot[:,line_plot]/ScalingParameters.MAX_x

    plot.figure(1)
    plot.plot(x_plot,temp_F_test_grid[:,line_plot,8])
    plot.plot(x_plot,np.sin((3*np.pi*ScalingParameters.MAX_x)*x_plot))
    plot.show()

def plot_gradients():
    global o_test_grid
    global X_grid
    global Y_grid
    meps = np.finfo(np.float64).eps

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    data_grads = np.zeros([o_test_grid.shape[0],o_test_grid.shape[1],14])

    labels = ['ux_x','ux_y','uy_x','uy_y','uxux_x','uxuy_x','uxuy_y','uyuy_y','p_x','p_y','ux_xx','ux_yy','uy_xx','uy_yy']

    o_test_grid_temp = 1.0*o_test_grid
    o_test_grid_temp[:,:,0] = o_test_grid_temp[:,:,0]*ScalingParameters.MAX_ux
    o_test_grid_temp[:,:,1] = o_test_grid_temp[:,:,1]*ScalingParameters.MAX_uy
    o_test_grid_temp[:,:,2] = o_test_grid_temp[:,:,2]*ScalingParameters.MAX_uxppuxpp
    o_test_grid_temp[:,:,3] = o_test_grid_temp[:,:,3]*ScalingParameters.MAX_uxppuypp
    o_test_grid_temp[:,:,4] = o_test_grid_temp[:,:,4]*ScalingParameters.MAX_uyppuypp
    o_test_grid_temp[:,:,5] = o_test_grid_temp[:,:,5]*ScalingParameters.MAX_p
    

    # first derivatives of data
    data_grads[:,:,0] = np.gradient(o_test_grid_temp[:,:,0],X_grid[:,0],axis=0) # ux_x
    data_grads[:,:,1] = np.gradient(o_test_grid_temp[:,:,0],Y_grid[0,:],axis=1) # ux_y
    data_grads[:,:,2] = np.gradient(o_test_grid_temp[:,:,1],X_grid[:,0],axis=0) # uy_x
    data_grads[:,:,3] = np.gradient(o_test_grid_temp[:,:,1],Y_grid[0,:],axis=1) # uy_y
    data_grads[:,:,4] = np.gradient(o_test_grid_temp[:,:,2],X_grid[:,0],axis=0) # uxux_x
    #data_uxux_y = np.gradient(o_test_grid_temp[:,:,2],Y_grid[0,:],axis=1)
    data_grads[:,:,5] = np.gradient(o_test_grid_temp[:,:,3],X_grid[:,0],axis=0) # uxuy_x
    data_grads[:,:,6] = np.gradient(o_test_grid_temp[:,:,3],Y_grid[0,:],axis=1) # uxuy_y
    #data_uyuy_x = np.gradient(o_test_grid_temp[:,:,4],X_grid[:,0],axis=0)
    data_grads[:,:,7] = np.gradient(o_test_grid_temp[:,:,4],Y_grid[0,:],axis=1) # uyuy_y
    data_grads[:,:,8] = np.gradient(o_test_grid_temp[:,:,5],X_grid[:,0],axis=0) # p_x
    data_grads[:,:,9] = np.gradient(o_test_grid_temp[:,:,5],Y_grid[0,:],axis=1) # p_y

    # second derivatives of data
    data_grads[:,:,10] = np.gradient(data_grads[:,:,0],X_grid[:,0],axis=0) # ux_xx
    data_grads[:,:,11] = np.gradient(data_grads[:,:,1],Y_grid[0,:],axis=1) # ux_yy
    data_grads[:,:,12] = np.gradient(data_grads[:,:,2],X_grid[:,0],axis=0) # uy_xx
    data_grads[:,:,13] = np.gradient(data_grads[:,:,3],Y_grid[0,:],axis=1) # uy_yy

    data_grads[cylinder_mask,:]=np.NaN


    NN_grads = RANS_gradients(model_mean,ScalingParameters,X_test)
    NN_grads_grid = 1.0*np.reshape(NN_grads,data_grads.shape)
    NN_grads_grid[cylinder_mask,:]=np.NaN

    for i in range(data_grads.shape[2]):
        plot.figure(1)
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,data_grads[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,NN_grads_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        plot.contourf(X_grid,Y_grid,data_grads[:,:,i]-NN_grads_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_gradients_'+labels[i]+'.png',dpi=300)
        plot.close(1)
    
    # f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy) 
    # f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)
    # f_mass = ux_x + uy_y

    f_x = (o_test_grid_temp[:,:,0]*data_grads[:,:,0] + o_test_grid_temp[:,:,1]*data_grads[:,:,1]) + (data_grads[:,:,4] + data_grads[:,:,6]) + data_grads[:,:,8] - (ScalingParameters.nu_mol)*(data_grads[:,:,10]+data_grads[:,:,11])  
    f_y = (o_test_grid_temp[:,:,0]*data_grads[:,:,2] + o_test_grid_temp[:,:,1]*data_grads[:,:,3]) + (data_grads[:,:,5] + data_grads[:,:,7]) + data_grads[:,:,9] - (ScalingParameters.nu_mol)*(data_grads[:,:,12]+data_grads[:,:,13])
    f_mass = data_grads[:,:,0] + data_grads[:,:,3]


    levels_fd = np.linspace(-0.01,0.01,21)
    plot.figure(1)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,f_x,levels=levels_fd,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,f_y,levels=levels_fd,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,f_mass,levels=levels_fd,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_RANS_residual_finite_difference.png',dpi=300)
    plot.close(1)

    plot.figure(1)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,np.log10(np.abs(f_x+meps)),levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,np.log10(np.abs(f_y+meps)),levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,np.log10(np.abs(f_mass+meps)),levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_RANS_residual_finite_difference_log.png',dpi=300)
    plot.close(1)




    FANS_NN_grads = FANS_gradients(model_FANS,X_test,ScalingParameters)
    FANS_NN_grads_grid = 1.0*np.reshape(NN_grads,data_grads.shape)
    FANS_NN_grads_grid[cylinder_mask,:]=np.NaN

    # governing equations
    # f_xr = -omega*phi_xi+(phi_xr*ux_x + phi_yr*ux_y+ ux*phi_xr_x +uy*phi_xr_y ) + (tau_xx_r_x + tau_xy_r_y) + psi_r_x - (ScalingParameters.nu_mol)*(phi_xr_xx+phi_xr_yy)  
    # f_xi =  omega*phi_xr+(phi_xi*ux_x + phi_yi*ux_y+ ux*phi_xi_x +uy*phi_xi_y ) + (tau_xx_i_x + tau_xy_i_y) + psi_i_x - (ScalingParameters.nu_mol)*(phi_xi_xx+phi_xi_yy)  
    # f_yr = -omega*phi_yi+(phi_xr*uy_x + phi_yr*uy_y+ ux*phi_yr_x +uy*phi_yr_y ) + (tau_xy_r_x + tau_yy_r_y) + psi_r_y - (ScalingParameters.nu_mol)*(phi_yr_xx+phi_yr_yy) 
    # f_yi =  omega*phi_yr+(phi_xi*uy_x + phi_yi*uy_y+ ux*phi_yi_x +uy*phi_yi_y ) + (tau_xy_i_x + tau_yy_i_y) + psi_i_y - (ScalingParameters.nu_mol)*(phi_yi_xx+phi_yi_yy)  
    # f_mr = phi_xr_x + phi_yr_y
    # f_mi = phi_xi_x + phi_yi_y

####### physics functions
# mean model functions
@tf.function
def RANS_cartesian(colloc_tensor):
    # dummy function to provide to the mean model
    return 0.0, 0.0, 0.0

# function wrapper, combine data and physics loss
def RANS_loss_wrapper(colloc_tensor_f,BC_ns,BC_p): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # we just need a dummy function here so that we can compile the original mean model                      
        return 0

    return mean_loss

@tf.function
def RANS_gradients(model_RANS,ScalingParameters,colloc_tensor):
    # the colloc tensor should be in FANS model scaling, so convert to 
    colloc_tensor =  colloc_tensor*(ScalingParameters.MAX_x/ScalingParameters.mean_MAX_x)
    # evaluate model and gradients:
    up = model_RANS(colloc_tensor)
    # knowns
    ux = up[:,0]*ScalingParameters.MAX_ux
    uy = up[:,1]*ScalingParameters.MAX_uy
    uxux = up[:,2]*ScalingParameters.MAX_uxppuxpp
    uxuy = up[:,3]*ScalingParameters.MAX_uxppuypp
    uyuy = up[:,4]*ScalingParameters.MAX_uyppuypp
    # unknowns
    p = up[:,5]*ScalingParameters.MAX_p
    
    # compute the gradients of the quantities
    
    # first gradients
    dux = tf.gradients((ux,), (colloc_tensor))[0]
    duy = tf.gradients((uy,), (colloc_tensor))[0]
    duxux = tf.gradients((uxux,), (colloc_tensor))[0]
    duxuy = tf.gradients((uxuy,), (colloc_tensor))[0]
    duyuy = tf.gradients((uyuy,), (colloc_tensor))[0]
    dp = tf.gradients((p,), (colloc_tensor))[0]
    # ux grads

    ux_x = dux[:,0]/ScalingParameters.mean_MAX_x
    ux_y = dux[:,1]/ScalingParameters.mean_MAX_y
    
    # uy gradient
    uy_x = duy[:,0]/ScalingParameters.mean_MAX_x
    uy_y = duy[:,1]/ScalingParameters.mean_MAX_y


    # gradient unmodeled reynolds stresses
    uxux_x = duxux[:,0]/ScalingParameters.mean_MAX_x
    uxuy_x = duxuy[:,0]/ScalingParameters.mean_MAX_x
    uxuy_y = duxuy[:,1]/ScalingParameters.mean_MAX_y
    uyuy_y = duyuy[:,1]/ScalingParameters.mean_MAX_y

    # pressure gradients
    p_x = dp[:,0]/ScalingParameters.mean_MAX_x
    p_y = dp[:,1]/ScalingParameters.mean_MAX_y

    # second gradients
    (dux_x) = tf.gradients((ux_x,),(colloc_tensor,))[0]
    (dux_y) = tf.gradients((ux_y,),(colloc_tensor,))[0]
    (duy_x) = tf.gradients((uy_x,),(colloc_tensor,))[0]
    (duy_y) = tf.gradients((uy_y,),(colloc_tensor,))[0]
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.mean_MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.mean_MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.mean_MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.mean_MAX_y

    # governing equations
    return tf.stack((ux_x, ux_y, uy_x, uy_y, uxux_x, uxuy_x, uxuy_y, uyuy_y, p_x, p_y, ux_xx, ux_yy, uy_xx, uy_yy),axis=1)

@tf.function
def mean_grads_cartesian(model_mean,colloc_tensor,ScalingParameters):
    # here we actually calculate the velocity and gradients we need for the FANS equation from the mean model
    # the colloc tensor should be normalized by the FANS equation scale, so we convert to mean model scaling:
    colloc_tensor = colloc_tensor*(ScalingParameters.MAX_x/ScalingParameters.mean_MAX_x)
    # now predict and compute derivatives:
    u_mean = model_mean(colloc_tensor)
    ux = u_mean[:,0]*ScalingParameters.MAX_ux
    uy = u_mean[:,1]*ScalingParameters.MAX_uy

    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/ScalingParameters.mean_MAX_x
    ux_y = dux[:,1]/ScalingParameters.mean_MAX_x

    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/ScalingParameters.mean_MAX_x
    uy_y = duy[:,1]/ScalingParameters.mean_MAX_x

    return tf.stack([ux,uy,ux_x,ux_y,uy_x,uy_y],axis=1)

@tf.function
def FANS_BC_pressure(model_FANS,BC_points):
    up = model_FANS(BC_points)
    # unknowns, pressure fourier modes
    psi_r = up[:,10]*ScalingParameters.MAX_psi
    psi_i = up[:,11]*ScalingParameters.MAX_psi
    return tf.reduce_mean(tf.square(psi_r))+tf.reduce_mean(tf.square(psi_i))

@tf.function
def FANS_BC_no_slip(model_FANS,BC_points):
    up = model_FANS(BC_points)
    # velocity fourier coefficients
    phi_xr = up[:,0]*ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*ScalingParameters.MAX_tau_yy_i
    return tf.reduce_mean(tf.square(phi_xr))+tf.reduce_mean(tf.square(phi_xi))+tf.reduce_mean(tf.square(phi_yr))+tf.reduce_mean(tf.square(phi_yi))+tf.reduce_mean(tf.square(tau_xx_r))+tf.reduce_mean(tf.square(tau_xx_i))+tf.reduce_mean(tf.square(tau_xy_r))+tf.reduce_mean(tf.square(tau_xy_i))+tf.reduce_mean(tf.square(tau_yy_r))+tf.reduce_mean(tf.square(tau_yy_i))

@tf.function
def FANS_boundary_loss(model_FANS,boundary_tuple,ScalingParameters):
    pts_no_slip, pts_pressure = boundary_tuple
    loss_no_slip = FANS_BC_no_slip(model_FANS,pts_no_slip)
    loss_pressure = FANS_BC_pressure(model_FANS,pts_pressure)
    return loss_no_slip + loss_pressure

# fourier NN functions
@tf.function
def FANS_gradients(model_FANS,colloc_tensor,ScalingParameters):
  
    up = model_FANS(colloc_tensor)
    # velocity fourier coefficients
    phi_xr = up[:,0]*ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*ScalingParameters.MAX_tau_yy_i
    # unknowns, pressure fourier modes
    psi_r = up[:,10]*ScalingParameters.MAX_psi
    psi_i = up[:,11]*ScalingParameters.MAX_psi
       
    # phi_xr gradient
    dphi_xr = tf.gradients(phi_xr, colloc_tensor)[0]
    phi_xr_x = dphi_xr[:,0]/ScalingParameters.MAX_x
    phi_xr_y = dphi_xr[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_xr_xx = tf.gradients(phi_xr_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_xr_yy = tf.gradients(phi_xr_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # phi_xi gradient
    dphi_xi = tf.gradients(phi_xi, colloc_tensor)[0]
    phi_xi_x = dphi_xi[:,0]/ScalingParameters.MAX_x
    phi_xi_y = dphi_xi[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_xi_xx = tf.gradients(phi_xi_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_xi_yy = tf.gradients(phi_xi_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # phi_yr gradient
    dphi_yr = tf.gradients(phi_yr, colloc_tensor)[0]
    phi_yr_x = dphi_yr[:,0]/ScalingParameters.MAX_x
    phi_yr_y = dphi_yr[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_yr_xx = tf.gradients(phi_yr_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_yr_yy = tf.gradients(phi_yr_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    
    # phi_yi gradient
    dphi_yi = tf.gradients(phi_yi, colloc_tensor)[0]
    phi_yi_x = dphi_yi[:,0]/ScalingParameters.MAX_x
    phi_yi_y = dphi_yi[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_yi_xx = tf.gradients(phi_yi_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_yi_yy = tf.gradients(phi_yi_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # gradient reynolds stress fourier component, real
    tau_xx_r_x = tf.gradients(tau_xx_r, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    dtau_xy_r = tf.gradients(tau_xy_r, colloc_tensor)[0]
    tau_xy_r_x = dtau_xy_r[:,0]/ScalingParameters.MAX_x
    tau_xy_r_y = dtau_xy_r[:,1]/ScalingParameters.MAX_y
    tau_yy_r_y = tf.gradients(tau_yy_r, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    # gradient reynolds stress fourier component, complex
    tau_xx_i_x = tf.gradients(tau_xx_i, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    dtau_xy_i = tf.gradients(tau_xy_i, colloc_tensor)[0]
    tau_xy_i_x = dtau_xy_i[:,0]/ScalingParameters.MAX_x
    tau_xy_i_y = dtau_xy_i[:,1]/ScalingParameters.MAX_y
    tau_yy_i_y = tf.gradients(tau_yy_i, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # pressure gradients
    dpsi_r = tf.gradients(psi_r, colloc_tensor)[0]
    psi_r_x = dpsi_r[:,0]/ScalingParameters.MAX_x
    psi_r_y = dpsi_r[:,1]/ScalingParameters.MAX_y
    dpsi_i = tf.gradients(psi_i, colloc_tensor)[0]
    psi_i_x = dpsi_i[:,0]/ScalingParameters.MAX_x
    psi_i_y = dpsi_i[:,1]/ScalingParameters.MAX_y


    return tf.stack((phi_xr,phi_xi,phi_yr,phi_yi,phi_xr_x,phi_xr_y,phi_xi_x,phi_xi_y,phi_yr_x,phi_yr_y,phi_yi_x,phi_yi_y,tau_xx_r_x,tau_xx_i_x,tau_xy_r_x,tau_xy_r_y,tau_xy_i_x,tau_xy_i_y,tau_yy_r_y,tau_yy_i_y,psi_r_x,psi_r_y,psi_i_x,psi_i_y,phi_xr_xx,phi_xr_yy,phi_xi_xx,phi_xi_yy,phi_yr_xx,phi_yr_yy,phi_yi_xx,phi_yi_yy),axis=1)

@tf.function
def FANS_cartesian(model_FANS,colloc_tensor, mean_grads,ScalingParameters):
  
    up = model_FANS(colloc_tensor)
    # velocity fourier coefficients
    phi_xr = up[:,0]*ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*ScalingParameters.MAX_tau_yy_i
    # unknowns, pressure fourier modes
    psi_r = up[:,10]*ScalingParameters.MAX_psi
    psi_i = up[:,11]*ScalingParameters.MAX_psi
    
    ux = mean_grads[:,0]
    uy = mean_grads[:,1]
    ux_x = mean_grads[:,2]
    ux_y = mean_grads[:,3]
    uy_x = mean_grads[:,4]
    uy_y = mean_grads[:,5]
    # compute the gradients of the quantities
    
    # phi_xr gradient
    dphi_xr = tf.gradients(phi_xr, colloc_tensor)[0]
    phi_xr_x = dphi_xr[:,0]/ScalingParameters.MAX_x
    phi_xr_y = dphi_xr[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_xr_xx = tf.gradients(phi_xr_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_xr_yy = tf.gradients(phi_xr_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # phi_xi gradient
    dphi_xi = tf.gradients(phi_xi, colloc_tensor)[0]
    phi_xi_x = dphi_xi[:,0]/ScalingParameters.MAX_x
    phi_xi_y = dphi_xi[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_xi_xx = tf.gradients(phi_xi_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_xi_yy = tf.gradients(phi_xi_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # phi_yr gradient
    dphi_yr = tf.gradients(phi_yr, colloc_tensor)[0]
    phi_yr_x = dphi_yr[:,0]/ScalingParameters.MAX_x
    phi_yr_y = dphi_yr[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_yr_xx = tf.gradients(phi_yr_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_yr_yy = tf.gradients(phi_yr_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    
    # phi_yi gradient
    dphi_yi = tf.gradients(phi_yi, colloc_tensor)[0]
    phi_yi_x = dphi_yi[:,0]/ScalingParameters.MAX_x
    phi_yi_y = dphi_yi[:,1]/ScalingParameters.MAX_y
    # and second derivative
    phi_yi_xx = tf.gradients(phi_yi_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    phi_yi_yy = tf.gradients(phi_yi_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # gradient reynolds stress fourier component, real
    tau_xx_r_x = tf.gradients(tau_xx_r, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    dtau_xy_r = tf.gradients(tau_xy_r, colloc_tensor)[0]
    tau_xy_r_x = dtau_xy_r[:,0]/ScalingParameters.MAX_x
    tau_xy_r_y = dtau_xy_r[:,1]/ScalingParameters.MAX_y
    tau_yy_r_y = tf.gradients(tau_yy_r, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    # gradient reynolds stress fourier component, complex
    tau_xx_i_x = tf.gradients(tau_xx_i, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    dtau_xy_i = tf.gradients(tau_xy_i, colloc_tensor)[0]
    tau_xy_i_x = dtau_xy_i[:,0]/ScalingParameters.MAX_x
    tau_xy_i_y = dtau_xy_i[:,1]/ScalingParameters.MAX_y
    tau_yy_i_y = tf.gradients(tau_yy_i, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # pressure gradients
    dpsi_r = tf.gradients(psi_r, colloc_tensor)[0]
    psi_r_x = dpsi_r[:,0]/ScalingParameters.MAX_x
    psi_r_y = dpsi_r[:,1]/ScalingParameters.MAX_y
    dpsi_i = tf.gradients(psi_i, colloc_tensor)[0]
    psi_i_x = dpsi_i[:,0]/ScalingParameters.MAX_x
    psi_i_y = dpsi_i[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_xr = -omega*phi_xi+(phi_xr*ux_x + phi_yr*ux_y+ ux*phi_xr_x +uy*phi_xr_y ) + (tau_xx_r_x + tau_xy_r_y) + psi_r_x - (ScalingParameters.nu_mol)*(phi_xr_xx+phi_xr_yy)  
    f_xi =  omega*phi_xr+(phi_xi*ux_x + phi_yi*ux_y+ ux*phi_xi_x +uy*phi_xi_y ) + (tau_xx_i_x + tau_xy_i_y) + psi_i_x - (ScalingParameters.nu_mol)*(phi_xi_xx+phi_xi_yy)  
    f_yr = -omega*phi_yi+(phi_xr*uy_x + phi_yr*uy_y+ ux*phi_yr_x +uy*phi_yr_y ) + (tau_xy_r_x + tau_yy_r_y) + psi_r_y - (ScalingParameters.nu_mol)*(phi_yr_xx+phi_yr_yy) 
    f_yi =  omega*phi_yr+(phi_xi*uy_x + phi_yi*uy_y+ ux*phi_yi_x +uy*phi_yi_y ) + (tau_xy_i_x + tau_yy_i_y) + psi_i_y - (ScalingParameters.nu_mol)*(phi_yi_xx+phi_yi_yy)  
    f_mr = phi_xr_x + phi_yr_y
    f_mi = phi_xi_x + phi_yi_y

    return f_xr, f_xi, f_yr, f_yi, f_mr, f_mi

def batch_FANS_cartesian(model_FANS,colloc_pts,mean_grads,ScalingParameters,batch_size=1000):
    # this version batches the physics computation for plotting
    n_batch = np.int64(np.ceil(colloc_pts.shape[0]/(1.0*batch_size)))
    f_xr_list = []
    f_xi_list = []
    f_yr_list = []
    f_yi_list = []
    f_mr_list = []
    f_mi_list = []
    progbar = keras.utils.Progbar(n_batch)
    for batch in range(0,n_batch):
        progbar.update(batch+1)
        batch_inds = np.arange(batch*batch_size,np.min([(batch+1)*batch_size,colloc_pts.shape[0]]))
        f_xr, f_xi, f_yr, f_yi, f_mr, f_mi = FANS_cartesian(model_FANS,tf.gather(colloc_pts,batch_inds,axis=0),tf.gather(mean_grads,batch_inds,axis=0),ScalingParameters)
        f_xr_list.append(f_xr)
        f_xi_list.append(f_xi)
        f_yr_list.append(f_yr)
        f_yi_list.append(f_yi)
        f_mr_list.append(f_mr)
        f_mi_list.append(f_mi)
    
    # combine the batches together
    f_xr = tf.concat(f_xr_list,axis=0)
    f_xi = tf.concat(f_xi_list,axis=0)
    f_yr = tf.concat(f_yr_list,axis=0)
    f_yi = tf.concat(f_yi_list,axis=0)
    f_mr = tf.concat(f_mr_list,axis=0)
    f_mi = tf.concat(f_mi_list,axis=0)

    return f_xr, f_xi, f_yr, f_yi, f_mr, f_mi

@ tf.function
def FANS_physics_loss(model_FANS,colloc_pts,mean_grads,ScalingParameters):
    f_xr, f_xi, f_yr, f_yi, f_mr, f_mi = FANS_cartesian(model_FANS,colloc_pts,mean_grads,ScalingParameters)
    return tf.reduce_mean(tf.square(f_xr))+tf.reduce_mean(tf.square(f_xi))+tf.reduce_mean(tf.square(f_yr))+tf.reduce_mean(tf.square(f_yi))+tf.reduce_mean(tf.square(f_mr))+tf.reduce_mean(tf.square(f_mi))

# function wrapper, combine data and physics loss
def colloc_points_function(npoints):

    colloc_limits2 = np.array([[-2.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(npoints)

    colloc_merged = colloc_lhs2
    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print(cylinder_inds.shape)
    colloc_merged = np.delete(colloc_merged,np.nonzero(cylinder_inds[0,:]),axis=0)
    

    rand_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])

    # add a random set of the data points to the collocation array too
    #data_point_inds = np.random.choice(np.arange(i_train.shape[0]),close+far)
    #pts_colloc_train = i_train[data_point_inds,:]

    #f_colloc_train = np.concatenate((rand_colloc_train,pts_colloc_train),axis=0)
    print('colloc.shape',rand_colloc_train.shape)
    return rand_colloc_train

# boundary condition points
def boundary_points_function(n_cyl):
    theta = np.reshape(np.linspace(0,2*np.pi,n_cyl),[n_cyl,])
    ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
    ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
    ns_BC_vec = np.stack((ns_BC_x.flatten(),ns_BC_y.flatten()),axis=1)
    p_BC_x = np.array([10.0,10.0,0.0])/ScalingParameters.MAX_x
    p_BC_y = np.array([-2.0,2.0,0.0])/ScalingParameters.MAX_y
    p_BC_vec = np.stack((p_BC_x.flatten(),p_BC_y.flatten()),axis=1)
    return (ns_BC_vec,p_BC_vec)

## training functions

@tf.function
def compute_loss(x,y,colloc_x,mean_grads,boundary_tuple,ScalingParameters):
    y_pred = model_FANS(x,training=True)
    data_loss = tf.reduce_sum(tf.reduce_mean(tf.square(y_pred[:,0:12]-y),axis=0),axis=0) 
    physics_loss = tf.cast(0.0,tf_dtype)#FANS_physics_loss(model_FANS,colloc_x,mean_grads,ScalingParameters) 
    boundary_loss = tf.cast(0.0,tf_dtype)#FANS_boundary_loss(model_FANS,boundary_tuple,ScalingParameters)

    # dynamic loss weighting, scale based on largest
    max_loss = tf.exp(tf.math.ceil(tf.math.log(1E-30+tf.reduce_max(tf.stack((data_loss,physics_loss,boundary_loss))))))
    log_data = max_loss/tf.exp(tf.math.log(1E-30+data_loss))
    log_physics = max_loss/tf.exp(tf.math.log(1E-30+physics_loss))
    log_boundary = max_loss/tf.exp(tf.math.log(1E-30+boundary_loss))

    total_loss = ScalingParameters.data_loss_coefficient*(1+log_data)*data_loss + ScalingParameters.physics_loss_coefficient*(1+log_physics)*physics_loss + ScalingParameters.boundary_loss_coefficient*(1+log_boundary)*boundary_loss
    return total_loss, data_loss, physics_loss, boundary_loss

@tf.function
def train_step(x,y,colloc_x,mean_grads,boundary_tuple,ScalingParameters):
    with tf.GradientTape() as tape:
        total_loss ,data_loss, physics_loss, boundary_loss = compute_loss(x,y,colloc_x,mean_grads,boundary_tuple,ScalingParameters)

    grads = tape.gradient(total_loss,model_FANS.trainable_weights)
    optimizer.apply_gradients(zip(grads,model_FANS.trainable_weights))
    return total_loss, data_loss, physics_loss, boundary_loss

def fit_epoch(i_train,f_train,colloc_vector,mean_grads,boundary_tuple,ScalingParameters):
    global training_steps
    batches = np.int64(np.ceil(i_train.shape[0]/(1.0*ScalingParameters.batch_size)))
    # sort colloc_points by error
    #i_sampled,o_sampled = data_sample_by_err(i_train,o_train,i_train.shape[0])
    i_sampled = i_train
    f_sampled = f_train
    colloc_rand_indices = np.random.choice(np.arange(colloc_vector.shape[0]),batches*ScalingParameters.colloc_batch_size)
    colloc_rand = tf.gather(colloc_vector,colloc_rand_indices,axis=0)
    grads_rand = tf.gather(mean_grads,colloc_rand_indices,axis=0)

    (BC_wall_epoch,BC_p_epoch) = boundary_tuple
    BC_wall_rand = tf.gather(BC_wall_epoch,np.random.choice(np.arange(BC_wall_epoch.shape[0]),batches*ScalingParameters.boundary_batch_size),axis=0)
  

    progbar = keras.utils.Progbar(batches)
    loss_vec = np.zeros((batches,),np.float64)
    data_vec = np.zeros((batches,),np.float64)
    physics_vec = np.zeros((batches,),np.float64)
    boundary_vec = np.zeros((batches,),np.float64)

    for batch in range(batches):
        progbar.update(batch+1)
        
        i_batch = i_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,i_train.shape[0]]),:]
        f_batch = f_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,f_train.shape[0]]),:]

        colloc_batch = colloc_rand[(batch*ScalingParameters.colloc_batch_size):np.min([(batch+1)*ScalingParameters.colloc_batch_size,colloc_rand.shape[0]]),:]
        grads_batch = grads_rand[(batch*ScalingParameters.colloc_batch_size):np.min([(batch+1)*ScalingParameters.colloc_batch_size,colloc_rand.shape[0]]),:]
        BC_wall_batch = BC_wall_rand[(batch*ScalingParameters.boundary_batch_size):np.min([(batch+1)*ScalingParameters.boundary_batch_size,BC_wall_rand.shape[0]]),:]

        loss_value, data_loss, physics_loss, boundary_loss = train_step(tf.cast(i_batch,tf_dtype),tf.cast(f_batch,tf_dtype),tf.cast(colloc_batch,tf_dtype),tf.cast(grads_batch,tf_dtype),(BC_wall_batch,BC_p_epoch),ScalingParameters) #
        loss_vec[batch] = loss_value.numpy()
        data_vec[batch] = data_loss.numpy()
        physics_vec[batch] = physics_loss.numpy()
        boundary_vec[batch] = boundary_loss.numpy()

    training_steps = training_steps+1
    print('Epoch',str(training_steps),f" Loss: {np.mean(loss_vec):.6e}",f" Data loss: {np.mean(data_vec):.6e}")
    print(f" Physics loss: {np.mean(physics_vec):.6e}",f" Boundary loss: {np.mean(boundary_vec):.6e}",)



def train_LBFGS(model,x,y,colloc_x,mean_grads,boundary_tuple,ScalingParameters):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = compute_loss(x,y,colloc_x,mean_grads,boundary_tuple,ScalingParameters)[0]

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value,"time ",tf.timestamp()-f.last_time,'s')
        f.last_time.assign(tf.timestamp())

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.last_time = tf.Variable(tf.timestamp())
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


# begin main script

if __name__=="__main__":
    keras.backend.set_floatx('float64')
    tf_dtype = tf.float64

    start_time = datetime.now()
    start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

    node_name = platform.node()

    assert len(sys.argv)==5
    job_number = int(sys.argv[1])
    mode_number = int(sys.argv[2])
    supersample_factor = int(sys.argv[3])
    job_hours = int(sys.argv[4])

    job_name = 'mfg_fbc004_f{:d}_S{:d}_j{:03d}'.format(mode_number,supersample_factor,job_number)


    LOCAL_NODE = 'DESKTOP-AMLVDAF'
    if node_name==LOCAL_NODE:
        import matplotlib.pyplot as plot
        import matplotlib
        job_duration = timedelta(hours=job_hours)
        end_time = start_time+job_duration
        useGPU=False    
        SLURM_TMPDIR='C:/projects/pinns_narval/sync/'
        HOMEDIR = 'C:/projects/pinns_narval/sync/'
        PROJECTDIR = HOMEDIR
        sys.path.append('C:/projects/pinns_local/code/')
        # set number of cores to compute on 
    else:
        # parameters for running on compute canada
        job_duration = timedelta(hours=job_hours)
        end_time = start_time+job_duration
        print("This job is: ",job_name)
        HOMEDIR = '/home/coneill/sync/'
        PROJECTDIR = '/home/coneill/projects/def-martinuz/coneill/'
        SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
        sys.path.append(HOMEDIR+'code/')
        # set number of cores to compute on 

    from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center
    #tf.config.threading.set_intra_op_parallelism_threads(16)
    #tf.config.threading.set_inter_op_parallelism_threads(16)
        
    from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists
    from pinns_data_assimilation.lib.file_util import find_highest_numbered_file


    

    # set the paths
    savedir = PROJECTDIR+'output/'+job_name+'_output/'
    create_directory_if_not_exists(savedir)
    global fig_dir
    fig_dir = savedir+'figures/'
    create_directory_if_not_exists(fig_dir)

    # read the data
    base_dir = HOMEDIR+'data/mazi_fixed_grid/'

    fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
    meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
    configFile = h5py.File(base_dir+'configuration.mat','r')
    meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
    reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

    x = np.array(configFile['X_vec'][0,:])
    X_grid = np.array(configFile['X_grid'])
    y = np.array(configFile['X_vec'][1,:])
    Y_grid = np.array(configFile['Y_grid'])
    global d
    d = np.array(configFile['cylinderDiameter'])

    X = np.stack((x,y),axis=1)
    

    ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
    uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

    uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
    uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
    uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

    p = np.array(meanPressureFile['meanPressure'])

    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,1]))

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_i = np.array(np.imag(fourierModeFile['pressureModes'][:,mode_number]))

    print(np.array(fourierModeFile['stressModes']).shape)

    tau_xx_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xx_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xy_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_xy_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_yy_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,2]))
    tau_yy_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,2]))

    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi
    print('Mode Frequency:',str(omega/(2*np.pi)))

    class UserScalingParameters(object):
        pass
    global ScalingParameters
    ScalingParameters = UserScalingParameters()

    ScalingParameters.fs = fs
    ScalingParameters.MAX_x = 20.0
    ScalingParameters.MAX_y = 20.0 # we use the larger of the two spatial scalings
    ScalingParameters.mean_MAX_x = 10.0
    ScalingParameters.mean_MAX_y = 10.0 # if the scaling is different it should be specified here
    ScalingParameters.MAX_ux = np.max(ux.flatten())
    ScalingParameters.MAX_uy = np.max(uy.flatten())
    ScalingParameters.MIN_x = -2.0
    ScalingParameters.MIN_y = -2.0
    ScalingParameters.MIN_ux = np.min(ux.flatten())
    ScalingParameters.MIN_uy = np.min(uy.flatten())
    ScalingParameters.MAX_uxppuxpp = np.max(uxux.flatten())
    ScalingParameters.MAX_uxppuypp = np.max(uxuy.flatten())
    ScalingParameters.MAX_uyppuypp = np.max(uyuy.flatten())

    ScalingParameters.MAX_phi_xr = np.max(phi_xr.flatten())
    ScalingParameters.MAX_phi_xi = np.max(phi_xi.flatten())
    ScalingParameters.MAX_phi_yr = np.max(phi_yr.flatten())
    ScalingParameters.MAX_phi_yi = np.max(phi_yi.flatten())

    ScalingParameters.MAX_tau_xx_r = np.max(tau_xx_r.flatten())
    ScalingParameters.MAX_tau_xx_i = np.max(tau_xx_i.flatten())
    ScalingParameters.MAX_tau_xy_r = np.max(tau_xy_r.flatten())
    ScalingParameters.MAX_tau_xy_i = np.max(tau_xy_i.flatten())
    ScalingParameters.MAX_tau_yy_r = np.max(tau_yy_r.flatten())
    ScalingParameters.MAX_tau_yy_i = np.max(tau_yy_i.flatten())

    ScalingParameters.MAX_p= 1 # estimated maximum pressure, we should 
    ScalingParameters.MAX_psi= 0.2*(omega_0/omega) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
    ScalingParameters.nu_mol = 0.0066667
    ScalingParameters.batch_size = 32
    ScalingParameters.colloc_batch_size = 32
    ScalingParameters.boundary_batch_size = 16
    ScalingParameters.physics_loss_coefficient = tf.cast(3.16E-1,tf_dtype)
    ScalingParameters.boundary_loss_coefficient = tf.cast(1.0,tf_dtype)
    ScalingParameters.data_loss_coefficient = tf.cast(1.0,tf_dtype)

    # prepare the reference dataset for plotting

    global X_plot
    global Y_plot
    X_grid_plot = X_grid
    Y_grid_plot = Y_grid
    X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)

    global F_test_grid
    X_test = X/ScalingParameters.MAX_x


    F_test = np.stack((phi_xr/ScalingParameters.MAX_phi_xr,phi_xi/ScalingParameters.MAX_phi_xi,phi_yr/ScalingParameters.MAX_phi_yr,phi_yi/ScalingParameters.MAX_phi_yi,tau_xx_r/ScalingParameters.MAX_tau_xx_r,tau_xx_i/ScalingParameters.MAX_tau_xx_i,tau_xy_r/ScalingParameters.MAX_tau_xy_r,tau_xy_i/ScalingParameters.MAX_tau_xy_i,tau_yy_r/ScalingParameters.MAX_tau_yy_r,tau_yy_i/ScalingParameters.MAX_tau_yy_i,psi_r/ScalingParameters.MAX_psi,psi_i/ScalingParameters.MAX_psi),axis=1)
    # interpolate the test data to the regular grid for plotting, since this doesnt change we do it on initialization
    

    # reshape the reference data into a grid
    F_test_grid = np.zeros([X_grid_plot.shape[0],X_grid_plot.shape[1],F_test.shape[1]])
    for c in range(F_test.shape[1]):
        F_test_grid[:,:,c] = np.reshape(F_test[:,c],X_grid_plot.shape)

    # need to remove training points inside the cylinder, so make a mask
    cylinder_mask = np.reshape(np.power(x,2.0)+np.power(y,2.0)<=np.power(d/2.0,2.0),[x.shape[0],])

    # if we are downsampling and then upsampling, downsample the source data
    if supersample_factor>0:
        downsample_inds, ndx,ndy = compute_downsample_inds_center(supersample_factor,X_grid[:,0],Y_grid[0,:].transpose())
        x = x[downsample_inds]
        y = y[downsample_inds]
        ux = ux[downsample_inds]
        uy = uy[downsample_inds]
        uxux = uxux[downsample_inds]
        uxuy = uxuy[downsample_inds]
        uyuy = uyuy[downsample_inds]
        phi_xr = phi_xr[downsample_inds]
        phi_xi = phi_xi[downsample_inds]
        phi_yr = phi_yr[downsample_inds]
        phi_yi = phi_yi[downsample_inds]
        tau_xx_r = tau_xx_r[downsample_inds]
        tau_xx_i = tau_xx_i[downsample_inds]
        tau_xy_r = tau_xy_r[downsample_inds]
        tau_xy_i = tau_xy_i[downsample_inds]
        tau_yy_r = tau_yy_r[downsample_inds]
        tau_yy_i = tau_yy_i[downsample_inds]
        psi_r = psi_r[downsample_inds]
        psi_i = psi_i[downsample_inds]
        cylinder_mask = cylinder_mask[downsample_inds]

    print('max_x: ',ScalingParameters.MAX_x)
    print('min_x: ',ScalingParameters.MIN_x)
    print('max_y: ',ScalingParameters.MAX_y)
    print('min_y: ',ScalingParameters.MIN_y)

    # remove the inside quantities
    x = np.delete(x,np.nonzero(cylinder_mask),axis=0)
    y = np.delete(y,np.nonzero(cylinder_mask),axis=0)
    ux = np.delete(ux,np.nonzero(cylinder_mask),axis=0)
    uy = np.delete(uy,np.nonzero(cylinder_mask),axis=0)
    uxux = np.delete(uxux,np.nonzero(cylinder_mask),axis=0)
    uxuy = np.delete(uxuy,np.nonzero(cylinder_mask),axis=0)
    uyuy = np.delete(uyuy,np.nonzero(cylinder_mask),axis=0)

    phi_xr = np.delete(phi_xr,np.nonzero(cylinder_mask),axis=0)
    phi_xi = np.delete(phi_xi,np.nonzero(cylinder_mask),axis=0)
    phi_yr = np.delete(phi_yr,np.nonzero(cylinder_mask),axis=0)
    phi_yi = np.delete(phi_yi,np.nonzero(cylinder_mask),axis=0)
    tau_xx_r = np.delete(tau_xx_r,np.nonzero(cylinder_mask),axis=0)
    tau_xx_i = np.delete(tau_xx_i,np.nonzero(cylinder_mask),axis=0)
    tau_xy_r = np.delete(tau_xy_r,np.nonzero(cylinder_mask),axis=0)
    tau_xy_i = np.delete(tau_xy_i,np.nonzero(cylinder_mask),axis=0)
    tau_yy_r = np.delete(tau_yy_r,np.nonzero(cylinder_mask),axis=0)
    tau_yy_i = np.delete(tau_yy_i,np.nonzero(cylinder_mask),axis=0)
    psi_r = np.delete(psi_r,np.nonzero(cylinder_mask),axis=0)
    psi_i = np.delete(psi_i,np.nonzero(cylinder_mask),axis=0)
    

    # normalize the training data:
    x_train = x/ScalingParameters.MAX_x
    y_train = y/ScalingParameters.MAX_y
    ux_train = ux/ScalingParameters.MAX_ux
    uy_train = uy/ScalingParameters.MAX_uy
    uxux_train = uxux/ScalingParameters.MAX_uxppuxpp
    uxuy_train = uxuy/ScalingParameters.MAX_uxppuypp
    uyuy_train = uyuy/ScalingParameters.MAX_uyppuypp
    phi_xr_train = phi_xr/ScalingParameters.MAX_phi_xr
    phi_xi_train = phi_xi/ScalingParameters.MAX_phi_xi
    phi_yr_train = phi_yr/ScalingParameters.MAX_phi_yr
    phi_yi_train = phi_yi/ScalingParameters.MAX_phi_yi

    tau_xx_r_train = tau_xx_r/ScalingParameters.MAX_tau_xx_r
    tau_xx_i_train = tau_xx_i/ScalingParameters.MAX_tau_xx_i
    tau_xy_r_train = tau_xy_r/ScalingParameters.MAX_tau_xy_r
    tau_xy_i_train = tau_xy_i/ScalingParameters.MAX_tau_xy_i
    tau_yy_r_train = tau_yy_r/ScalingParameters.MAX_tau_yy_r
    tau_yy_i_train = tau_yy_i/ScalingParameters.MAX_tau_yy_i

    psi_r_train = psi_r/ScalingParameters.MAX_psi
    psi_i_train = psi_i/ScalingParameters.MAX_psi

    print("MAX_phi_xr:",ScalingParameters.MAX_phi_xr)
    print("MAX_phi_xi:",ScalingParameters.MAX_phi_xi)
    print("MAX_phi_yr:",ScalingParameters.MAX_phi_yr)
    print("MAX_phi_yi:",ScalingParameters.MAX_phi_yi)
    print("MAX_tau_xx_r:",ScalingParameters.MAX_tau_xx_r)
    print("MAX_tau_xx_i:",ScalingParameters.MAX_tau_xx_i)
    print("MAX_tau_xy_r:",ScalingParameters.MAX_tau_xy_r)
    print("MAX_tau_xy_i:",ScalingParameters.MAX_tau_xy_i)
    print("MAX_tau_yy_r:",ScalingParameters.MAX_tau_yy_r)
    print("MAX_tau_yy_i:",ScalingParameters.MAX_tau_yy_i)

    # the order here must be identical to inside the cost functions
    # LBFGS, since we form a single matrix anyway, dont duplicate the data
    X_train_LBFGS = np.stack((x_train,y_train),axis=1)
    O_train_LBFGS = np.stack((ux_train,uy_train,uxux_train,uxuy_train,uyuy_train),axis=1) # training data
    # with psi
    F_train_LBFGS = np.stack((phi_xr_train,phi_xi_train,phi_yr_train,phi_yi_train,tau_xx_r_train,tau_xx_i_train,tau_xy_r_train,tau_xy_i_train,tau_yy_r_train,tau_yy_i_train,psi_r_train,psi_i_train),axis=1) # training data
    # without psi: 
    #F_train_LBFGS = np.stack((phi_xr_train,phi_xi_train,phi_yr_train,phi_yi_train,tau_xx_r_train,tau_xx_i_train,tau_xy_r_train,tau_xy_i_train,tau_yy_r_train,tau_yy_i_train),axis=1) # training data

    # backprop, duplicate data size so that the epoch length doesnt depend on supersample factor
    if supersample_factor>0:
        X_train_backprop = np.stack((np.concatenate([x_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([y_train for i in range(supersample_factor*supersample_factor)])),axis=1)
        O_train_backprop = np.stack((np.concatenate([ux_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([uy_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([uxux_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([uxuy_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([uyuy_train for i in range(supersample_factor*supersample_factor)])),axis=1) # training data
        # with psi
        F_train_backprop = np.stack((np.concatenate([phi_xr_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([phi_xi_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([phi_yr_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([phi_yi_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_xx_r_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_xx_i_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_xy_r_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_xy_i_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_yy_r_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([tau_yy_i_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([psi_r_train for i in range(supersample_factor*supersample_factor)]),np.concatenate([psi_i_train for i in range(supersample_factor*supersample_factor)])),axis=1) # training data
        # without psi
    else:
        X_train_backprop = 1.0*X_train_LBFGS
        O_train_backprop = 1.0*O_train_LBFGS
        F_train_backprop = 1.0*F_train_LBFGS
    print('X_train.shape: ',X_train_backprop.shape)
    print('O_train.shape: ',O_train_backprop.shape)
    # the order here must be identical to inside the cost functions
    boundary_tuple  = boundary_points_function(720)
    X_colloc = colloc_points_function(20000)


    tf_device_string ='/GPU:0'
    # create the NNs
    from pinns_data_assimilation.lib.layers import QuadraticInputPassthroughLayer
    from pinns_data_assimilation.lib.layers import InputPassthroughLayer
    from pinns_data_assimilation.lib.layers import FourierPassthroughEmbeddingLayer
    from pinns_data_assimilation.lib.layers import FourierPassthroughReductionLayer
    # load the saved mean model

    with tf.device('/CPU:0'):
        mean_model_filename,mean_training_steps = find_highest_numbered_file(PROJECTDIR+'output/mfg_fbc003_001_S'+str(supersample_factor)+'/mfg_fbc003_001_S'+str(supersample_factor)+'_ep','[0-9]*','_model.h5')
        model_mean = keras.models.load_model(mean_model_filename,custom_objects={'QuadraticInputPassthroughLayer':QuadraticInputPassthroughLayer,'FourierPassthroughEmbeddingLayer':FourierPassthroughEmbeddingLayer,'FourierPassthroughReductionLayer':FourierPassthroughReductionLayer})
        model_mean.trainable=False

    # get the values for the mean_data tensor
    mean_data = mean_grads_cartesian(model_mean,X_colloc,ScalingParameters) # values at the collocation points
    mean_data_plot = mean_grads_cartesian(model_mean,X_plot*(1.0/ScalingParameters.MAX_x),ScalingParameters) # at the plotting points




    
    # check if the model has been created before, if so load it

    optimizer = keras.optimizers.SGD(learning_rate=1E-4,momentum=0.0)
    embedding_wavenumber_vector = np.linspace(0,3*np.pi*ScalingParameters.MAX_x,60) # in normalized domain! in this case the wavenumber of the 3rd harmonic is roughly pi rad/s so we double that
    # we need to check if there are already checkpoints for this job
    model_file,temp_trainingsteps = find_highest_numbered_file(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep','[0-9]*','_model.h5')
    # check if the model has been created, if so check if weights exist
    if model_file is not None:
        with tf.device(tf_device_string):
            model_FANS,training_steps = load_custom()
    else:
        # create a new model
        training_steps = 0
        with tf.device(tf_device_string):        
            inputs = keras.Input(shape=(2,),name='coordinates')
            #lo = FourierPassthroughEmbeddingLayer(embedding_wavenumber_vector,2)(lo)
            lo = QuadraticInputPassthroughLayer(60,2,activation='tanh',dtype=tf_dtype)(inputs)
            for i in range(4):
                lo = QuadraticInputPassthroughLayer(60,2,activation='tanh',dtype=tf_dtype)(lo)
            lo = FourierPassthroughReductionLayer(embedding_wavenumber_vector,30)(lo)
            for i in range(4):
                lo = QuadraticInputPassthroughLayer(100,2,activation='tanh',dtype=tf_dtype)(lo)
            outputs = keras.layers.Dense(12,activation='linear',name='dynamical_quantities')(lo)
            model_FANS = keras.Model(inputs=inputs,outputs=outputs)
            model_FANS.summary()
            # save the model only on startup
            model_FANS.save(savedir+job_name+'_ep0_model.h5')


    # setup the training data
    # this time we randomly shuffle the order of X and O
    rng = np.random.default_rng()

    #if node_name==LOCAL_NODE:
    #    plot_err()
    #    plot_NS_residual()
    #    plot_frequencies()
    #    exit()

    # train the network
    last_epoch_time = datetime.now()
    average_epoch_time=timedelta(minutes=10)
    backprop_flag=False
    while backprop_flag:
        # regular training with physics
        lr_schedule = np.array([1E-5, 3.16E-6, 1E-6,    0.0])
        ep_schedule = np.array([0,      30,     100,    200])
        phys_schedule = np.array([3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1])
        #lr_schedule = np.array([3.16E-7, 1E-7,    3.16E-8, 1E-8,    0.0])
        #ep_schedule = np.array([0,       50,      150,   300,       400,])
        #phys_schedule = np.array([3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1, 3.16E-1])

        # reset the correct learing rate on load
        i_temp = 0
        for i in range(len(ep_schedule)):
            if training_steps>=ep_schedule[i]:
                i_temp = i
        if lr_schedule[i_temp]==0.0:
            backprop_flag=False
        ScalingParameters.physics_loss_coefficient = tf.cast(phys_schedule[i_temp],tf.float64)

        print('physics loss =',str(ScalingParameters.physics_loss_coefficient))
        print('learning rate =',str(lr_schedule[i_temp]))

        while backprop_flag:
            for i in range(1,len(ep_schedule)):
                if training_steps==ep_schedule[i]:
                    if lr_schedule[i] == 0.0:
                        backprop_flag=False
                        # we will do one last epoch then lbfgs
                    keras.backend.set_value(optimizer.learning_rate, lr_schedule[i])
                    print('epoch',str(training_steps))
                    ScalingParameters.physics_loss_coefficient = tf.cast(phys_schedule[i],tf.float64)

                    print('physics loss =',str(ScalingParameters.physics_loss_coefficient))
                    print('learning rate =',str(lr_schedule[i]))               

            fit_epoch(X_train_backprop,F_train_backprop,X_colloc,mean_data,boundary_tuple,ScalingParameters)

            if np.mod(training_steps,10)==0:
                if node_name==LOCAL_NODE:
                    plot_NS_residual()
                    plot_err()
                    #plot_inlet_profile()
            if np.mod(training_steps,10)==0:
                model_FANS.save_weights(savedir+job_name+'_ep'+str(np.uint(training_steps))+'.weights.h5')
            if np.mod(training_steps,50)==0:
                    # rerandomize the collocation points 
                boundary_tuple = boundary_points_function(720)
                X_colloc = colloc_points_function(20000)
                mean_data = mean_grads_cartesian(model_mean,X_colloc,ScalingParameters)
            
            # check if we are out of time
            average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
            if (datetime.now()+average_epoch_time)>end_time:
                model_FANS.save_weights(savedir+job_name+'_ep'+str(np.uint(training_steps))+'.weights.h5')
                model_FANS.save(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_model.h5')
                exit()
            last_epoch_time = datetime.now()


    LBFGS_steps = 333
    LBFGS_epochs = 3*LBFGS_steps
    if True:
        # final polishing of solution using LBFGS
        import tensorflow_probability as tfp
        L_iter = 0
        boundary_tuple = boundary_points_function(720)
        X_colloc = colloc_points_function(10000) # one A100 max = 60k?
        mean_data = mean_grads_cartesian(model_mean,X_colloc,ScalingParameters)
        func = train_LBFGS(model_FANS,tf.cast(X_train_LBFGS,tf_dtype),tf.cast(F_train_LBFGS,tf_dtype),X_colloc,mean_data,boundary_tuple,ScalingParameters)
        init_params = tf.dynamic_stitch(func.idx, model_FANS.trainable_variables)
                
        while True:
            
            last_epoch_time = datetime.now()
            # train the model with L-BFGS solver
            results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,f_relative_tolerance=1E-16,stopping_condition=tfp.optimizer.converged_all)
            func.assign_new_model_parameters(results.position)
            init_params = tf.dynamic_stitch(func.idx, model_FANS.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
            training_steps = training_steps + LBFGS_epochs
            L_iter = L_iter+1
                
            # after training, the final optimized parameters are still in results.position
            # so we have to manually put them back to the model
    
            # check if we are out of time
            average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
            if (datetime.now()+average_epoch_time)>end_time:
                #save_pred()
                model_FANS.save_weights(savedir+job_name+'_ep'+str(np.uint(training_steps))+'.weights.h5')
                model_FANS.save(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_model.h5')
                exit()

            if np.mod(L_iter,10)==0:
                model_FANS.save_weights(savedir+job_name+'_ep'+str(np.uint(training_steps))+'.weights.h5')
                boundary_tuple = boundary_points_function(720)
                X_colloc = colloc_points_function(10000)
                mean_data = mean_grads_cartesian(model_mean,X_colloc,ScalingParameters)
                func = train_LBFGS(model_FANS,tf.cast(X_train_LBFGS,tf_dtype),tf.cast(F_train_LBFGS,tf_dtype),X_colloc,mean_data,boundary_tuple,ScalingParameters)
                init_params = tf.dynamic_stitch(func.idx, model_FANS.trainable_variables)

            if node_name==LOCAL_NODE:
                #save_pred()
                model_FANS.save_weights(savedir+job_name+'_ep'+str(np.uint(training_steps))+'.weights.h5')
                plot_err()
                #plot_inlet_profile()
                plot_NS_residual()

