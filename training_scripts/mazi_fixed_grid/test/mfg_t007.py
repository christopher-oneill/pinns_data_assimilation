


import tensorflow as tf

import tensorflow.keras as keras
import h5py
#import tensorflow_probability as tfp
from smt.sampling_methods import LHS

keras.backend.set_floatx('float64')
tf_dtype=tf.float64
import numpy as np

from datetime import datetime
from datetime import timedelta

import sys
import os
import platform


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


    NN_grads = RANS_gradients(model_RANS,ScalingParameters,i_test)
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
        if saveFig:
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
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_NS_residual_finite_difference.png',dpi=300)
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
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_NS_residual_finite_difference_log.png',dpi=300)
    plot.close(1)

    # create profile plot

def plot_NS_residual():
    # NS residual
    global X_grid
    global Y_grid
    global model_RANS
    global ScalingParameters
    global i_test
    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)
    mx,my,mass = batch_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,tf.cast(i_test,tf_dtype),1000)
    mx = 1.0*np.reshape(mx,X_grid.shape)
    mx[cylinder_mask] = np.NaN
    my = 1.0*np.reshape(my,X_grid.shape)
    my[cylinder_mask] = np.NaN
    mass = 1.0*np.reshape(mass,X_grid.shape)
    mass[cylinder_mask] = np.NaN

    plot.figure(training_steps)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    mx_min = np.nanpercentile(mx.ravel(),0.1)
    mx_max = np.nanpercentile(mx.ravel(),99.9)
    mx_level = np.max([abs(mx_min),abs(mx_max)])
    mx_levels = np.linspace(-mx_level,mx_level,21)
    plot.contourf(X_grid,Y_grid,mx,levels=mx_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    my_min = np.nanpercentile(my.ravel(),0.1)
    my_max = np.nanpercentile(my.ravel(),99.9)
    my_level = np.max([abs(my_min),abs(my_max)])
    my_levels = np.linspace(-my_level,my_level,21)
    plot.contourf(X_grid,Y_grid,my,levels=my_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    mass_min = np.nanpercentile(mass.ravel(),0.1)
    mass_max = np.nanpercentile(mass.ravel(),99.9)
    mass_level = np.max([abs(mass_min),abs(mass_max)])
    mass_levels = np.linspace(-mass_level,mass_level,21)
    plot.contourf(X_grid,Y_grid,mass,levels=mass_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_NS_residual.png',dpi=300)


def plot_err():
    global p_grid
    global X_grid
    global Y_grid
    global i_test
    
    global o_test_grid
    global saveFig
    global fig_dir
    global d
    global model_RANS
    global training_steps
    global ScalingParameters

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    o_test_grid_temp = 1.0*o_test_grid
    o_test_grid_temp[cylinder_mask,:] = np.NaN

    pred_test = model_RANS(i_test,training=False)
    
    pred_test_grid = 1.0*np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask,:] = np.NaN

    i_train_plot = i_train_LBFGS*ScalingParameters.MAX_x

    err_test = o_test_grid_temp-pred_test_grid
    plot_save_exts = ['_ux.png','_uy.png','_uxux.png','_uxuy.png','_uyuy.png','_p.png']
    # quantities
    for i in range(6):
        plot.figure(1)
        plot.title('Full Resolution')
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        if (supersample_factor>1) and (i<5):
            plot.scatter(i_train_plot[:,0],i_train_plot[:,1],2,'k','.')
        
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        e_test_min = np.nanpercentile(err_test[:,:,i].ravel(),0.1)
        e_test_max = np.nanpercentile(err_test[:,:,i].ravel(),99.9)
        e_test_level = np.max([abs(e_test_min),abs(e_test_max)])
        e_test_levels = np.linspace(-e_test_level,e_test_level,21)
        plot.contourf(X_grid,Y_grid,err_test[:,:,i],levels=e_test_levels,extend='both')
        plot.set_cmap('bwr')
        plot.colorbar()
        if saveFig:
            plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts[i],dpi=300)
        plot.close(1)

    # also plot the profiles
    profile_locations = np.linspace(-1.5,9.5,22) # spacing of 0.5
    X_line_locations = X_grid[:,0]

    X_distance_matrix = np.power(np.power(np.reshape(X_line_locations,[1,X_line_locations.size])-np.reshape(profile_locations,[profile_locations.size,1]),2.0),0.5)
    # find index of closest data line  
    line_inds = np.argmin(X_distance_matrix,axis=1)
    profile_locations = X_grid[line_inds,0]



    #point_locations = np.linspace(-2,2,40)
    #Y_line_locations = Y_plot[:,0]
    #Y_distance_matrix = np.power(np.power(np.reshape(Y_line_locations,[Y_line_locations.size,1])-np.reshape(point_locations,[1,point_locations.size]),2.0),0.5)
    #point_inds = np.argmin(Y_distance_matrix,axis=0)
    point_locations = Y_grid[0,:]

    plot_save_exts2 = ['_ux_profile.png','_uy_profile.png','_uxux_profile.png','_uxuy_profile.png','_uyuy_profile.png','_p_profile.png',]
    x_offset = np.array([-1.0/ScalingParameters.MAX_ux,0,0,0,0,0])
    x_scale = np.array([0.5,0.5,0.5,0.5,0.5,0.5,])
    err_scale = np.nanmax(np.abs(err_test),axis=(0,1))
    
    for i in range(6):
        plot.figure(1)
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,o_test_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        
        plot.set_cmap('bwr')
        plot.xlim(-2,10)
        
        plot.subplot(3,1,2)
        for k in range(profile_locations.shape[0]):
            plot.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
            plot.plot((o_test_grid_temp[line_inds[k],:,i]+x_offset[i])*x_scale[i]+profile_locations[k],point_locations,'-k',linewidth=0.5)
            plot.plot((pred_test_grid[line_inds[k],:,i]+x_offset[i])*x_scale[i]+profile_locations[k],point_locations,'-r',linewidth=0.5)
        plot.xlim(-2,10)
        plot.subplot(3,1,3)
        for k in range(profile_locations.shape[0]):
            plot.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
            plot.plot((0.5/err_scale[i])*err_test[line_inds[k],:,i]+profile_locations[k],point_locations,'-r',linewidth=0.5)
        plot.text(4.5,1.7,"x-Scaled by (0.5/MaxErr). MaxErr={txterr:.4f}".format(txterr=err_scale[i]),fontsize=7.0)
        plot.xlim(-2,10)
        plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts2[i],dpi=300)
        plot.close(1)


# plotting functions
# save and load functions

def save_pred():
    global i_test
    global savedir
    global ScalingParameters
    global training_steps
    pred = model_RANS(tf.cast(i_test,tf_dtype),training=False)
    h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
    if ScalingParameters.physics_loss_coefficient!=0:
        t_mx,t_my,t_mass,t_cr = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,tf.cast(i_test,tf_dtype))
        h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()

def load_custom():
    model_filename,model_training_steps = find_highest_numbered_file(savedir+job_name+'_ep','[0-9]*','_model.h5')
    model_RANS = keras.models.load_model(model_filename,custom_objects={'QuadraticInputPassthroughLayer':QuadraticInputPassthroughLayer,'FourierPassthroughEmbeddingLayer':FourierPassthroughEmbeddingLayer,'FourierPassthroughReductionLayer':FourierPassthroughReductionLayer})
    # check if the weights are newer
    checkpoint_filename,weights_training_steps = find_highest_numbered_file(savedir+job_name+'_ep','[0-9]*','.weights.h5')

    if checkpoint_filename is not None:
        if weights_training_steps>model_training_steps:
            model_RANS.load_weights(checkpoint_filename)
            training_steps = weights_training_steps
        else:
            training_steps = model_training_steps
    
    model_RANS.summary()
    print('Model Loaded. Epoch',str(training_steps))
    #optimizer.build(model_RANS.trainable_variables)
    return model_RANS, training_steps

# import the physics
@tf.function
def BC_RANS_reynolds_stress_pressure(model_RANS,BC_points):
    up = model_RANS(BC_points)
    # knowns
    # unknowns
    p = up[:,5]
    return tf.reduce_mean(tf.square(p))

@tf.function
def BC_RANS_wall(model_RANS,ScalingParameters,BC_points):
    wall_coord = BC_points[:,0:2]
    #wall_angle = BC_points[:,2]
    up = model_RANS(wall_coord)
    # knowns
    ux = up[:,0]
    uy = up[:,1]
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    #p = up[:,5]
    #(dp,) = tf.gradients((p,),(wall_coord)) 
    #p_x = dp[:,0]/ScalingParameters.MAX_x
    #p_y = dp[:,1]/ScalingParameters.MAX_y
    #grad_p_norm = p_x*tf.cos(wall_angle)+p_y*tf.sin(wall_angle)

    return tf.reduce_mean(tf.square(ux))+tf.reduce_mean(tf.square(uy))+tf.reduce_mean(tf.square(uxppuxpp))+tf.reduce_mean(tf.square(uxppuypp))+tf.reduce_mean(tf.square(uyppuypp)) #+tf.square(grad_p_norm)


@tf.function
def RANS_gradients(model_RANS,ScalingParameters,colloc_tensor):
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

    ux_x = dux[:,0]/ScalingParameters.MAX_x
    ux_y = dux[:,1]/ScalingParameters.MAX_y
    
    # uy gradient
    uy_x = duy[:,0]/ScalingParameters.MAX_x
    uy_y = duy[:,1]/ScalingParameters.MAX_y


    # gradient unmodeled reynolds stresses
    uxux_x = duxux[:,0]/ScalingParameters.MAX_x
    uxuy_x = duxuy[:,0]/ScalingParameters.MAX_x
    uxuy_y = duxuy[:,1]/ScalingParameters.MAX_y
    uyuy_y = duyuy[:,1]/ScalingParameters.MAX_y

    # pressure gradients
    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y

    # second gradients
    (dux_x) = tf.gradients((ux_x,),(colloc_tensor,))[0]
    (dux_y) = tf.gradients((ux_y,),(colloc_tensor,))[0]
    (duy_x) = tf.gradients((uy_x,),(colloc_tensor,))[0]
    (duy_y) = tf.gradients((uy_y,),(colloc_tensor,))[0]
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    return tf.stack((ux_x, ux_y, uy_x, uy_y, uxux_x, uxuy_x, uxuy_y, uyuy_y, p_x, p_y, ux_xx, ux_yy, uy_xx, uy_yy),axis=1)

@tf.function
def RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,colloc_tensor):
    # in this version we try to trace the graph only twice (first gradient, second gradient)
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

    ux_x = dux[:,0]/ScalingParameters.MAX_x
    ux_y = dux[:,1]/ScalingParameters.MAX_y
    
    # uy gradient
    uy_x = duy[:,0]/ScalingParameters.MAX_x
    uy_y = duy[:,1]/ScalingParameters.MAX_y


    # gradient unmodeled reynolds stresses
    uxux_x = duxux[:,0]/ScalingParameters.MAX_x
    uxuy_x = duxuy[:,0]/ScalingParameters.MAX_x
    uxuy_y = duxuy[:,1]/ScalingParameters.MAX_y
    uyuy_y = duyuy[:,1]/ScalingParameters.MAX_y

    # pressure gradients
    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y

    # second gradients
    (dux_x) = tf.gradients((ux_x,),(colloc_tensor,))[0]
    (dux_y) = tf.gradients((ux_y,),(colloc_tensor,))[0]
    (duy_x) = tf.gradients((uy_x,),(colloc_tensor,))[0]
    (duy_y) = tf.gradients((uy_y,),(colloc_tensor,))[0]
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    f_cr = tf.multiply(tf.cast(tf.math.less(uxux,tf.cast(0.0,tf_dtype)),tf_dtype),tf.abs(uxux))+tf.multiply(tf.cast(tf.math.less(uyuy,tf.cast(0.0,tf_dtype)),tf_dtype),tf.abs(uyuy)) # tr(ReStress)>0 by defn
    
    return f_x, f_y, f_mass, f_cr

def batch_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,colloc_tensor,batch_size=1000):
    n_batch = np.int64(np.ceil(colloc_tensor.shape[0]/(1.0*batch_size)))
    f_x_list = []
    f_y_list = []
    f_m_list = []
    progbar = keras.utils.Progbar(n_batch)
    for batch in range(0,n_batch):
        progbar.update(batch+1)
        batch_inds = np.arange(batch*batch_size,np.min([(batch+1)*batch_size,colloc_tensor.shape[0]]))
        f_x, f_y, f_m, c_r = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,tf.gather(colloc_tensor,batch_inds))
        f_x_list.append(f_x)
        f_y_list.append(f_y)
        f_m_list.append(f_m)
    
    # combine the batches together
    f_x = tf.concat(f_x_list,axis=0)
    f_y = tf.concat(f_y_list,axis=0)
    f_m = tf.concat(f_m_list,axis=0)

    return f_x, f_y, f_m

@tf.function
def RANS_physics_loss(model_RANS,ScalingParameters,colloc_points,): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    mx,my,mass,cr = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,colloc_points)
    physical_loss1 = tf.reduce_mean(tf.square(mx))
    physical_loss2 = tf.reduce_mean(tf.square(my))
    physical_loss3 = tf.reduce_mean(tf.square(mass))
    constraint_loss = tf.reduce_mean(tf.square(cr)) # non-negative reynolds stresses
    physics_loss = physical_loss1 + physical_loss2 + physical_loss3 + constraint_loss
    return physics_loss




@tf.function
def RANS_boundary_loss(model_RANS,ScalingParameters,boundary_tuple):
    (BC_p,BC_wall) = boundary_tuple
    BC_pressure_loss = BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) 
    BC_wall_loss = BC_RANS_wall(model_RANS,ScalingParameters,BC_wall)
                       
    boundary_loss = (BC_wall_loss + BC_pressure_loss) #  + BC_wall2 +   + BC_cylinder_inside_loss + BC_inlet_loss2
    return boundary_loss


# collloc and boundary functions

# define the collocation points

def colloc_points_function(b):
    colloc_limits3 = np.array([[-2.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs3 = LHS(xlimits=colloc_limits3)
    colloc_lhs3 = colloc_sample_lhs3(b)

    colloc_merged = colloc_lhs3

    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print('points inside_cylinder: ',np.sum(cylinder_inds))
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])
    np.random.shuffle(f_colloc_train)
    return tf.cast(f_colloc_train,tf_dtype)


def boundary_points_function(cyl):
    # define boundary condition points
    theta = np.linspace(0,2*np.pi,cyl)
    ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
    ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
    cyl_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1),theta.reshape(-1,1)))

    p_BC_x = np.array([10.0,10.0,0.0])/ScalingParameters.MAX_x
    p_BC_y = np.array([-2.0,2.0,0.0])/ScalingParameters.MAX_y
    p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

    # random points outside the domain for no reynolds stress condition
    return tf.cast(p_BC_vec,tf_dtype),tf.cast(cyl_BC_vec,tf_dtype)

# training functions




start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

assert len(sys.argv)==4

job_number = int(sys.argv[1])
supersample_factor = int(sys.argv[2])
job_hours = int(sys.argv[3])

global job_name 
job_name = 'mfg_t002_{:03d}_S{:d}'.format(job_number,supersample_factor)

job_duration = timedelta(hours=job_hours,minutes=0)
end_time = start_time+job_duration

LOCAL_NODE = 'DESKTOP-GMOIE9C'
if node_name==LOCAL_NODE:
    import matplotlib
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_narval/sync/'
    HOMEDIR = 'C:/projects/pinns_narval/sync/'
    PROJECTDIR = HOMEDIR
    sys.path.append('C:/projects/pinns_local/code/')
else:
    # parameters for running on compute canada   
    HOMEDIR = '/home/coneill/sync/'
    PROJECTDIR = '/home/coneill/projects/def-martinuz/coneill/'
    sys.path.append(HOMEDIR+'code/')



#from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.file_util import find_highest_numbered_file
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center
from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
global savedir
savedir = PROJECTDIR+'output/'+job_name+'/'
create_directory_if_not_exists(savedir)
global fig_dir
fig_dir = savedir + 'figures/'
create_directory_if_not_exists(fig_dir)

reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')

global colloc_sampled

global X_grid
global Y_grid

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
global d
d = np.array(configFile['cylinderDiameter'])

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

MAX_ux = np.max(ux.ravel())
MAX_uy = np.max(uy.ravel())
MAX_uxux = np.max(uxux.ravel())
MAX_uxuy = np.max(uxuy.ravel())
MAX_uyuy = np.max(uyuy.ravel())
MAX_p = 1.0



# remove training points inside the cylinder
cylinder_mask = np.reshape(np.power(x,2.0)+np.power(y,2.0)<=np.power(d/2.0,2.0),[x.shape[0],])

# actually we need to solve the pressure inside the cylinder, which means we should supply that the inside quantities are zero
ux[cylinder_mask] = 0.0
uy[cylinder_mask] = 0.0
uxux[cylinder_mask] = 0.0
uxuy[cylinder_mask] = 0.0
uyuy[cylinder_mask] = 0.0

MAX_x = np.max(X_grid)
MIN_x = np.min(X_grid)
MAX_y = np.max(Y_grid)
MIN_y = np.min(Y_grid)

x_test = 1.0*x/MAX_x
y_test = 1.0*y/MAX_x
global i_test
i_test = np.stack((x_test,y_test),axis=1)

global p_grid
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]

o_test_grid = np.stack([np.reshape(i,[X_grid.shape[0],X_grid.shape[1]]) for i in [ux/MAX_ux,uy/MAX_uy,uxux/MAX_uxux,uxuy/MAX_uxuy,uyuy/MAX_uyuy,p/MAX_p]],axis=2)


fs=10.0
# create a dummy object to contain all the scaling parameters
class UserScalingParameters(object):
    pass
global ScalingParameters
ScalingParameters = UserScalingParameters()
ScalingParameters.fs = fs
ScalingParameters.MAX_x = np.max(x.flatten())
ScalingParameters.MAX_y = ScalingParameters.MAX_x # we scale based on the largest spatial dimension
ScalingParameters.MAX_ux = MAX_ux # we scale based on the max of the whole output array
ScalingParameters.MAX_uy = MAX_uy
ScalingParameters.MIN_x = np.min(x.flatten())
ScalingParameters.MIN_y = np.min(y.flatten())
ScalingParameters.MIN_ux = np.min(ux.flatten())
ScalingParameters.MIN_uy = np.min(uy.flatten())
ScalingParameters.MAX_uxppuxpp = MAX_uxux
ScalingParameters.MAX_uxppuypp = MAX_uxuy
ScalingParameters.MAX_uyppuypp = MAX_uyuy
ScalingParameters.nu_mol = tf.cast(0.0066667,tf_dtype)
ScalingParameters.MAX_p= MAX_p # estimated maximum pressure, we should
ScalingParameters.batch_size = 32
ScalingParameters.colloc_batch_size = 32
ScalingParameters.boundary_batch_size = 16
ScalingParameters.physics_loss_coefficient = tf.cast(1.0,tf_dtype)
ScalingParameters.boundary_loss_coefficient = tf.cast(1.0,tf_dtype)
ScalingParameters.data_loss_coefficient = tf.cast(1.0,tf_dtype)

print('MAX_x: ',ScalingParameters.MAX_x)
print('MIN_x: ',ScalingParameters.MIN_x)
print('MAX_y: ',ScalingParameters.MAX_y)
print('MAX_x: ',ScalingParameters.MIN_y)


# if we are downsampling and then upsampling, downsample the source data
if supersample_factor>1:
    n_x = np.array(configFile['x_grid']).size
    n_y = np.array(configFile['y_grid']).size
    downsample_inds, ndx,ndy = compute_downsample_inds_center(supersample_factor,X_grid[:,0],Y_grid[0,:].transpose())
    x = x[downsample_inds]
    y = y[downsample_inds]
    ux = ux[downsample_inds]
    uy = uy[downsample_inds]
    uxux = uxux[downsample_inds]
    uxuy = uxuy[downsample_inds]
    uyuy = uyuy[downsample_inds]

    cylinder_mask = cylinder_mask[downsample_inds]

# remove the inside quantities
x = np.delete(x,np.nonzero(cylinder_mask),axis=0)
y = np.delete(y,np.nonzero(cylinder_mask),axis=0)
ux = np.delete(ux,np.nonzero(cylinder_mask),axis=0)
uy = np.delete(uy,np.nonzero(cylinder_mask),axis=0)
uxux = np.delete(uxux,np.nonzero(cylinder_mask),axis=0)
uxuy = np.delete(uxuy,np.nonzero(cylinder_mask),axis=0)
uyuy = np.delete(uyuy,np.nonzero(cylinder_mask),axis=0)



# for LBFGS we don't need to duplicate since all points and collocs are evaluated in a single step
o_train_LBFGS = np.stack((ux/MAX_ux,uy/MAX_uy,uxux/MAX_uxux,uxuy/MAX_uxuy,uyuy/MAX_uyuy),axis=1)
i_train_LBFGS = np.stack((x/MAX_x,y/MAX_x),axis=1)

# copy the arrays if supersample factor is >1 so that the data size is approximately consistent
if supersample_factor>0:
    o_train_backprop = np.stack((np.concatenate([ux for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_ux,np.concatenate([uy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uy,np.concatenate([uxux for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uxux,np.concatenate([uxuy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uxuy,np.concatenate([uyuy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uyuy),axis=1)
    i_train_backprop = np.stack((np.concatenate([x for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_x, np.concatenate([y for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_x),axis=1)
else:
    o_train_backprop = 1.0*o_train_LBFGS
    i_train_backprop = 1.0*i_train_LBFGS

global colloc_vector
colloc_vector = colloc_points_function(25000)

strategy = tf.distribute.MirroredStrategy()
BUFFER_SIZE = o_train_backprop.shape[0]

BATCH_SIZE_PER_REPLICA = 64
COLLOC_PER_REPLICA = 128
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
COLLOC_BATCH_SIZE = COLLOC_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

# compute the number of global batches
data_batches = np.int32(np.ceil(o_train_backprop.shape[0]/GLOBAL_BATCH_SIZE))

distributed_dataset = []
# construct the batches for each replica
for batch in range(data_batches):
    replica_list = []
    for replica in range(strategy.num_replicas_in_sync):
        i_batch = i_train_backprop[(batch*GLOBAL_BATCH_SIZE+replica*BATCH_SIZE_PER_REPLICA):np.min([batch*GLOBAL_BATCH_SIZE+(replica+1)*BATCH_SIZE_PER_REPLICA,i_train_backprop.shape[0]]),:]
        o_batch = o_train_backprop[(batch*GLOBAL_BATCH_SIZE+replica*BATCH_SIZE_PER_REPLICA):np.min([batch*GLOBAL_BATCH_SIZE+(replica+1)*BATCH_SIZE_PER_REPLICA,o_train_backprop.shape[0]]),:]
        
        replica_list.append(tf.types.experimental.distributed.PerReplica((i_batch,o_batch)))
    distributed_dataset.append(tuple(replica_list))

# we need to distribute
def batch_fn(ctx):
    this_replica = ctx.replica_id_in_sync_group
    return distributed_dataset[:][this_replica]

distributed_values = strategy.experimental_distribute_values_from_function(batch_fn)
for _ in range(4):
  result = strategy.run(lambda x: x, args=(distributed_values,))
  print(result)

exit()
#distributed_values = (strategy.experimental_distribute_values_from_function(batch_fn))

#local_result = strategy.experimental_local_results(distributed_values)

#print(local_result)


boundary_tuple = boundary_points_function(720)
colloc_vector = colloc_points_function(25000)

global training_steps
global model_RANS
# model creation
tf_device_string ='/GPU:0'



from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import QresBlock
from pinns_data_assimilation.lib.layers import QresBlock2
from pinns_data_assimilation.lib.file_util import get_filepaths_with_glob
from pinns_data_assimilation.lib.layers import QuadraticInputPassthroughLayer
from pinns_data_assimilation.lib.layers import InputPassthroughLayer
from pinns_data_assimilation.lib.layers import FourierPassthroughEmbeddingLayer
from pinns_data_assimilation.lib.layers import FourierPassthroughReductionLayer

embedding_wavenumber_vector = np.linspace(0,3*np.pi*ScalingParameters.MAX_x,60)


with strategy.scope():

    # load the model
    model_filename,model_training_steps = find_highest_numbered_file(savedir+job_name+'_ep','[0-9]*','_model.h5')
    if model_filename is not None:
        model_RANS,training_steps = load_custom()
    else: 
        training_steps = 0
        
        inputs = keras.Input(shape=(2,),name='coordinates')
        lo = ResidualLayer(100,'tanh')(inputs)
        for i in range(9):
            lo=ResidualLayer(100,'tanh')(lo)
        outputs = keras.layers.Dense(6,activation='linear',name='dynamical_quantities')(lo)
        model_RANS = keras.Model(inputs=inputs,outputs=outputs)
        # save the model architecture only once on setup
        model_RANS.save(savedir+job_name+'ep'+str(training_steps)+'_model.h5')
        model_RANS.summary()

    # optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1E-4)


@tf.function
def compute_loss(x,y):
    y_pred = model_RANS(x,training=True)
    data_loss = tf.reduce_sum(tf.reduce_mean(tf.square(y_pred[:,0:5]-y),axis=0),axis=0) 
    total_loss = data_loss 
    return total_loss

# define the training functions
@tf.function
def train_step(dataset_inputs):
    x,y = dataset_inputs
    with tf.GradientTape() as tape:
        total_loss = compute_loss(x,y)

    grads = tape.gradient(total_loss,model_RANS.trainable_weights)
    optimizer.apply_gradients(zip(grads,model_RANS.trainable_weights))
    return total_loss

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)



for epoch in range(EPOCHS):
    print(epoch)
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches
    print(num_batches)
    print(train_loss)



