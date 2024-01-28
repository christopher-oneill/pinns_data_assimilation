


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

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = 1.0*o_test_grid
    o_test_grid_temp[:,:,5]=1.0*p_grid
    o_test_grid_temp[cylinder_mask,:] = np.NaN

    pred_test = model_RANS(i_test[:],training=False)
    
    pred_test_grid = 1.0*np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask,:] = np.NaN

    plot.close('all')

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
        if supersample_factor>1:
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

def plot_large():
    global i_test_large
    global X_grid_large
    global Y_grid_large
    global training_steps
    global model_RANS
    global ScalingParameters
    plot_save_exts = ['_ux_large.png','_uy_large.png','_uxux_large.png','_uxuy_large.png','_uyuy_large.png','_p_large.png']
    cylinder_mask_large = (np.power(X_grid_large,2.0)+np.power(Y_grid_large,2.0))<=np.power(d/2.0,2.0)
    pred_test_large = model_RANS.predict(i_test_large,batch_size=1000)
    pred_test_large_grid = 1.0*np.reshape(pred_test_large,[X_grid_large.shape[0],X_grid_large.shape[1],6])
    pred_test_large_grid[cylinder_mask_large,:] = np.NaN

    for i in range(6):
        plot.figure(1)
        plot.contourf(X_grid_large,Y_grid_large,pred_test_large_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts[i],dpi=300)
        plot.close(1)

def plot_NS_large():
    global i_test_large
    global X_grid_large
    global Y_grid_large
    global training_steps
    global model_RANS
    global ScalingParameters
    cylinder_mask_large = (np.power(X_grid_large,2.0)+np.power(Y_grid_large,2.0))<=np.power(d/2.0,2.0)
    mx_large,my_large,mass_large = batch_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,i_test_large,1000)
    mx_large = 1.0*np.reshape(mx_large,X_grid_large.shape)
    mx_large[cylinder_mask_large] = np.NaN
    my_large = 1.0*np.reshape(my_large,X_grid_large.shape)
    my_large[cylinder_mask_large] = np.NaN
    mass_large = 1.0*np.reshape(mass_large,X_grid_large.shape)
    mass_large[cylinder_mask_large] = np.NaN


    mx_min = np.nanpercentile(mx_large.ravel(),1)
    mx_max = np.nanpercentile(mx_large.ravel(),99)
    mx_level = np.max([abs(mx_min),abs(mx_max)])
    mx_levels = np.linspace(-mx_level,mx_level,21)
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,mx_large,levels=mx_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_mx_large.png',dpi=300)
    plot.close(1)
    my_min = np.nanpercentile(my_large.ravel(),1)
    my_max = np.nanpercentile(my_large.ravel(),99)
    my_level = np.max([abs(my_min),abs(my_max)])
    my_levels = np.linspace(-my_level,my_level,21)
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,my_large,levels=my_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_my_large.png',dpi=300)
    plot.close(1)
    mass_min = np.nanpercentile(mass_large.ravel(),1)
    mass_max = np.nanpercentile(mass_large.ravel(),99)
    mass_level = np.max([abs(mass_min),abs(mass_max)])
    mass_levels = np.linspace(-mass_level,mass_level,21)
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,mass_large,levels=mass_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_mass_large.png',dpi=300)
    plot.close(1)



# plotting functions
# save and load functions

def save_model():
    global i_test
    global savedir
    global ScalingParameters
    global training_steps
    global model_RANS
    model_RANS.save(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_model.h5')

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
        t_mx,t_my,t_mass = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,tf.cast(i_test,tf_dtype))
        h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()

def load_custom():
    checkpoint_filename,training_steps = find_highest_numbered_file(savedir+job_name+'_ep','[0-9]*','_model.h5')
    model_RANS = keras.models.load_model(checkpoint_filename,custom_objects={'QresBlock2':QresBlock2,'QresBlock':QresBlock,})
    model_RANS.summary()
    print('Model Loaded. Epoch',str(training_steps))
    #optimizer.build(model_RANS.trainable_variables)
    return model_RANS, training_steps

# import the physics
@tf.function
def BC_RANS_no_stresses(model_RANS,BC_points):
    up = model_RANS(BC_points)
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_mean(tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))

@tf.function
def BC_RANS_reynolds_stress_pressure(model_RANS,BC_points):
    up = model_RANS(BC_points)
    # knowns
    # unknowns
    p = up[:,5]
    return tf.reduce_mean(tf.square(p))

@tf.function
def BC_RANS_inlet(model_RANS,ScalingParameters,BC_points):
    up = model_RANS(BC_points)
    ux = up[:,0]*ScalingParameters.MAX_ux
    uy = up[:,1] # no need to scale since they should go to zero
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_mean(tf.square(ux-1.0))+tf.reduce_mean(tf.square(uy))+tf.reduce_mean(tf.square(uxppuxpp))+tf.reduce_mean(tf.square(uxppuypp))+tf.reduce_mean(tf.square(uyppuypp))
 # note there is no point where the pressure is close to zero, so we neglect it in the mean field model

@tf.function
def BC_cylinder_inside(model_RANS,ScalingParameters,BC_points):
    up = model_RANS(BC_points)
    return tf.reduce_mean(tf.square(up[:,0]))+tf.reduce_mean(tf.square(up[:,1]))+tf.reduce_mean(tf.square(up[:,2]))+tf.reduce_mean(tf.square(up[:,3]))+tf.reduce_mean(tf.square(up[:,4]))

@tf.function
def BC_RANS_wall(model_RANS,ScalingParameters,BC_points):
    wall_coord = BC_points[:,0:2]
    wall_angle = BC_points[:,2]
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

    return tf.reduce_sum(tf.square(ux)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp)) #+tf.square(grad_p_norm)


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
    f_cr = tf.multiply(tf.math.less(uxux,tf.cast(0.0,tf_dtype)),tf.abs(uxux))+tf.multiply(tf.math.less(uyuy,tf.cast(0.0,tf_dtype)),tf.abs(uyuy)) # tr(ReStress)>0 by defn
    
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
        f_x, f_y, f_m = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,tf.gather(colloc_tensor,batch_inds))
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
    (BC_p,BC_wall,BC_cylinder_inside_pts) = boundary_tuple


    BC_pressure_loss = BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) 
    BC_wall_loss = BC_RANS_wall(model_RANS,ScalingParameters,BC_wall)
    BC_cylinder_inside_loss = BC_cylinder_inside(model_RANS,ScalingParameters,BC_cylinder_inside_pts)
                       
    boundary_loss = (BC_wall_loss + BC_pressure_loss + BC_cylinder_inside_loss) #  + BC_wall2 +   + BC_cylinder_inside_loss + BC_inlet_loss2
    return boundary_loss


# collloc and boundary functions

# define the collocation points

def colloc_points_function(a,b):
    colloc_limits2 = np.array([[4.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(a)

    colloc_limits3 = np.array([[-2.0,4.0],[-2.0,2.0]])
    colloc_sample_lhs3 = LHS(xlimits=colloc_limits3)
    colloc_lhs3 = colloc_sample_lhs3(b)

    colloc_merged = np.vstack((colloc_lhs2,colloc_lhs3))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print('points inside_cylinder: ',np.sum(cylinder_inds))
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])
    np.random.shuffle(f_colloc_train)
    return tf.cast(f_colloc_train,tf_dtype)


def boundary_points_function(cyl,inside):
    # define boundary condition points
    theta = np.linspace(0,2*np.pi,cyl)
    ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
    ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
    cyl_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1),theta.reshape(-1,1)))

    p_BC_x = np.array([10.0,10.0,0.0])/ScalingParameters.MAX_x
    p_BC_y = np.array([-2.0,2.0,0.0])/ScalingParameters.MAX_y
    p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

    # random points inside the cylinder
    cylinder_inside_limits1 = np.array([[-0.5,0.5],[-0.5,0.5]])
    cylinder_inside_LHS = LHS(xlimits=cylinder_inside_limits1)
    cylinder_inside_vec = cylinder_inside_LHS(inside)
    c1_loc = np.array([0.0,0.0])
    outside_cylinder_inds = np.greater_equal(np.power(np.power(cylinder_inside_vec[:,0]-c1_loc[0],2)+np.power(cylinder_inside_vec[:,1]-c1_loc[1],2),0.5),0.5*d)
    cylinder_inside_vec = np.delete(cylinder_inside_vec,np.nonzero(outside_cylinder_inds[0,:]),axis=0)
    cylinder_inside_vec[:,0] = cylinder_inside_vec[:,0]/ScalingParameters.MAX_x
    cylinder_inside_vec[:,1] = cylinder_inside_vec[:,1]/ScalingParameters.MAX_y

    # random points outside the domain for no reynolds stress condition
    boundary_tuple = (tf.cast(p_BC_vec,tf_dtype),tf.cast(cyl_BC_vec,tf_dtype),tf.cast(cylinder_inside_vec,tf_dtype))
    return boundary_tuple

# training functions

@tf.function
def compute_loss(x,y,colloc_x,boundary_tuple,ScalingParameters):
    y_pred = model_RANS(x,training=True)
    data_loss = ScalingParameters.data_loss_coefficient*tf.reduce_sum(tf.reduce_mean(tf.square(y_pred[:,0:5]-y),axis=0),axis=0) 
    physics_loss = ScalingParameters.physics_loss_coefficient*RANS_physics_loss(model_RANS,ScalingParameters,colloc_x) #tf.cast(0.0,tf_dtype)#
    boundary_loss = ScalingParameters.boundary_loss_coefficient*RANS_boundary_loss(model_RANS,ScalingParameters,boundary_tuple)

    total_loss = data_loss + physics_loss + boundary_loss
    return total_loss, data_loss, physics_loss, boundary_loss

# define the training functions
@tf.function
def train_step(x,y,colloc_x,boundary_tuple,ScalingParameters):
    with tf.GradientTape() as tape:
        total_loss ,data_loss, physics_loss, boundary_loss = compute_loss(x,y,colloc_x,boundary_tuple,ScalingParameters)

    grads = tape.gradient(total_loss,model_RANS.trainable_weights)
    optimizer.apply_gradients(zip(grads,model_RANS.trainable_weights))
    return total_loss, data_loss, physics_loss, boundary_loss

def fit_epoch(i_train,o_train,colloc_tuple,boundary_tuple,ScalingParameters):
    global training_steps
    batches = np.int64(np.ceil(i_train.shape[0]/(1.0*ScalingParameters.batch_size)))
    # sort colloc_points by error
    (colloc_grad,colloc_rand) = colloc_tuple
    assert(colloc_grad.shape[0]==batches*ScalingParameters.colloc_batch_size)
    assert(colloc_rand.shape[0]==batches*ScalingParameters.colloc_batch_size)
    #i_sampled,o_sampled = data_sample_by_err(i_train,o_train,i_train.shape[0])
    i_sampled = i_train
    o_sampled = o_train
    
    progbar = keras.utils.Progbar(batches)
    loss_vec = np.zeros((batches,),np.float64)
    data_vec = np.zeros((batches,),np.float64)
    physics_vec = np.zeros((batches,),np.float64)
    boundary_vec = np.zeros((batches,),np.float64)

    for batch in range(batches):
        progbar.update(batch+1)
        
        i_batch = i_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,i_train.shape[0]]),:]
        o_batch = o_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,o_train.shape[0]]),:]
        colloc_grad_batch = colloc_grad[(batch*ScalingParameters.colloc_batch_size):np.min([(batch+1)*ScalingParameters.colloc_batch_size,colloc_grad.shape[0]]),:]
        colloc_rand_batch = colloc_rand[(batch*ScalingParameters.colloc_batch_size):np.min([(batch+1)*ScalingParameters.colloc_batch_size,colloc_rand.shape[0]]),:]

        loss_value, data_loss, physics_loss, boundary_loss = train_step(tf.cast(i_batch,tf_dtype),tf.cast(o_batch,tf_dtype),tf.cast(tf.concat((i_batch,colloc_grad_batch,colloc_rand_batch),axis=0),tf_dtype),sample_boundary(ScalingParameters,boundary_tuple),ScalingParameters) #
        loss_vec[batch] = loss_value.numpy()
        data_vec[batch] = data_loss.numpy()
        physics_vec[batch] = physics_loss.numpy()
        boundary_vec[batch] = boundary_loss.numpy()

    training_steps = training_steps+1
    print('Epoch',str(training_steps),f" Loss: {np.mean(loss_vec):.6e}",f" Data loss: {np.mean(data_vec):.6e}")
    print(f" Physics loss: {np.mean(physics_vec):.6e}",f" Boundary loss: {np.mean(boundary_vec):.6e}",)



def train_LBFGS(model,x,y,colloc_x,boundary_tuple,ScalingParameters):
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
            #loss_value = loss(model(train_x, training=True), train_y) # jvs
            loss_value = compute_loss(x,y,colloc_x,boundary_tuple,ScalingParameters)[0]

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

assert len(sys.argv)==4

job_number = int(sys.argv[1])
supersample_factor = int(sys.argv[2])
job_hours = int(sys.argv[3])

global job_name 
job_name = 'mfg_boundary_dense2_{:03d}_S{:d}'.format(job_number,supersample_factor)

job_duration = timedelta(hours=job_hours,minutes=0)
end_time = start_time+job_duration

LOCAL_NODE = 'DESKTOP-AMLVDAF'
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
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    sys.path.append(HOMEDIR+'code/')



#from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import QresBlock
from pinns_data_assimilation.lib.layers import QresBlock2
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

print('MAX_x: ',MAX_x)
print('MIN_x: ',MIN_x)
print('MAX_y: ',MAX_y)
print('MAX_x: ',MIN_y)


x_test = 1.0*x/MAX_x
y_test = 1.0*y/MAX_x
global i_test
i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))

global p_grid
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]
p_grid = np.reshape(p,X_grid.shape)/MAX_p

x_large = np.linspace(-10,10,2000)
y_large = 1.0*x_large
global X_grid_large
global Y_grid_large
X_grid_large, Y_grid_large = np.meshgrid(x_large,y_large)
global i_test_large
i_test_large = np.stack((X_grid_large.ravel(),Y_grid_large.ravel()),axis=1)/MAX_x


global o_test_grid
o_test_grid = np.reshape(np.hstack((ux.reshape(-1,1)/MAX_ux,uy.reshape(-1,1)/MAX_uy,uxux.reshape(-1,1)/MAX_uxux,uxuy.reshape(-1,1)/MAX_uxuy,uyuy.reshape(-1,1)/MAX_uyuy)),[X_grid.shape[0],X_grid.shape[1],5])


# copy inlet data
inds_inlet_data = np.concatenate((np.nonzero(x==MIN_x)[0],np.nonzero(y==MAX_y)[0],np.nonzero(y==MIN_y)[0]),axis=0)

inlet_x = x[inds_inlet_data]
inlet_x = x[inds_inlet_data]
inlet_y = y[inds_inlet_data]
inlet_ux = ux[inds_inlet_data]
inlet_uy = uy[inds_inlet_data]
inlet_uxux = uxux[inds_inlet_data]
inlet_uxuy = uxuy[inds_inlet_data]
inlet_uyuy = uyuy[inds_inlet_data]

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
x = np.delete(x,cylinder_mask,axis=0)
y = np.delete(y,cylinder_mask,axis=0)
ux = np.delete(ux,cylinder_mask,axis=0)
uy = np.delete(uy,cylinder_mask,axis=0)
uxux = np.delete(uxux,cylinder_mask,axis=0)
uxuy = np.delete(uxuy,cylinder_mask,axis=0)
uyuy = np.delete(uyuy,cylinder_mask,axis=0)

# re append the inlet data
x = np.concatenate((x,inlet_x),axis=0)
y = np.concatenate((y,inlet_y),axis=0)
ux = np.concatenate((ux,inlet_ux),axis=0)
uy = np.concatenate((uy,inlet_uy),axis=0)
uxux = np.concatenate((uxux,inlet_uxux),axis=0)
uxuy = np.concatenate((uxuy,inlet_uxuy),axis=0)
uyuy = np.concatenate((uyuy,inlet_uyuy),axis=0)


# for LBFGS we don't need to duplicate since all points and collocs are evaluated in a single step
o_train_LBFGS = np.stack((ux/MAX_ux,uy/MAX_uy,uxux/MAX_uxux,uxuy/MAX_uxuy,uyuy/MAX_uyuy),axis=1)
i_train_LBFGS = np.stack((x/MAX_x,y/MAX_x),axis=1)

# copy the arrays if supersample factor is >1 so that the data size is approximately consistent
o_train_backprop = np.stack((np.concatenate([ux for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_ux,np.concatenate([uy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uy,np.concatenate([uxux for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uxux,np.concatenate([uxuy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uxuy,np.concatenate([uyuy for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_uyuy),axis=1)
i_train_backprop = np.stack((np.concatenate([x for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_x, np.concatenate([y for i in range(supersample_factor*supersample_factor)],axis=0)/MAX_x),axis=1)

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
ScalingParameters.boundary_batch_size = 32
ScalingParameters.physics_loss_coefficient = tf.cast(1.0,tf_dtype)
ScalingParameters.boundary_loss_coefficient = tf.cast(1.0,tf_dtype)
ScalingParameters.data_loss_coefficient = tf.cast(1.0,tf_dtype)



global colloc_vector
boundary_tuple = boundary_points_function(360,200)
colloc_vector = colloc_points_function(10000,50000)

global training_steps
global model_RANS
# model creation
tf_device_string ='/GPU:0'

optimizer = keras.optimizers.Adam(learning_rate=1E-4)

from pinns_data_assimilation.lib.file_util import get_filepaths_with_glob
from pinns_data_assimilation.lib.layers import QresBlock2

checkpoint_files = get_filepaths_with_glob(PROJECTDIR+'/output/'+job_name+'/',job_name+'_ep*_model.h5')

if len(checkpoint_files)>0:
    with tf.device(tf_device_string):
        model_RANS,training_steps = load_custom()
else: 
    training_steps = 0
    with tf.device(tf_device_string):        
        inputs = keras.Input(shape=(2,),name='coordinates')
        lo = keras.layers.Dense(100,activation='tanh',dtype=tf_dtype)(inputs)
        for i in range(10):
            lo = keras.layers.Dense(100,activation='tanh',dtype=tf_dtype)(lo)
        outputs = keras.layers.Dense(6,activation='linear',name='dynamical_quantities')(lo)
        model_RANS = keras.Model(inputs=inputs,outputs=outputs)
        model_RANS.summary()


LBFGS_steps = 333
LBFGS_epochs = 3*LBFGS_steps

d_ts = 100
# training

global saveFig
saveFig=True

history_list = []




if True:
    last_epoch_time = datetime.now()
    average_epoch_time=timedelta(minutes=10)
    # LBFGS
    import tensorflow_probability as tfp
    L_iter = 0
    boundary_tuple = boundary_points_function(720,200)
    colloc_vector = colloc_points_function(10000,50000) # one A100 max = 60k?
    func = train_LBFGS(model_RANS,tf.cast(i_train_LBFGS,tf_dtype),tf.cast(o_train_LBFGS,tf_dtype),colloc_vector,boundary_tuple,ScalingParameters)
    init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables)
            
    while True:
        
        last_epoch_time = datetime.now()
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,f_relative_tolerance=1E-16,stopping_condition=tfp.optimizer.converged_all)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        training_steps = training_steps + LBFGS_epochs
        L_iter = L_iter+1
            
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
 
        # check if we are out of time
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            save_pred()
            save_model()
            exit()

        if np.mod(L_iter,10)==0:
            save_model()
            boundary_tuple = boundary_points_function(720,200)
            colloc_vector = colloc_points_function(10000,50000)
            func = train_LBFGS(model_RANS,tf.cast(i_train_LBFGS,tf_dtype),tf.cast(o_train_LBFGS,tf_dtype),colloc_vector,boundary_tuple,ScalingParameters)
            init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables)

        #if node_name==LOCAL_NODE:
        #    save_pred()
        #    plot_err()
        #    plot_NS_residual()



