


import tensorflow as tf

import tensorflow.keras as keras
import h5py
#import tensorflow_probability as tfp
from smt.sampling_methods import LHS

keras.backend.set_floatx('float64')


import numpy as np

import matplotlib.pyplot as plot
import matplotlib

import time

import sys
sys.path.append('C:/projects/pinns_local/code/')


#from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import CylindricalEmbeddingLayer

from pinns_data_assimilation.lib.file_util import find_highest_numbered_file


from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center



def plot_err(epoch,model_RANS):
    global p_grid
    global X_grid
    global Y_grid
    global i_test
    global o_test_grid
    global saveFig
    global fig_dir
    global d
    global ScalingParameters

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = 1.0*o_test_grid
    o_test_grid_temp[:,:,5]=1.0*p_grid
    o_test_grid_temp[cylinder_mask,0:5] = np.NaN

    pred_test = model_RANS(i_test[:],training=False)
    
    pred_test_grid = 1.0*np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask,0:5] = np.NaN

    mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,ScalingParameters,i_test[:])
    mx = 1.0*np.reshape(mx,X_grid.shape)
    mx[cylinder_mask] = np.NaN
    my = 1.0*np.reshape(my,X_grid.shape)
    my[cylinder_mask] = np.NaN
    mass = 1.0*np.reshape(mass,X_grid.shape)
    mass[cylinder_mask] = np.NaN


    plot.close('all')

    err_test = o_test_grid_temp-pred_test_grid

    # NS residual

    plot.figure(epoch)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,mx,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,my,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,mass,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(epoch)+'_NS_residual.png',dpi=300)


    plot_save_exts = ['_ux.png','_uy.png','_uxux.png','_uxuy.png','_uyuy.png','_p.png']

    # quantities
    for i in range(6):

        o_test_max = np.nanmax(np.abs(o_test_grid_temp[:,:,i].ravel()))
        o_test_levels = np.linspace(-o_test_max,o_test_max,21)

        plot.figure(epoch)
        plot.title('Full Resolution')
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,i],levels=o_test_levels)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,i],levels=21,vmin=-o_test_max,vmax=o_test_max)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        plot.contourf(X_grid,Y_grid,err_test[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        if saveFig:
            plot.savefig(fig_dir+'ep'+str(epoch)+plot_save_exts[i],dpi=300)



def save_custom():
    global i_test
    global savedir
    global ScalingParameters
    global training_steps
    global model_RANS
    model_RANS.save(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_model.h5')
    pred = model_RANS(i_test,training=False)
    h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
    if ScalingParameters.physics_loss_coefficient!=0:
        t_mx,t_my,t_mass = RANS_reynolds_stress_cartesian2(model_RANS,ScalingParameters,i_test)
        h5f = h5py.File(savedir+job_name+'_ep'+str(np.uint(training_steps))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()

def load_custom():
    checkpoint_filename,training_steps = find_highest_numbered_file(savedir+job_name+'_ep','[0-9]*','_model.h5')
    model_RANS = keras.models.load_model(checkpoint_filename,custom_objects={'ResidualLayer':ResidualLayer,'CylindricalEmbeddingLayer':CylindricalEmbeddingLayer})
    model_RANS.summary()
    print('Model Loaded. Epoch',str(training_steps))
    optimizer.build(model_RANS.trainable_variables)
    return model_RANS, training_steps

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists





HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
global savedir
global job_name 
job_name= 'mfg_res_mean10c_003'
savedir = HOMEDIR+'output/'+job_name+'/'
create_directory_if_not_exists(savedir)
global fig_dir
fig_dir = savedir + 'figures/'
create_directory_if_not_exists(fig_dir)

reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')



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




# create the test data
global o_test_grid
o_test_grid = np.reshape(np.hstack((ux.reshape(-1,1)/MAX_ux,uy.reshape(-1,1)/MAX_uy,uxux.reshape(-1,1)/MAX_uxux,uxuy.reshape(-1,1)/MAX_uxuy,uyuy.reshape(-1,1)/MAX_uyuy)),[X_grid.shape[0],X_grid.shape[1],5])
ux_grid = np.reshape(ux,X_grid.shape)


MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)
x_test = 1.0*x/MAX_x
y_test = 1.0*y/MAX_x
global i_test
i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))

supersample_factor=1
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

o_train = np.hstack((ux.reshape(-1,1)/MAX_ux,uy.reshape(-1,1)/MAX_uy,uxux.reshape(-1,1)/MAX_uxux,uxuy.reshape(-1,1)/MAX_uxuy,uyuy.reshape(-1,1)/MAX_uyuy))
i_train = np.hstack((x.reshape(-1,1)/MAX_x,y.reshape(-1,1)/MAX_x))

global p_grid
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]
p_grid = np.reshape(p,X_grid.shape)/MAX_p



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
ScalingParameters.nu_mol = 0.0066667
ScalingParameters.MAX_p= MAX_p # estimated maximum pressure, we should
ScalingParameters.batch_size = 32
ScalingParameters.colloc_batch_size = 10000
ScalingParameters.physics_loss_coefficient = np.float64(0.0)
ScalingParameters.boundary_loss_coefficient = np.float64(0.0)
ScalingParameters.data_loss_coefficient = np.float64(1.0)
ScalingParameters.pressure_loss_coefficient=np.float64(0.0)

# define boundary condition points
theta = np.linspace(0,2*np.pi,360)
ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
cyl_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1),theta.reshape(-1,1)))

p_BC_x = np.array([10.0,10.0])/ScalingParameters.MAX_x
p_BC_y = np.array([-2.0,2.0])/ScalingParameters.MAX_y
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

inlet_BC_x = -4.0*np.ones([100,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,100)/ScalingParameters.MAX_y
inlet_BC_vec = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

inlet_BC_x = -2.0*np.ones([100,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,100)/ScalingParameters.MAX_y
inlet_BC_vec2 = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

cylinder_inside_limits1 = np.array([[-0.5,0.5],[-0.5,0.5]])
cylinder_inside_LHS = LHS(xlimits=cylinder_inside_limits1)
cylinder_inside_vec = cylinder_inside_LHS(1000)
c1_loc = np.array([0.0,0.0])
outside_cylinder_inds = np.greater(np.power(np.power(cylinder_inside_vec[:,0]-c1_loc[0],2)+np.power(cylinder_inside_vec[:,1]-c1_loc[1],2),0.5),0.5*d)
cylinder_inside_vec = np.delete(cylinder_inside_vec,outside_cylinder_inds[0,:],axis=0)
cylinder_inside_vec[:,0] = cylinder_inside_vec[:,0]/ScalingParameters.MAX_x
cylinder_inside_vec[:,1] = cylinder_inside_vec[:,1]/ScalingParameters.MAX_y

boundary_tuple = (p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2,cylinder_inside_vec)

# define the collocation points

def colloc_points_function():
    colloc_limits1 = np.array([[-2.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
    colloc_lhs1 = colloc_sample_lhs1(100000)


    colloc_limits2 = np.array([[-1,2],[-1,1]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(25000)

    colloc_merged = np.vstack((colloc_lhs1,colloc_lhs2))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print('points inside_cylinder: ',np.sum(cylinder_inds))
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])
    np.random.shuffle(f_colloc_train)
    return f_colloc_train

global colloc_vector
colloc_vector = colloc_points_function()


# import the physics
@tf.function
def BC_RANS_inlet2(model_RANS,BC_points):
    up = model_RANS(BC_points)
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_mean(tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))
 # note there is no point where the pressure is close to zero, so we neglect it in the mean field model



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
    return tf.reduce_mean(tf.square(ux-1.0)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))
 # note there is no point where the pressure is close to zero, so we neglect it in the mean field model

@tf.function
def BC_cylinder_inside(model_RANS,ScalingParameters,BC_points):
    up = model_RANS(BC_points)
    return tf.reduce_mean(tf.square(up[:,0:5]))

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
    p = up[:,5]
    (dp,) = tf.gradients((p,),(wall_coord)) 
    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y
    grad_p_norm = p_x*tf.cos(wall_angle)+p_y*tf.sin(wall_angle)

    return tf.reduce_sum(tf.square(ux)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp)+tf.square(grad_p_norm))


@tf.function
def RANS_reynolds_stress_cartesian2(model_RANS,ScalingParameters,colloc_tensor):
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
    (dux,duy,duxux,duxuy,duyuy,dp) = tf.gradients((ux,uy,uxux,uxuy,uyuy,p), (colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    
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

    (dux_x,dux_y,duy_x,duy_y) = tf.gradients((ux_x,ux_y,uy_x,uy_y),(colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    
    return f_x, f_y, f_mass

@tf.function
def value_and_grad_magnitude(model_RANS,x):
    y_pred = model_RANS(x,training=True)
    y_pred_shape = tf.shape(y_pred)
    (dux,duy,duxux,duxuy,duyuy,dp)= tf.gradients((y_pred[:,0],y_pred[:,1],y_pred[:,2],y_pred[:,3],y_pred[:,4],y_pred[:,5]),(x,x,x,x,x,x))
    y_grads = tf.stack((dux,duy,duxux,duxuy,duyuy,dp),axis=1)
    y_weights = tf.sqrt(tf.square(y_grads[:,:,0])+tf.square(y_grads[:,:,1]))+1

    return y_pred, y_weights
# sampling related functions

@tf.function
def RANS_error_gradient2(model_RANS,ScalingParameters,colloc_tensor):
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
    (dux,duy,duxux,duxuy,duyuy,dp) = tf.gradients((ux,uy,uxux,uxuy,uyuy,p), (colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    
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

    (dux_x,dux_y,duy_x,duy_y) = tf.gradients((ux_x,ux_y,uy_x,uy_y),(colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_m = ux_x + uy_y

    (df_x,df_y,df_m) = tf.gradients((f_x,f_y,f_m),(colloc_tensor,colloc_tensor,colloc_tensor))

    
    f_x_x = df_x[:,0]/ScalingParameters.MAX_x
    f_x_y = df_x[:,1]/ScalingParameters.MAX_y
    f_y_x = df_y[:,0]/ScalingParameters.MAX_x
    f_y_y = df_y[:,1]/ScalingParameters.MAX_y
    f_m_x = df_m[:,0]/ScalingParameters.MAX_x
    f_m_y = df_m[:,1]/ScalingParameters.MAX_y


    return f_x, f_y, f_m, f_x_x, f_x_y, f_y_x, f_y_y, f_m_x, f_m_y


def RANS_sort_by_grad_err(model_RANS,colloc_points,ScalingParameters,batch_size=1000):
    n_batch = np.int64(np.ceil(colloc_points.shape[0]/(1.0*batch_size)))
    f_x_x_list = []
    f_x_y_list = []
    f_y_x_list = []
    f_y_y_list = []
    f_m_x_list = []
    f_m_y_list = []
    progbar = keras.utils.Progbar(n_batch)
    for batch in range(0,n_batch):
        progbar.update(batch+1)
        batch_inds = np.arange(batch*batch_size,np.min([(batch+1)*batch_size,colloc_points.shape[0]]))
        f_x, f_y, f_m, f_x_x, f_x_y, f_y_x, f_y_y, f_m_x, f_m_y = RANS_error_gradient2(model_RANS,ScalingParameters,tf.gather(colloc_points,batch_inds))
        f_x_x_list.append(f_x_x)
        f_x_y_list.append(f_x_y)
        f_y_x_list.append(f_y_x)
        f_y_y_list.append(f_y_y)
        f_m_x_list.append(f_m_x)
        f_m_y_list.append(f_m_y)
    
    # combine the batches together
    f_x_x = tf.concat(f_x_x_list,axis=0)
    f_x_y = tf.concat(f_x_y_list,axis=0)
    f_y_x = tf.concat(f_y_x_list,axis=0)
    f_y_y = tf.concat(f_y_y_list,axis=0)
    f_m_x = tf.concat(f_m_x_list,axis=0)
    f_m_y = tf.concat(f_m_y_list,axis=0)

    abs_err = np.abs(f_x_x)+np.abs(f_x_y) + np.abs(f_y_x) + np.abs(f_y_y)+ np.abs(f_m_x) + np.abs(f_m_y)
    sort_inds = np.argsort(abs_err,axis=0)
    colloc_points_sorted = colloc_points[sort_inds[::-1],:]

    return colloc_points_sorted


def RANS_sample_by_grad_err(model_RANS,colloc_points,ScalingParameters,size):
    colloc_points_sorted = RANS_sort_by_grad_err(model_RANS,colloc_points,ScalingParameters)
    rand_colloc_points = np.abs(np.random.randn(size,)/3.0)
    rand_colloc_points[rand_colloc_points>1.0]=1.0
    rand_colloc_points = np.int64((colloc_points.shape[0]-1)*rand_colloc_points)
    colloc_points_sampled = colloc_points_sorted[rand_colloc_points,:]

    return colloc_points_sampled

def data_sample_by_err(i_train,o_train,size):
    global model_RANS
    # evaluate the error
    pred = model_RANS(i_train,training=False)
    err = np.sum(np.power(pred[:,0:5] - o_train,2.0),axis=1)
    err_order = np.argsort(err)
    # sort
    i_train_sorted = i_train[err_order[::-1],:]
    o_train_sorted = o_train[err_order[::-1],:]
    # sample based on the error
    rand_points = np.abs(np.random.randn(size,)/3.0)
    rand_points[rand_points>1.0]=1.0
    rand_points = np.int64((i_train.shape[0]-1)*rand_points)
    i_train_sampled = i_train_sorted[rand_points,:]
    o_train_sampled = o_train_sorted[rand_points,:]

    return i_train_sampled, o_train_sampled

# define the model

@tf.function
def RANS_physics_loss(model_RANS,ScalingParameters,colloc_points,): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,ScalingParameters,colloc_points)
    physical_loss1 = tf.reduce_mean(tf.square(mx))
    physical_loss2 = tf.reduce_mean(tf.square(my))
    physical_loss3 = tf.reduce_mean(tf.square(mass))
    physics_loss = physical_loss1 + physical_loss2 + physical_loss3
    return physics_loss

@tf.function
def RANS_boundary_loss(model_RANS,ScalingParameters,boundary_tuple):
    (BC_p,BC_wall,BC_inlet,BC_inlet2,BC_cylinder_inside_pts) = boundary_tuple
    # outlet pressure
    BC_pressure_loss = BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) # scaled to compensate the reduce sum on other BCs
    # wall
    #BC_wall_loss = 0.0*BC_RANS_wall(model_RANS,ScalingParameters,BC_wall)
    #mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,ScalingParameters,BC_wall[:,0:2])
    #BC_wall2 = tf.reduce_mean(tf.square(mx))+tf.reduce_mean(tf.square(my))+tf.reduce_mean(tf.square(mass))
    BC_cylinder_inside_loss = BC_cylinder_inside(model_RANS,ScalingParameters,BC_cylinder_inside_pts)
    
    #inlets
    #BC_inlet_loss = BC_RANS_inlet(model_RANS,ScalingParameters,BC_inlet)
    BC_inlet_loss2 = BC_RANS_inlet2(model_RANS,BC_inlet2)
                       
    boundary_loss = (BC_pressure_loss + BC_cylinder_inside_loss+  BC_inlet_loss2) # BC_wall_loss + BC_wall2 +  + BC_inlet_loss+
    return boundary_loss

global training_steps
global model_RANS
# model creation
tf_device_string ='/CPU:0'

optimizer = keras.optimizers.Adam(learning_rate=1E-4)


if True:
    training_steps = 0
    nodes = np.concatenate((np.linspace(10,100,10),np.array([100,100,100])))
    with tf.device(tf_device_string):        
        inputs = keras.Input(shape=(2,),name='coordinates')
        lo = keras.layers.Dense(nodes[0],activation='linear',name='dense1')(inputs)
        for i in range(1,nodes.shape[0]):
            lo = ResidualLayer(int(nodes[i]),activation='tanh',name='res'+str(i))(lo)
        outputs = keras.layers.Dense(6,activation='linear',name='dynamical_quantities')(lo)
        model_RANS = keras.Model(inputs=inputs,outputs=outputs)
        model_RANS.summary()
else:
    with tf.device(tf_device_string):
        model_RANS,training_steps = load_custom()



# define the training functions
@tf.function
def train_step(x,y,colloc_x,boundary_tuple,ScalingParameters):
    with tf.GradientTape() as tape:
        #y_pred,y_weights  = value_and_grad_magnitude(model_RANS,x)
        y_pred = model_RANS(x,training=True)
        data_err = tf.reduce_mean(tf.square(y_pred[:,0:5]-y),axis=0)
        dynamic_data_weight = tf.exp(tf.math.floor(tf.math.log(tf.reduce_max(data_err)/data_err+1E-30)))
        data_loss = tf.reduce_sum(tf.multiply(dynamic_data_weight,data_err),axis=0) 


        physics_loss = ScalingParameters.physics_loss_coefficient*RANS_physics_loss(model_RANS,ScalingParameters,colloc_x)
        boundary_loss = ScalingParameters.boundary_loss_coefficient*RANS_boundary_loss(model_RANS,ScalingParameters,boundary_tuple)

        combined_physics_loss = physics_loss + boundary_loss
        dynamic_physics_weight = tf.math.exp(tf.math.ceil(tf.math.log(combined_physics_loss+1E-30)))
        total_loss = tf.math.reduce_max((dynamic_physics_weight,tf.cast(1.0,tf.float64)))*ScalingParameters.data_loss_coefficient*data_loss + combined_physics_loss

    grads = tape.gradient(total_loss,model_RANS.trainable_weights)
    optimizer.apply_gradients(zip(grads,model_RANS.trainable_weights))
    return total_loss


def fit_epoch(i_train,o_train,colloc_points,boundary_tuple,ScalingParameters):
    global training_steps
    batches = np.int64(np.ceil(i_train.shape[0]/(1.0*ScalingParameters.batch_size)))
    # sort colloc_points by error
    i_sampled = i_train
    o_sampled = o_train
    
    progbar = keras.utils.Progbar(batches)
    loss_vec = np.zeros((batches,),np.float64)
    for batch in range(batches):
        progbar.update(batch+1)
        colloc_inds = np.random.randint(0,colloc_points.shape[0],ScalingParameters.colloc_batch_size)
        i_batch = i_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,i_train.shape[0]]),:]
        o_batch = o_sampled[(batch*ScalingParameters.batch_size):np.min([(batch+1)*ScalingParameters.batch_size,o_train.shape[0]]),:]
        colloc_batch = colloc_points[colloc_inds,:]
        loss_value = train_step(tf.cast(i_batch,tf.float64),tf.cast(o_batch,tf.float64),tf.cast(colloc_batch,tf.float64),boundary_tuple,ScalingParameters)
        loss_vec[batch] = loss_value.numpy()

    training_steps = training_steps+1
    if (np.mod(training_steps,10)==0):
        save_custom()
    plot_err(training_steps,model_RANS)
    print('Epoch',str(training_steps),f" Loss Value: {np.mean(loss_vec):.6e}")



LBFGS_steps = 333
LBFGS_epochs = 3*LBFGS_steps


d_ts = 100
# training

global saveFig
saveFig=True

ScalingParameters.data_loss_coefficient=1.0
ScalingParameters.colloc_batch_size=256
history_list = []
if True:
    # new style training with probabalistic training dataset
    lr_schedule = np.array([    3.3E-5,  1E-5,   3.3E-6,   1E-6,        1E-5,   3.3E-6,   1E-6,         1E-5,   3.3E-6, 1E-6, ])
    ep_schedule = np.array([    0,       30,     150,      450,         600,    660,      960,          1200,    1290,   1500,])
    phys_schedule = np.array([  1E-30,   1E-30,  1E-30,    1E-30,       1E-7,   1E-7,     1E-7,         1E-5,   1E-5,   1E-5,   ])
    boundary_schedule = np.array([0.01,  0.01,   0.01,     0.01,        0.03,   0.03,     0.03,         0.1,    0.1,    0.1,    ])

    # reset the correct learing rate on load
    i_temp = 0
    for i in range(len(ep_schedule)):
        if training_steps>=ep_schedule[i]:
            i_temp = i
    keras.backend.set_value(optimizer.learning_rate, lr_schedule[i_temp])
    ScalingParameters.physics_loss_coefficient = tf.cast(phys_schedule[i_temp],tf.float64)
    ScalingParameters.boundary_loss_coefficient = tf.cast(boundary_schedule[i_temp],tf.float64)
    print('physics loss =',str(ScalingParameters.physics_loss_coefficient))
    print('learning rate =',str(lr_schedule[i_temp]))

    while True:
        for i in range(1,len(ep_schedule)):
            if training_steps==ep_schedule[i]:
                keras.backend.set_value(optimizer.learning_rate, lr_schedule[i])
                print('epoch',str(training_steps))
                ScalingParameters.physics_loss_coefficient = tf.cast(phys_schedule[i],tf.float64)
                ScalingParameters.boundary_loss_coefficient = tf.cast(boundary_schedule[i],tf.float64)
                print('physics loss =',str(ScalingParameters.physics_loss_coefficient))
                print('learning rate =',str(lr_schedule[i]))               
        colloc_vector = colloc_points_function()
        fit_epoch(i_train,o_train,colloc_vector,boundary_tuple,ScalingParameters)

        


