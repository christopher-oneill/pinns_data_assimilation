


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

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<np.power(d/2.0,2.0)

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = o_test_grid
    o_test_grid_temp[:,:,5]=p_grid
    o_test_grid_temp[cylinder_mask] = np.NaN

    pred_test = model_RANS.predict(i_test[:],batch_size=1000)
    
    pred_test_grid = np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask] = np.NaN

    mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,i_test[:])
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



def save_custom(model,epochs):
    global i_test
    global savedir
    model.save(savedir+'mfg_res_mean3_ep'+str(np.uint(epochs))+'_model.h5')
    pred = model.predict(i_test,batch_size=1024)
    h5f = h5py.File(savedir+'mfg_res_mean3_ep'+str(np.uint(epochs))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
    if model.ScalingParameters.physics_loss_coefficient!=0:
        t_mx,t_my,t_mass = RANS_reynolds_stress_cartesian2(model,i_test)
        h5f = h5py.File(savedir+'mfg_res_mean3_ep'+str(np.uint(epochs))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()


def load_custom(epochs,lr):
    global savedir
    model_RANS = keras.models.load_model(savedir+'mfg_res_mean3_ep'+str(epochs)+'_model.h5',custom_objects={'mean_loss':RANS_reynolds_stress_loss_wrapper(None,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),'ResidualLayer':ResidualLayer,'FourierEmbeddingLayer':FourierEmbeddingLayer})
    # we need to compile again after loading once we can populate the loss function with the model object
    model_RANS.ScalingParameters = ScalingParameters
    model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)
    

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists





HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
global savedir
savedir = HOMEDIR+'output/mfg_res_mean3_004/'
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
x_test = x
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
y_test = y
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

# set points inside the cylinder to zero

cylinder_mask = np.reshape(np.power(x,2.0)+np.power(y,2.0)<np.power(d/2.0,2.0),[x.shape[0],])

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
x_test = X_grid/MAX_x
y_test = Y_grid/MAX_x
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
ScalingParameters.colloc_batch_size = 32
ScalingParameters.physics_loss_coefficient = np.float64(0.0)
ScalingParameters.boundary_loss_coefficient = np.float64(0.0)
ScalingParameters.data_loss_coefficient = np.float64(1.0)
ScalingParameters.pressure_loss_coefficient=np.float64(0.0)

# define boundary condition points
theta = np.linspace(0,2*np.pi,360)
ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
cyl_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1)))

p_BC_x = np.array([10.0,10.0])/ScalingParameters.MAX_x
p_BC_y = np.array([-2.0,2.0])/ScalingParameters.MAX_y
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

inlet_BC_x = -10.0*np.ones([100,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,100)/ScalingParameters.MAX_y
inlet_BC_vec = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

inlet_BC_x = -2.0*np.ones([100,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,100)/ScalingParameters.MAX_y
inlet_BC_vec2 = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))


# define the collocation points

def colloc_points_function():
    
    data_points = np.hstack((np.reshape(x,[x.size,1]),np.reshape(y,[y.size,1])))

    colloc_limits1 = np.array([[-10.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
    colloc_lhs1 = colloc_sample_lhs1(78000-x.size+20000)


    colloc_limits2 = np.array([[-1.5,1.5],[-1.5,1.5]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(20000)

    colloc_merged = np.vstack((data_points,colloc_lhs1,colloc_lhs2))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print(cylinder_inds.shape)
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])
    return f_colloc_train

global colloc_vector
colloc_vector = colloc_points_function()
# shuffle the colloc vector
np.random.shuffle(colloc_vector)



# import the physics
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_pressure
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_no_slip
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet2

@tf.function
def RANS_reynolds_stress_cartesian2(model_RANS,colloc_tensor):
    # in this version we try to trace the graph only twice (first gradient, second gradient)
    up = model_RANS(colloc_tensor)
    # knowns
    ux = up[:,0]*model_RANS.ScalingParameters.MAX_ux
    uy = up[:,1]*model_RANS.ScalingParameters.MAX_uy
    uxux = up[:,2]*model_RANS.ScalingParameters.MAX_uxppuxpp
    uxuy = up[:,3]*model_RANS.ScalingParameters.MAX_uxppuypp
    uyuy = up[:,4]*model_RANS.ScalingParameters.MAX_uyppuypp
    # unknowns
    p = up[:,5]*model_RANS.ScalingParameters.MAX_p
    
    # compute the gradients of the quantities
    
    # first gradients
    (dux,duy,duxux,duxuy,duyuy,dp) = tf.gradients((ux,uy,uxux,uxuy,uyuy,p), (colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    
    # ux grads
    ux_x = dux[:,0]/model_RANS.ScalingParameters.MAX_x
    ux_y = dux[:,1]/model_RANS.ScalingParameters.MAX_y
    
    # uy gradient
    uy_x = duy[:,0]/model_RANS.ScalingParameters.MAX_x
    uy_y = duy[:,1]/model_RANS.ScalingParameters.MAX_y


    # gradient unmodeled reynolds stresses
    uxux_x = duxux[:,0]/model_RANS.ScalingParameters.MAX_x
    uxuy_x = duxuy[:,0]/model_RANS.ScalingParameters.MAX_x
    uxuy_y = duxuy[:,1]/model_RANS.ScalingParameters.MAX_y
    uyuy_y = duyuy[:,1]/model_RANS.ScalingParameters.MAX_y

    # pressure gradients
    p_x = dp[:,0]/model_RANS.ScalingParameters.MAX_x
    p_y = dp[:,1]/model_RANS.ScalingParameters.MAX_y

    (dux_x,dux_y,duy_x,duy_y) = tf.gradients((ux_x,ux_y,uy_x,uy_y),(colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    # and second derivative
    ux_xx = dux_x[:,0]/model_RANS.ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/model_RANS.ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/model_RANS.ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/model_RANS.ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (model_RANS.ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (model_RANS.ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    
    return f_x, f_y, f_mass



def RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_points,BC_p,BC_ns,BC_inlet,BC_inlet2): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0])#+tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,0])-1,0)) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1])#+tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,1])-1,0)) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2])#+tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,2])-1,0)) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3])#+tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,3])-1,0)) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4])#+tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,4])-1,0)) # v''v''
        scaling_loss_p = (model_RANS.ScalingParameters.physics_loss_coefficient==0.0)*tf.reduce_mean(tf.square(y_pred[:,5]))#tf.reduce_mean(tf.math.maximum(tf.square(y_pred[:,5])-1,0))
        # we add the pressure scaling loss as well because it helps the numerical condition of the model
        data_loss = tf.reduce_mean(data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp+ scaling_loss_p)
        
        if (model_RANS.ScalingParameters.physics_loss_coefficient==0):
            physics_loss = 0.0
        else:
            if (model_RANS.ScalingParameters.batch_size==colloc_points.shape[0]):
                # all colloc points
                mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,colloc_points)
            else:
                # uniform random selection of collocation points with batch size
                # rand_colloc_points = np.random.choice(colloc_points.shape[0],model_RANS.ScalingParameters.batch_size)
                
                # gaussian random sampling based on the gradient error, we just need to have pre-sorted the collocation points based on the error
                #rand_colloc_points = np.abs(np.random.randn(model_RANS.ScalingParameters.batch_size,)/3.0)
                #rand_colloc_points[rand_colloc_points>1.0]=1.0
                #rand_colloc_points = np.int64((colloc_points.shape[0]-1)*rand_colloc_points)

                ## random batch indexing
                batch_start = np.random.randint(0,colloc_points.shape[0]-model_RANS.ScalingParameters.colloc_batch_size-1,)
                mx,my,mass = RANS_reynolds_stress_cartesian2(model_RANS,colloc_points[batch_start:(batch_start+model_RANS.ScalingParameters.colloc_batch_size-1),:])
            physical_loss1 = tf.reduce_mean(tf.square(mx))
            physical_loss2 = tf.reduce_mean(tf.square(my))
            physical_loss3 = tf.reduce_mean(tf.square(mass))
            physics_loss = physical_loss1 + physical_loss2 + physical_loss3
        if (model_RANS.ScalingParameters.boundary_loss_coefficient==0):
            boundary_loss = 0.0
        else:
            BC_pressure_loss = 500.0*BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) # scaled to compensate the reduce sum on other BCs
            BC_no_slip_loss = BC_RANS_reynolds_stress_no_slip(model_RANS,BC_ns)
            BC_inlet_loss = BC_RANS_inlet(model_RANS,BC_inlet)
            BC_inlet_loss2 = BC_RANS_inlet2(model_RANS,BC_inlet2)
                       
            boundary_loss = (BC_pressure_loss + BC_no_slip_loss + BC_inlet_loss+BC_inlet_loss2)

        combined_phyisics_loss = model_RANS.ScalingParameters.physics_loss_coefficient*physics_loss + model_RANS.ScalingParameters.boundary_loss_coefficient*boundary_loss
        dynamic_data_weight = tf.math.exp(tf.math.ceil(tf.math.log(combined_phyisics_loss+1E-30)))
        return  tf.math.reduce_max((dynamic_data_weight,tf.cast(1.0,tf.float64)))*model_RANS.ScalingParameters.data_loss_coefficient*data_loss + combined_phyisics_loss
        
    return mean_loss

def data_sample_by_err(i_train,o_train,size):
    global model_RANS
    # evaluate the error
    pred = model_RANS.predict(i_train,batch_size=1024)
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


global training_steps
global model_RANS
# model creation
tf_device_string ='/CPU:0'

frequency_vector = tf.constant(np.linspace(0,30,61,True))

if True:
    training_steps = 0
    nodes = 100
    with tf.device(tf_device_string):
        
        model_RANS = keras.Sequential()
        model_RANS.add(keras.layers.Dense(nodes,activation='linear',input_shape=(2,)))
        for k in range(20):
            model_RANS.add(ResidualLayer(nodes,activation='swish'))
        model_RANS.add(keras.layers.Dense(6,activation='linear'))
        model_RANS.summary()
        model_RANS.ScalingParameters = ScalingParameters
        model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2)) #,jit_compile=False 
else:
    with tf.device(tf_device_string):
        model_RANS = keras.models.load_model(savedir+'mfg_res_mean3_ep100_model.h5',custom_objects={'mean_loss':RANS_reynolds_stress_loss_wrapper(None,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),'ResidualLayer':ResidualLayer})
        # we need to compile again after loading once we can populate the loss function with the model object
        model_RANS.ScalingParameters = ScalingParameters
        model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=1E-5), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)
        model_RANS.summary()
        training_steps=100


#model_RANS.colloc_batch = tf.Variable(tf.zeros((model_RANS.ScalingParameters.batch_size,2),dtype=tf.float64),trainable=False,name='colloc_batch')

LBFGS_steps = 333
LBFGS_epochs = 3*LBFGS_steps


d_ts = 100
# training

global saveFig
saveFig=True



history_list = []
if True:
    # new style training with probabalistic training dataset
    lr_schedule = np.array([1E-4, 1E-5, 1E-6, 1E-7, 1E-5, 1E-5, 5E-6, 1E-6, 5E-7])
    ep_schedule = np.array([1, 25, 50, 75, 100, 150, 200, 250, 300])
    phys_schedule = np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

    # reset the correct learing rate on load
    i_temp = 0
    for i in range(len(ep_schedule)):
        if training_steps>=ep_schedule[i]:
            i_temp = i
    
    model_RANS.ScalingParameters.physics_loss_coefficient = phys_schedule[i_temp]
    model_RANS.ScalingParameters.boundary_loss_coefficient = phys_schedule[i_temp]
    model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule[i_temp]), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)


    while True:

        for i in range(len(ep_schedule)):
            if training_steps==ep_schedule[i]:
                keras.backend.set_value(model_RANS.optimizer.learning_rate, lr_schedule[i])
                print('epoch',str(training_steps))
                print('physics loss =',str(phys_schedule[i]))
                print('learning rate =',str(lr_schedule[i]))
                model_RANS.ScalingParameters.physics_loss_coefficient = phys_schedule[i]
                model_RANS.ScalingParameters.boundary_loss_coefficient = phys_schedule[i]
                if (i>0):
                    if (phys_schedule[i]!=phys_schedule[i-1]):
                        model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule[i]), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_vector,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)



        i_sampled, o_sampled = data_sample_by_err(i_train,o_train,i_train.shape[0])

        hist = model_RANS.fit(i_sampled,o_sampled, batch_size=32, epochs=1)
        training_steps = training_steps+1
        if (np.mod(training_steps,10)==0):
            save_custom(model_RANS,training_steps)
        plot_err(training_steps,model_RANS)

        


history_arr = history_list[0]
for i in range(1,len(history_list)):
    history_arr = np.concatenate((history_arr,history_list[i]),axis=0)

epoch_arr = np.arange(history_arr.shape[0])

plot.figure(1001)
plot.scatter(epoch_arr,history_arr)
plot.yscale('log')
plot.xscale('log')

plot.savefig(savedir+time.strftime("%Y%m%d_%H%M",time.localtime())+'training_history.png',dpi=300)
