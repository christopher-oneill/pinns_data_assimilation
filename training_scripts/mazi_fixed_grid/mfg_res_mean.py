


import tensorflow as tf
import tensorflow.keras as keras
import h5py
#import tensorflow_probability as tfp
from smt.sampling_methods import LHS

keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')


#from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import FourierResidualLayer64
from pinns_data_assimilation.lib.layers import ProductResidualLayer64
from pinns_data_assimilation.lib.layers import CubicFourierProductBlock64
from pinns_data_assimilation.lib.layers import FourierEmbeddingLayer
from pinns_data_assimilation.lib.layers import AdjustableFourierTransformLayer
from pinns_data_assimilation.lib.layers import QuarticFourierProductBlock64

from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center



def plot_err(epoch,model_RANS):
    global p_grid
    global X_grid
    global Y_grid
    global i_test
    global o_test_grid
    pred_test = model_RANS.predict(i_test[:],batch_size=1000)
    pred_test_grid = np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])

    plot.close(epoch)
    plot.close(epoch+1)
    plot.close(epoch+2)
    plot.close(epoch+3)
    plot.close(epoch+4)
    plot.close(epoch+5)

    err_test1 = o_test_grid[:,:,0]-pred_test_grid[:,:,0]
    plot.figure(epoch)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,0],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,0],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test1,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test2 = o_test_grid[:,:,1]-pred_test_grid[:,:,1]
    plot.figure(epoch+1)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,1],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,1],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test2,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test3 = o_test_grid[:,:,2]-pred_test_grid[:,:,2]
    plot.figure(epoch+2)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,2],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,2],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test3,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test4 = o_test_grid[:,:,3]-pred_test_grid[:,:,3]
    plot.figure(epoch+3)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,3],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,3],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test4,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test5 = o_test_grid[:,:,4]-pred_test_grid[:,:,4]
    plot.figure(epoch+4)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,4],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,4],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test5,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
   
    err_test5 = p_grid-pred_test_grid[:,:,5]
    plot.figure(epoch+5)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,p_grid,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,5],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test5,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    plot.show(block=False)
    plot.pause(1)

def save_custom(model,epochs):
    global i_test
    global savedir
    model.save(savedir+'mfg_res_mean_ep'+str(np.uint(epochs))+'_model.h5')
    pred = model.predict(i_test,batch_size=1024)
    h5f = h5py.File(savedir+'mfg_res_mean_ep'+str(np.uint(epochs))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
    if model.ScalingParameters.physics_loss_coefficient!=0:
        t_mx,t_my,t_mass = RANS_reynolds_stress_cartesian(model,i_test)
        h5f = h5py.File(savedir+'mfg_res_mean_ep'+str(np.uint(epochs))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists



plot.ion()


HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
global savedir
savedir = HOMEDIR+'output/mfg_res_mean003/'
create_directory_if_not_exists(savedir)

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

global i_test
i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))


MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)
x_test = X_grid/MAX_x
y_test = Y_grid/MAX_x


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







i_train = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

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
ScalingParameters.physics_loss_coefficient = 1.0
ScalingParameters.boundary_loss_coefficient = 1.0
ScalingParameters.data_loss_coefficient = 1.0E3

# define boundary condition points
theta = np.linspace(0,2*np.pi,1000)
ns_BC_x = 0.5*d*np.cos(theta)/ScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/ScalingParameters.MAX_y
cyl_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1)))

p_BC_x = np.array([10.0,10.0])/ScalingParameters.MAX_x
p_BC_y = np.array([-2.0,2.0])/ScalingParameters.MAX_y
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

inlet_BC_x = -10.0*np.ones([500,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,500)/ScalingParameters.MAX_y
inlet_BC_vec = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

inlet_BC_x = -2.0*np.ones([500,1])/ScalingParameters.MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,500)/ScalingParameters.MAX_y
inlet_BC_vec2 = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        global training_steps
        global model_RANS
        if np.mod(epoch,1000)==0:
            plot_err(training_steps,model_RANS)
            save_custom(model_RANS,training_steps + epoch)
        





# define the collocation points

def colloc_points():
    
    data_points = np.hstack((np.reshape(x,[x.size,1]),np.reshape(y,[y.size,1])))

    colloc_limits1 = np.array([[-10.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
    colloc_lhs1 = colloc_sample_lhs1(70000)


    colloc_limits2 = np.array([[-1.0,3.0],[-1.5,1.5]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(10000)

    colloc_merged = np.vstack((data_points,colloc_lhs1,colloc_lhs2))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print(cylinder_inds.shape)
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/ScalingParameters.MAX_x,1/ScalingParameters.MAX_y])
    return f_colloc_train

f_colloc_train = colloc_points()

# import the physics
from pinns_data_assimilation.lib.navier_stokes_cartesian import RANS_reynolds_stress_cartesian
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_pressure
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_no_slip
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet2

def RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_points,BC_p,BC_ns,BC_inlet,BC_inlet2): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''
        scaling_loss_p = tf.reduce_mean(tf.square(tf.math.maximum(tf.math.abs(y_pred[:,5])-1,0)))
        # we add the pressure scaling loss as well because it helps the numerical condition of the model
        data_loss = tf.reduce_mean(data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp+ scaling_loss_p)

        if (model_RANS.ScalingParameters.physics_loss_coefficient==0):
            physics_loss = 0.0
        else:
            if (model_RANS.ScalingParameters.batch_size==colloc_points.shape[0]):
                # all colloc points
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,colloc_points)
            else:
                # random selection of collocation points with batch size
                rand_colloc_points = np.random.choice(colloc_points.shape[0],model_RANS.ScalingParameters.batch_size)
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,tf.gather(colloc_points,rand_colloc_points))
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
        return  tf.math.reduce_max((dynamic_data_weight,1))*model_RANS.ScalingParameters.data_loss_coefficient*data_loss + combined_phyisics_loss
        
    return mean_loss


global training_steps
global model_RANS
# model creation
if True:
    if False:
        training_steps = 0
        nodes = 100
        with tf.device('/GPU:0'):
            model_RANS = keras.Sequential()
            model_RANS.add(keras.layers.Dense(nodes,activation='linear',input_shape=(2,)))
            for k in range(10):
                model_RANS.add(ResidualLayer(nodes,activation='elu'))
            model_RANS.add(keras.layers.Dense(6,activation='linear'))
            model_RANS.summary()
            model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=RANS_reynolds_stress_loss_wrapper(model_RANS,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) 

    else:
        with tf.device('/GPU:0'):
            model_RANS = keras.models.load_model(savedir+'mfg_res_mean_ep320_model.h5',custom_objects={'mean_loss':RANS_reynolds_stress_loss_wrapper(None,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),'ResidualLayer':ResidualLayer})
            # we need to compile again after loading once we can populate the loss function with the model object
            model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)
            model_RANS.ScalingParameters = ScalingParameters
            model_RANS.summary()
            training_steps=320
else:
    # model on CPU
    if False:
        
        training_steps = 0
        nodes = 100
        with tf.device('/CPU:0'):
            
            model_RANS = keras.Sequential()
            model_RANS.add(keras.layers.Dense(nodes,activation='linear',input_shape=(2,)))
            for k in range(10):
                model_RANS.add(ResidualLayer(nodes,activation='elu'))
            model_RANS.add(keras.layers.Dense(6,activation='linear'))
            model_RANS.summary()
            model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=RANS_reynolds_stress_loss_wrapper(model_RANS,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) 
    else:
        with tf.device('/CPU:0'):
            model_RANS = keras.models.load_model(savedir+'mfg_res_mean_ep19_model.h5',custom_objects={'mean_loss':RANS_reynolds_stress_loss_wrapper(None,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),'ResidualLayer':ResidualLayer})
            # we need to compile again after loading once we can populate the loss function with the model object
            model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss = RANS_reynolds_stress_loss_wrapper(model_RANS,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2),jit_compile=False) #(...,BC_points1,...,BC_points3)
            model_RANS.ScalingParameters = ScalingParameters
            model_RANS.summary()
            training_steps=20



model_RANS.ScalingParameters = ScalingParameters

# shuffle
shuffle_inds = np.array(range((i_train).shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

i_train_shuffle = tf.cast((i_train[shuffle_inds,:])[0,:,:],tf.float64)
o_train_shuffle = tf.cast((o_train[shuffle_inds])[0,:],tf.float64)
print(i_train_shuffle.shape)
print(o_train_shuffle.shape)

LBFGS_steps = 333
LBFGS_epochs = 3*LBFGS_steps


d_ts = 100
# training

if False:
    model_RANS.ScalingParameters.batch_size=128
    model_RANS.ScalingParameters.physics_loss_coefficient=1.0
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-5)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=d_ts,callbacks=[CustomCallback()])
    training_steps = training_steps+d_ts

    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-6)
    model_RANS.ScalingParameters.batch_size=128
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=d_ts,callbacks=[CustomCallback()])
    training_steps = training_steps+d_ts

    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-7)
    model_RANS.ScalingParameters.batch_size=128
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=d_ts,callbacks=[CustomCallback()])
    training_steps = training_steps+d_ts

    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-8)
    model_RANS.ScalingParameters.batch_size=128
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=d_ts,callbacks=[CustomCallback()])
    training_steps = training_steps+d_ts
    
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-9)
    model_RANS.ScalingParameters.batch_size=128
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=d_ts,callbacks=[CustomCallback()])
    training_steps = training_steps+d_ts
    
if False:
    
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-4)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(1000,model_RANS)
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-5)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(1500,model_RANS)
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-6)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(2000,model_RANS)
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-7)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(2500,model_RANS)
    pred =  model_RANS.predict(i_train,batch_size=1000)


model_RANS.ScalingParameters.physics_loss_coefficient=np.float64(1.0)
model_RANS.ScalingParameters.boundary_loss_coefficient=np.float64(1.0)
model_RANS.ScalingParameters.batch_size = 128

#keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-5)
#hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=(supersample_factor*supersample_factor)*8,callbacks=[CustomCallback()])
#training_steps = training_steps+8

model_RANS.ScalingParameters.batch_size = 1024
keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-6)
hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=(supersample_factor*supersample_factor)*200,callbacks=[CustomCallback()])
training_steps = training_steps+200
save_custom(model_RANS,training_steps)

model_RANS.ScalingParameters.batch_size = 2048
keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-7)
hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=(supersample_factor*supersample_factor)*500,callbacks=[CustomCallback()])
training_steps = training_steps+200
save_custom(model_RANS,training_steps)

model_RANS.ScalingParameters.batch_size = 8192
keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-8)
hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=(supersample_factor*supersample_factor)*1000,callbacks=[CustomCallback()])
training_steps = training_steps+300
save_custom(model_RANS,training_steps)

model_RANS.ScalingParameters.batch_size = 8192 #f_colloc_train.shape[0]
if False:
    for L_iter in range(100):

        func = function_factory(model_RANS, RANS_reynolds_stress_loss_wrapper(model_RANS,f_colloc_train,p_BC_vec,cyl_BC_vec,inlet_BC_vec,inlet_BC_vec2), tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
        init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables)

        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        training_steps = training_steps + LBFGS_epochs
        plot_err(training_steps,model_RANS)
        save_custom(model_RANS,training_steps)



plot.show(block=True)

