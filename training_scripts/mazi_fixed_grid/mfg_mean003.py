#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:59:20 2022

@author: saldern

Moritz S. PIV data

Assimilate the teta velocity based on 2D Velocity data
This case includes more equations, we try to find nut and ut simultaneous

we need to apply a boundary condition for ut
add pressure to the system

same as _8 but with gradient in eddy viscosity

this case has a hole line of training data in theta at x/max_x=0.5
and ut = 0 at r/max_r = 1 and x/max_x<0.3

FINAL CASE FOR JFM

# this case is the final one used for the PINN paper
# during second revision we realized that the units of the length are in mm not in m
# this is not problamatic, since the error cancels out in almost every term! (since the eq. are multiplied with r (also in mm))
# the viscous terms have two spatial derivatives (so two time the error multiplied with r yields only one error left of factor 
10-3) this in turn means that the PINN estimates an viscosity which is 10^3 larger! This in turn corresponds to using
the length in the correct unit (meters) and choosing a MAX_nu of 10^3 larger.
-> when using this code keep in mind that the resulting eddy viscosity is 10^3 too large

-> alternatively run the code again, case rev2_2 with meters

we decided to stick with this case here and update the MAX_nut stated in the manuscript (otherwise we would have to change to figures/results)

(however, if equations are not multiplied with r than training will be more difficult (as the mm do not cancel out)
So this case should not be copied without correcting it so meters.)

13.10.22 This case is still the final one used for POF
(To check that everythin is correct I compute the same solution with length in m and MAX_nut = 0.00001 in swirl_pinn_POF_rev2_3.py)


"""


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

keras.backend.set_floatx('float64')
dtype_train = tf.float64

case = 'JFM'
start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

PLOT = False


job_name = 'mfg_mean002'

# Job mgf_mean002
# mean field assimilation for the fixed cylinder, now on a regular grid, only one gpu
# also with poisson equation


LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_local/'
    HOMEDIR = 'C:/projects/pinns_local/'
else:
    # parameters for running on compute canada
    
    job_duration = timedelta(hours=22,minutes=30)
    end_time = start_time+job_duration
    print("This job is: ",job_name)
    useGPU=True
    HOMEDIR = '/home/coneill/sync/'
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    

# set the paths
save_loc = HOMEDIR+'output/'+job_name+'_output/'
checkpoint_filepath = save_loc+'checkpoint'
physics_loss_coefficient = 1.0
# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

if useGPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    # if we are on the cluster, we need to check we use the right number of gpu, else we should raise an error
    expected_GPU=1
    assert len(physical_devices)==expected_GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')


ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxppuxpp = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxppuypp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyppuypp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()


print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
y = np.array(configFile['X_vec'][1,:])
d = np.array(configFile['cylinderDiameter'])
print('u.shape: ',ux.shape)
print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)

nu_mol = 0.0066667

MAX_x = max(x.flatten())
MAX_y = max(y.flatten())
MAX_ux = max(ux.flatten())
MAX_uy = max(uy.flatten())
MIN_x = min(x.flatten())
MIN_y = min(y.flatten())
MIN_ux = min(ux.flatten())
MIN_uy = min(uy.flatten())
MAX_uxppuxpp = max(uxppuxpp.flatten())
MAX_uxppuypp = max(uxppuypp.flatten())
MAX_uyppuypp = max(uyppuypp.flatten())


print('max_x: ',MAX_x)
print('min_x: ',MIN_x)
print('max_y: ',MAX_y)
print('min_y: ',MIN_y)



MAX_p= 1 # estimated maximum pressure, we should 

# reduce the collocation points to 25k
colloc_limits1 = np.array([[-2.0,10.0],[-2.0,2.0]])
colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
colloc_lhs1 = colloc_sample_lhs1(20000)
print('colloc_lhs1.shape',colloc_lhs1.shape)

# remove points inside the cylinder
c1_loc = np.array([0,0],dtype=np.float64)
cylinder_inds = np.less(np.power(np.power(colloc_lhs1[:,0]-c1_loc[0],2)+np.power(colloc_lhs1[:,1]-c1_loc[1],2),0.5*d),0.5)
print(cylinder_inds.shape)
colloc_merged = np.delete(colloc_lhs1,cylinder_inds[0,:],axis=0)
print('colloc_merged.shape',colloc_merged.shape)

f_colloc_train = colloc_merged*np.array([1/MAX_x,1/MAX_y])

# normalize the training data:
x_train = x/MAX_x
y_train = y/MAX_y
ux_train = ux/MAX_ux
uy_train = uy/MAX_uy
uxppuxpp_train = uxppuxpp/MAX_uxppuxpp
uxppuypp_train = uxppuypp/MAX_uxppuypp
uyppuypp_train = uyppuypp/MAX_uyppuypp


# the order here must be identical to inside the cost functions
O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))

print('X_train.shape: ',X_train.shape)
print('O_train.shape: ',O_train.shape)

@tf.function
def net_f_cartesian(colloc_tensor):
    
    up = model(colloc_tensor)
    # knowns
    ux = up[:,0]*MAX_ux
    uy = up[:,1]*MAX_uy
    uxppuxpp = up[:,2]*MAX_uxppuxpp
    uxppuypp = up[:,3]*MAX_uxppuypp
    uyppuypp = up[:,4]*MAX_uyppuypp
    # unknowns
    p = up[:,5]*MAX_p
    
    # compute the gradients of the quantities
    
    # ux gradient
    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/MAX_x
    ux_y = dux[:,1]/MAX_y
    # and second derivative
    ux_xx = tf.gradients(ux_x, colloc_tensor)[0][:,0]/MAX_x
    ux_yy = tf.gradients(ux_y, colloc_tensor)[0][:,1]/MAX_y
    
    # uy gradient
    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/MAX_x
    uy_y = duy[:,1]/MAX_y
    # and second derivative
    uy_xx = tf.gradients(uy_x, colloc_tensor)[0][:,0]/MAX_x
    uy_yy = tf.gradients(uy_y, colloc_tensor)[0][:,1]/MAX_y

    # gradient unmodeled reynolds stresses
    uxppuxpp_x = tf.gradients(uxppuxpp, colloc_tensor)[0][:,0]/MAX_x
    uxppuxpp_xx = tf.gradients(uxppuxpp_x, colloc_tensor)[0][:,0]/MAX_x
    duxppuypp = tf.gradients(uxppuypp, colloc_tensor)[0]
    uxppuypp_x = duxppuypp[:,0]/MAX_x
    uxppuypp_y = duxppuypp[:,1]/MAX_y
    uxppuypp_xy = tf.gradients(uxppuypp_x, colloc_tensor)[0][:,1]/MAX_y
    uyppuypp_y = tf.gradients(uyppuypp, colloc_tensor)[0][:,1]/MAX_y
    uyppuypp_yy = tf.gradients(uyppuypp_y, colloc_tensor)[0][:,1]/MAX_y

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y
    p_xx = tf.gradients(p_x,colloc_tensor)[0][:,0]/MAX_x
    p_yy = tf.gradients(p_y,colloc_tensor)[0][:,1]/MAX_y


    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxppuxpp_x + uxppuypp_y) + p_x - (nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxppuypp_x + uyppuypp_y) + p_y - (nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    
    # poisson equation
    f_p = p_xx + p_yy + tf.math.pow(ux_x,tf.constant(2.0,dtype=dtype_train)) + 2*ux_y*uy_x + tf.math.pow(uy_y,tf.constant(2.0,dtype=dtype_train))+uxppuxpp_xx+2*uxppuypp_xy+uyppuypp_yy

    return f_x, f_y, f_mass, f_p


# function wrapper, combine data and physics loss
def custom_loss_wrapper(colloc_tensor_f): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''


        mx,my,mass,mp = net_f_cartesian(colloc_tensor_f)
        loss_mx = tf.reduce_mean(tf.square(mx))
        loss_my = tf.reduce_mean(tf.square(my))
        loss_mass = tf.reduce_mean(tf.square(mass))
        loss_pe = tf.reduce_mean(tf.square(mp))
                      
        return data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp + physics_loss_coefficient*(loss_mx + loss_my + loss_mass + loss_pe) # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

    return custom_loss


# create NN
dense_nodes = 50
dense_layers = 10
if useGPU:
    tf_device_string = ['GPU:0']
    for ngpu in range(1,len(physical_devices)):
        tf_device_string.append('GPU:'+str(ngpu))
else:
    tf_device_string = '/CPU:0'

strategy = tf.distribute.MirroredStrategy(devices=tf_device_string)

print('Using devices: ',tf_device_string)
with strategy.scope():
    model = keras.Sequential()
    model.add(keras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
    for i in range(dense_layers-1):
        model.add(keras.layers.Dense(dense_nodes, activation='tanh'))
    model.add(keras.layers.Dense(6,activation='linear'))
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = custom_loss_wrapper(tf.cast(f_colloc_train,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))

# this time we randomly shuffle the order of X and O
rng = np.random.default_rng()

# job_name = 'RS3_CC0001_23h'
# we need to check if there are already checkpoints for this job
checkpoint_files = get_filepaths_with_glob(HOMEDIR+'/output/'+job_name+'_output/',job_name+'_ep*.index')
if len(checkpoint_files)>0:
    files_epoch_number = np.zeros([len(checkpoint_files),1],dtype=np.uint)
    # if there are checkpoints, train based on the most recent checkpoint
    for f_indx in range(len(checkpoint_files)):
        re_result = re.search("ep[0-9]*",checkpoint_files[f_indx])
        files_epoch_number[f_indx]=int(checkpoint_files[f_indx][(re_result.start()+2):re_result.end()])
    epochs = np.uint(np.max(files_epoch_number))
    print(HOMEDIR+'/output/'+job_name+'_output/',job_name+'_ep'+str(epochs))
    model.load_weights(HOMEDIR+'/output/'+job_name+'_output/'+job_name+'_ep'+str(epochs))
else:
    # if not, we train from the beginning
    epochs = 0
    if ~os.path.isdir(HOMEDIR+'/output/'+job_name+'_output/'):
        os.mkdir(HOMEDIR+'/output/'+job_name+'_output/')


# train the network
d_epochs = 1
X_train = tf.cast(X_train,dtype_train)
O_train = tf.cast(O_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs

if node_name ==LOCAL_NODE:
    # local node training loop, save every epoch for testing
    if True:
        for e in range(10):
            shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
            temp_X_train = X_train[shuffle_inds,:]
            temp_Y_train = O_train[shuffle_inds,:]
            hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=16, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
            epochs = epochs+d_epochs
            model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))   
else:
    shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
    temp_X_train = X_train[shuffle_inds,:]
    temp_Y_train = O_train[shuffle_inds,:]
    # compute canada training loop; use time based training
    while True:
        if np.mod(epochs,10)==0:
            shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
            temp_X_train = X_train[shuffle_inds,:]
            temp_Y_train = O_train[shuffle_inds,:]
        hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
        epochs = epochs+d_epochs

        if epochs>=25:
            
            keras.backend.set_value(model.optimizer.learning_rate, 0.005)
        if epochs>=50:
            
            keras.backend.set_value(model.optimizer.learning_rate, 0.001)
            
        if np.mod(epochs,10)==0:
            # save every 10th epoch
            model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))
            pred = model.predict(X_train,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close() 

        # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            # save the last epoch before exiting
            model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))
            pred = model.predict(X_train,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close()
            exit()
        last_epoch_time = datetime.now()
