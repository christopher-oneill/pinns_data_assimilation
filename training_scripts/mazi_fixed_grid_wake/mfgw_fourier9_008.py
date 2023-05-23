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
import sys

keras.backend.set_floatx('float64')
dtype_train = tf.float64

case = 'JFM'
start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

PLOT = False


job_name = 'mfgw_fourier9_008'

# Job mgfw_fourier008 (same as 007)
# 20230523: fourier mode assimilation, fixed cylinder Re=150
# physics loss set to 1, slow learning rate schedule


LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_beluga/sync/'
    HOMEDIR = 'C:/projects/pinns_beluga/sync/'
    sys.path.append('C:/projects/pinns_local/code/')
    # set number of cores to compute on 
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)
else:
    # parameters for running on compute canada    
    job_duration = timedelta(hours=22,minutes=30)
    end_time = start_time+job_duration
    print("This job is: ",job_name)
    useGPU=True
    HOMEDIR = '/home/coneill/sync/'
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    sys.path.append(HOMEDIR+'code/')
    # set number of cores to compute on 
    tf.config.threading.set_intra_op_parallelism_threads(12)
    tf.config.threading.set_inter_op_parallelism_threads(12)
    

# set the paths
save_loc = HOMEDIR+'output/'+job_name+'_output/'
checkpoint_filepath = save_loc+'checkpoint'
physics_loss_coefficient = 1.0
mode_number=8 # the number of the truncated mode to assimilate, note that this is mode 9 in matlab!


if useGPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    # if we are on the cluster, we need to check we use the right number of gpu, else we should raise an error
    expected_GPU=4
    assert len(physical_devices)==expected_GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid_wake/'
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
fourierModeFile = h5py.File(base_dir+'fourierDataShort.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')


ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxppuxpp = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxppuypp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyppuypp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

#psi_r = np.real(np.array(fourierModeFile['pressureModesShort'][mode_number,:])).transpose()
#psi_i = np.imag(np.array(fourierModeFile['pressureModesShort'][mode_number,:])).transpose()

tau_xx_r = np.array(fourierModeFile['stressModesShortReal'][0,mode_number,:]).transpose()
tau_xx_i = np.array(fourierModeFile['stressModesShortImag'][0,mode_number,:]).transpose()
tau_xy_r = np.array(fourierModeFile['stressModesShortReal'][1,mode_number,:]).transpose()
tau_xy_i = np.array(fourierModeFile['stressModesShortImag'][1,mode_number,:]).transpose()
tau_yy_r = np.array(fourierModeFile['stressModesShortReal'][2,mode_number,:]).transpose()
tau_yy_i = np.array(fourierModeFile['stressModesShortImag'][2,mode_number,:]).transpose()

omega = np.array(fourierModeFile['fShort'][0,mode_number])*2*np.pi

print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
y = np.array(configFile['X_vec'][1,:])
d = np.array(configFile['cylinderDiameter'])
print('u.shape: ',ux.shape)
print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)

nu_mol = 0.0066667

MAX_x = np.max(x.flatten())
MAX_y = np.max(y.flatten())
MAX_ux = np.max(ux.flatten())
MAX_uy = np.max(uy.flatten())
MIN_x = np.min(x.flatten())
MIN_y = np.min(y.flatten())
MIN_ux = np.min(ux.flatten())
MIN_uy = np.min(uy.flatten())
MAX_uxppuxpp = np.max(uxppuxpp.flatten())
MAX_uxppuypp = np.max(uxppuypp.flatten())
MAX_uyppuypp = np.max(uyppuypp.flatten())


MAX_phi_xr = np.max(phi_xr.flatten())
MAX_phi_xi = np.max(phi_xi.flatten())
MAX_phi_yr = np.max(phi_yr.flatten())
MAX_phi_yi = np.max(phi_yi.flatten())

MAX_tau_xx_r = np.max(tau_xx_r.flatten())
MAX_tau_xx_i = np.max(tau_xx_i.flatten())
MAX_tau_xy_r = np.max(tau_xy_r.flatten())
MAX_tau_xy_i = np.max(tau_xy_i.flatten())
MAX_tau_yy_r = np.max(tau_yy_r.flatten())
MAX_tau_yy_i = np.max(tau_yy_i.flatten())

print('max_x: ',MAX_x)
print('min_x: ',MIN_x)
print('max_y: ',MAX_y)
print('min_y: ',MIN_y)

MAX_p= 1 # estimated maximum pressure, we should 
MAX_psi= 0.1 # chosen based on abs(max(psi))

# reduce the collocation points to 25k
colloc_limits1 = np.array([[0.5,10.0],[-2.0,2.0]])
colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
colloc_merged = colloc_sample_lhs1(20000)
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
phi_xr_train = phi_xr/MAX_phi_xr
phi_xi_train = phi_xi/MAX_phi_xi
phi_yr_train = phi_yr/MAX_phi_yr
phi_yi_train = phi_yi/MAX_phi_yi

tau_xx_r_train = tau_xx_r/MAX_tau_xx_r
tau_xx_i_train = tau_xx_i/MAX_tau_xx_i
tau_xy_r_train = tau_xy_r/MAX_tau_xy_r
tau_xy_i_train = tau_xy_i/MAX_tau_xy_i
tau_yy_r_train = tau_yy_r/MAX_tau_yy_r
tau_yy_i_train = tau_yy_i/MAX_tau_yy_i

# the order here must be identical to inside the cost functions
O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
F_train = np.hstack(((phi_xr_train).reshape(-1,1),(phi_xi_train).reshape(-1,1),(phi_yr_train).reshape(-1,1),(phi_yi_train).reshape(-1,1),(tau_xx_r).reshape(-1,1),(tau_xx_i).reshape(-1,1),(tau_xy_r).reshape(-1,1),(tau_xy_i).reshape(-1,1),(tau_yy_r).reshape(-1,1),(tau_yy_i).reshape(-1,1))) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
# the order here must be identical to inside the cost functions



print('X_train.shape: ',X_train.shape)
print('O_train.shape: ',O_train.shape)

# mean model functions
@tf.function
def net_f_mean_cartesian(colloc_tensor):
    
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
    duxppuypp = tf.gradients(uxppuypp, colloc_tensor)[0]
    uxppuypp_x = duxppuypp[:,0]/MAX_x
    uxppuypp_y = duxppuypp[:,1]/MAX_y
    uyppuypp_y = tf.gradients(uyppuypp, colloc_tensor)[0][:,1]/MAX_y

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y


    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxppuxpp_x + uxppuypp_y) + p_x - (nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxppuypp_x + uyppuypp_y) + p_y - (nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    

    return f_x, f_y, f_mass

def mean_loss_wrapper(colloc_tensor_f): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''


        mx,my,mass = net_f_mean_cartesian(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(mx))
        physical_loss2 = tf.reduce_mean(tf.square(my))
        physical_loss3 = tf.reduce_mean(tf.square(mass))
                      
        return data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp + physics_loss_coefficient*(physical_loss1 + physical_loss2 + physical_loss3) # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

    return custom_loss

@tf.function
def mean_cartesian(colloc_tensor):

    u_mean = model(colloc_tensor)
    ux = u_mean[:,0]*MAX_ux
    uy = u_mean[:,1]*MAX_uy

    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/MAX_x
    ux_y = dux[:,1]/MAX_y

    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/MAX_x
    uy_y = duy[:,1]/MAX_y

    return tf.stack([ux,uy,ux_x,ux_y,uy_x,uy_y],axis=1)

# fourier NN functions
@tf.function
def net_f_fourier_cartesian(colloc_tensor, mean_grads):
    
    up = model_fourier(colloc_tensor)
    # velocity fourier coefficients
    phi_xr = up[:,0]*MAX_phi_xr
    phi_xi = up[:,1]*MAX_phi_xi
    phi_yr = up[:,2]*MAX_phi_yr
    phi_yi = up[:,3]*MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*MAX_tau_xx_r
    tau_xx_i = up[:,5]*MAX_tau_xx_i
    tau_xy_r = up[:,6]*MAX_tau_xy_r
    tau_xy_i = up[:,7]*MAX_tau_xy_i
    tau_yy_r = up[:,8]*MAX_tau_yy_r
    tau_yy_i = up[:,9]*MAX_tau_yy_i
    # unknowns, pressure fourier modes
    psi_r = up[:,10]*MAX_psi
    psi_i = up[:,11]*MAX_psi
    
    ux = mean_grads[:,0]
    uy = mean_grads[:,1]
    ux_x = mean_grads[:,2]
    ux_y = mean_grads[:,3]
    uy_x = mean_grads[:,4]
    uy_y = mean_grads[:,5]


    # compute the gradients of the quantities
    
    # phi_xr gradient
    dphi_xr = tf.gradients(phi_xr, colloc_tensor)[0]
    phi_xr_x = dphi_xr[:,0]/MAX_x
    phi_xr_y = dphi_xr[:,1]/MAX_y
    # and second derivative
    phi_xr_xx = tf.gradients(phi_xr_x, colloc_tensor)[0][:,0]/MAX_x
    phi_xr_yy = tf.gradients(phi_xr_x, colloc_tensor)[0][:,1]/MAX_y

    # phi_xi gradient
    dphi_xi = tf.gradients(phi_xi, colloc_tensor)[0]
    phi_xi_x = dphi_xi[:,0]/MAX_x
    phi_xi_y = dphi_xi[:,1]/MAX_y
    # and second derivative
    phi_xi_xx = tf.gradients(phi_xi_x, colloc_tensor)[0][:,0]/MAX_x
    phi_xi_yy = tf.gradients(phi_xi_x, colloc_tensor)[0][:,1]/MAX_y

    # phi_yr gradient
    dphi_yr = tf.gradients(phi_yr, colloc_tensor)[0]
    phi_yr_x = dphi_yr[:,0]/MAX_x
    phi_yr_y = dphi_yr[:,1]/MAX_y
    # and second derivative
    phi_yr_xx = tf.gradients(phi_yr_x, colloc_tensor)[0][:,0]/MAX_x
    phi_yr_yy = tf.gradients(phi_yr_x, colloc_tensor)[0][:,1]/MAX_y
    
    # phi_yi gradient
    dphi_yi = tf.gradients(phi_yi, colloc_tensor)[0]
    phi_yi_x = dphi_yi[:,0]/MAX_x
    phi_yi_y = dphi_yi[:,1]/MAX_y
    # and second derivative
    phi_yi_xx = tf.gradients(phi_yi_x, colloc_tensor)[0][:,0]/MAX_x
    phi_yi_yy = tf.gradients(phi_yi_x, colloc_tensor)[0][:,1]/MAX_y

    # gradient reynolds stress fourier component, real
    tau_xx_r_x = tf.gradients(tau_xx_r, colloc_tensor)[0][:,0]/MAX_x
    dtau_xy_r = tf.gradients(tau_xy_r, colloc_tensor)[0]
    tau_xy_r_x = dtau_xy_r[:,0]/MAX_x
    tau_xy_r_y = dtau_xy_r[:,1]/MAX_y
    tau_yy_r_y = tf.gradients(tau_yy_r, colloc_tensor)[0][:,1]/MAX_y
    # gradient reynolds stress fourier component, complex
    tau_xx_i_x = tf.gradients(tau_xx_i, colloc_tensor)[0][:,0]/MAX_x
    dtau_xy_i = tf.gradients(tau_xy_i, colloc_tensor)[0]
    tau_xy_i_x = dtau_xy_i[:,0]/MAX_x
    tau_xy_i_y = dtau_xy_i[:,1]/MAX_y
    tau_yy_i_y = tf.gradients(tau_yy_i, colloc_tensor)[0][:,1]/MAX_y

    # pressure gradients
    dpsi_r = tf.gradients(psi_r, colloc_tensor)[0]
    psi_r_x = dpsi_r[:,0]/MAX_x
    psi_r_y = dpsi_r[:,1]/MAX_y
    dpsi_i = tf.gradients(psi_i, colloc_tensor)[0]
    psi_i_x = dpsi_i[:,0]/MAX_x
    psi_i_y = dpsi_i[:,1]/MAX_y

    # governing equations
    f_xr = -omega*phi_xi+(phi_xr*ux_x + phi_yr*ux_y+ ux*phi_xr_x +uy*phi_xr_y ) + (tau_xx_r_x + tau_xy_r_y) + psi_r_x - (nu_mol)*(phi_xr_xx+phi_xr_yy)  
    f_xi =  omega*phi_xr+(phi_xi*ux_x + phi_yi*ux_y+ ux*phi_xi_x +uy*phi_xi_y ) + (tau_xx_i_x + tau_xy_i_y) + psi_i_x - (nu_mol)*(phi_xi_xx+phi_xi_yy)  
    f_yr = -omega*phi_yi+(phi_xr*uy_x + phi_yr*uy_y+ ux*phi_yr_x +uy*phi_yr_y ) + (tau_xy_r_x + tau_yy_r_y) + psi_r_y - (nu_mol)*(phi_yr_xx+phi_yr_yy) 
    f_yi =  omega*phi_yr+(phi_xi*uy_x + phi_yi*uy_y+ ux*phi_yi_x +uy*phi_yi_y ) + (tau_xy_i_x + tau_yy_i_y) + psi_i_y - (nu_mol)*(phi_yi_xx+phi_yi_yy)  
    f_mr = phi_xr_x + phi_yr_y
    f_mi = phi_xi_x + phi_yi_y

    return f_xr,f_xi, f_yr,f_yi, f_mr, f_mi


# function wrapper, combine data and physics loss
def fourier_loss_wrapper(colloc_tensor_f,colloc_grads): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_phi_xr = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0])
        data_loss_phi_xi = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) 
        data_loss_phi_yr = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) 
        data_loss_phi_yi = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) 
        data_loss_tau_xx_r = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) 
        data_loss_tau_xx_i = keras.losses.mean_squared_error(y_true[:,5], y_pred[:,5]) 
        data_loss_tau_xy_r = keras.losses.mean_squared_error(y_true[:,6], y_pred[:,6]) 
        data_loss_tau_xy_i = keras.losses.mean_squared_error(y_true[:,7], y_pred[:,7])
        data_loss_tau_yy_r = keras.losses.mean_squared_error(y_true[:,8], y_pred[:,8]) 
        data_loss_tau_yy_i = keras.losses.mean_squared_error(y_true[:,9], y_pred[:,9]) 


        mxr,mxi,myr,myi,massr,massi = net_f_fourier_cartesian(colloc_tensor_f,colloc_grads)
        loss_mxr = tf.reduce_mean(tf.square(mxr))
        loss_mxi = tf.reduce_mean(tf.square(mxi))
        loss_myr = tf.reduce_mean(tf.square(myr))
        loss_myi = tf.reduce_mean(tf.square(myi))
        loss_massr = tf.reduce_mean(tf.square(massr))
        loss_massi = tf.reduce_mean(tf.square(massi))
                      
        return data_loss_phi_xr + data_loss_phi_xi + data_loss_phi_yr + data_loss_phi_yi +data_loss_tau_xx_r+data_loss_tau_xx_i+data_loss_tau_xy_r+data_loss_tau_xy_i+data_loss_tau_yy_r+data_loss_tau_yy_i + physics_loss_coefficient*(loss_mxr + loss_mxi + loss_myr+loss_myi+loss_massr+loss_massi) # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

    return custom_loss

# create the NNs
# create mean NN on the CPU
dense_nodes = 50
dense_layers = 10

with tf.device('/CPU:0'):
    model = keras.Sequential()
    model.add(keras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
    for i in range(dense_layers-1):
        model.add(keras.layers.Dense(dense_nodes, activation='tanh'))
    model.add(keras.layers.Dense(6,activation='linear'))
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = mean_loss_wrapper(tf.cast(f_colloc_train,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)
    model.trainable=False

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))


# load the saved mean model
model.load_weights(HOMEDIR+'/output/mfgw_mean003_output/mfgw_mean003_ep416')
# get the values for the mean_data tensor
mean_data = mean_cartesian(f_colloc_train)

# clear the session, we will now create the fourier model
tf.keras.backend.clear_session()

fourier_nodes = 75
fourier_layers = 10
if useGPU:
    tf_device_string = ['GPU:0']
    for ngpu in range(1,len(physical_devices)):
        tf_device_string.append('GPU:'+str(ngpu))
            
    strategy = tf.distribute.MirroredStrategy(devices=tf_device_string)
    print('Using devices: ',tf_device_string)
    with strategy.scope():
        model_fourier = keras.Sequential()
        model_fourier.add(keras.layers.Dense(fourier_nodes, activation='tanh', input_shape=(2,)))
        for i in range(fourier_layers-1):
            model_fourier.add(keras.layers.Dense(fourier_nodes, activation='tanh'))
        model_fourier.add(keras.layers.Dense(12,activation='linear'))
        model_fourier.summary()
        model_fourier.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = fourier_loss_wrapper(tf.cast(f_colloc_train,dtype_train),tf.cast(mean_data,dtype_train)),jit_compile=False) 
else:
    with tf.device('/CPU:0'):
        model_fourier = keras.Sequential()
        model_fourier.add(keras.layers.Dense(fourier_nodes, activation='tanh', input_shape=(2,)))
        for i in range(dense_layers-1):
            model_fourier.add(keras.layers.Dense(fourier_nodes, activation='tanh'))
        model_fourier.add(keras.layers.Dense(12,activation='linear'))
        model_fourier.summary()
        model_fourier.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = fourier_loss_wrapper(tf.cast(f_colloc_train,dtype_train),tf.cast(mean_data,dtype_train)),jit_compile=False) 

fourier_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
fourier_early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)



# check if the model has been created before, if so load it
def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))
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
    model_fourier.load_weights(HOMEDIR+'/output/'+job_name+'_output/'+job_name+'_ep'+str(epochs))
else:
    # if not, we train from the beginning
    epochs = 0
    if ~os.path.isdir(HOMEDIR+'/output/'+job_name+'_output/'):
        os.mkdir(HOMEDIR+'/output/'+job_name+'_output/')


# this time we randomly shuffle the order of X and O
rng = np.random.default_rng()
# train the network
d_epochs = 1
X_train = tf.cast(X_train,dtype_train)
F_train = tf.cast(F_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs

if node_name ==LOCAL_NODE:
    pass    
else:
    shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
    temp_X_train = X_train[shuffle_inds,:]
    temp_Y_train = F_train[shuffle_inds,:]
    # compute canada training loop; use time based training
    while True:

        if np.mod(epochs,10)==0:
            shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
            temp_X_train = X_train[shuffle_inds,:]
            temp_Y_train = F_train[shuffle_inds,:]
        hist = model_fourier.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[fourier_early_stop_callback,fourier_checkpoint_callback])
        epochs = epochs+d_epochs

        if epochs>20:
            keras.backend.set_value(model.optimizer.learning_rate, 1E-3)
        if epochs>50:
            keras.backend.set_value(model.optimizer.learning_rate, 1E-4)
        if epochs>100:
            keras.backend.set_value(model.optimizer.learning_rate, 1E-5)
        if epochs>200:
            keras.backend.set_value(model.optimizer.learning_rate, 1E-6)

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
