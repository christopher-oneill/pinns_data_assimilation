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


job_name = 'CC0005'

# Job CC0005 Notes
# Case: Mazi Fixed
# 75 nodes wide 
# Now trying to train with 6 POD modes. No poisson equation. 

LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_local/'
else:
    # parameters for running on compute canada
    
    job_duration = timedelta(hours=1,minutes=30)
    end_time = start_time+job_duration

    useGPU=True
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    

# set the paths
save_loc = SLURM_TMPDIR+'/output/'+job_name+'_output/'
checkpoint_filepath = save_loc+'checkpoint'

# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

# limit the gpu memory

if useGPU:
    physical_devices = tf.config.list_physical_devices('GPU')
   # try:
   #     tf.config.set_logical_device_configuration(
    #        physical_devices[0],
    #        [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])

    #except:
    # Invalid device or cannot modify logical devices once initialized.
       # pass
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# read the data
base_dir = SLURM_TMPDIR+'/data/mazi_fixed_modes/'
meanFieldFile = h5py.File(base_dir+'meanField.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
mode_dataFile = h5py.File(base_dir+'mode_data6.mat','r')


ux = np.array(meanFieldFile['meanField'][0,:]).transpose()
uy = np.array(meanFieldFile['meanField'][1,:]).transpose()

phi_1x = np.array(mode_dataFile['Phi_i'][0,0,:]).transpose()
phi_1y = np.array(mode_dataFile['Phi_i'][1,0,:]).transpose()
phi_2x = np.array(mode_dataFile['Phi_i'][0,1,:]).transpose()
phi_2y = np.array(mode_dataFile['Phi_i'][1,1,:]).transpose()
phi_3x = np.array(mode_dataFile['Phi_i'][0,2,:]).transpose()
phi_3y = np.array(mode_dataFile['Phi_i'][1,2,:]).transpose()
phi_4x = np.array(mode_dataFile['Phi_i'][0,3,:]).transpose()
phi_4y = np.array(mode_dataFile['Phi_i'][1,3,:]).transpose()
phi_5x = np.array(mode_dataFile['Phi_i'][0,4,:]).transpose()
phi_5y = np.array(mode_dataFile['Phi_i'][1,4,:]).transpose()
phi_6x = np.array(mode_dataFile['Phi_i'][0,5,:]).transpose()
phi_6y = np.array(mode_dataFile['Phi_i'][1,5,:]).transpose()


A1 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,0])
A2 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,1])
A3 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,2])
A4 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,3])
A5 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,4])
A6 = np.ones(ux.shape,dtype=np.float64)*np.float64(mode_dataFile['A_i'][0,5])

uxppuxpp = np.array(mode_dataFile['residual_stress'][0,:]).transpose()
uxppuypp = np.array(mode_dataFile['residual_stress'][1,:]).transpose()
uyppuypp = np.array(mode_dataFile['residual_stress'][2,:]).transpose()


x = np.array(configFile['X'][0,:])
y = np.array(configFile['X'][1,:])
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

MAX_phi_1x = max(phi_1x.flatten())
MAX_phi_1y = max(phi_1y.flatten())
MAX_phi_2x = max(phi_2x.flatten())
MAX_phi_2y = max(phi_2y.flatten())
MAX_phi_3x = max(phi_3x.flatten())
MAX_phi_3y = max(phi_3y.flatten())
MAX_phi_4x = max(phi_4x.flatten())
MAX_phi_4y = max(phi_4y.flatten())
MAX_phi_5x = max(phi_5x.flatten())
MAX_phi_5y = max(phi_5y.flatten())
MAX_phi_6x = max(phi_6x.flatten())
MAX_phi_6y = max(phi_6y.flatten())


print('max_x: ',MAX_x)
print('max_y: ',MAX_y)

MAX_p= 1 # estimated maximum pressure

# reduce the collocation points to 25k
colloc_limits1 = np.array([[-2.0,4.0],[0.0,2.0]])
colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
colloc_lhs1 = colloc_sample_lhs1(5000)
print('colloc_lhs1.shape',colloc_lhs1.shape)

colloc_limits2 = np.array([[4.0,10.0],[0.0,2.0]])
colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
colloc_lhs2 = colloc_sample_lhs2(5000)
print('colloc_lhs2.shape',colloc_lhs2.shape)

colloc_merged = np.vstack((colloc_lhs1,colloc_lhs2))
# remove points inside the cylinder
cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0],2)+np.power(colloc_merged[:,1],2),0.5),0.5)
colloc_merged = np.delete(colloc_merged,cylinder_inds,axis=0)
colloc_merged = np.vstack((colloc_merged,colloc_merged*np.array([1,-1])))
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
phi_1x_train = phi_1x/MAX_phi_1x
phi_1y_train = phi_1y/MAX_phi_1y
phi_2x_train = phi_2x/MAX_phi_2x
phi_2y_train = phi_2y/MAX_phi_2y
phi_3x_train = phi_3x/MAX_phi_3x
phi_3y_train = phi_3y/MAX_phi_3y
phi_4x_train = phi_4x/MAX_phi_4x
phi_4y_train = phi_4y/MAX_phi_4y
phi_5x_train = phi_5x/MAX_phi_5x
phi_5y_train = phi_5y/MAX_phi_5y
phi_6x_train = phi_6x/MAX_phi_6x
phi_6y_train = phi_6y/MAX_phi_6y


# the order here must be identical to inside the cost functions
O_train = np.hstack((A1.reshape(-1,1),A2.reshape(-1,1),A3.reshape(-1,1),A4.reshape(-1,1),A5.reshape(-1,1),A6.reshape(-1,1),(ux_train).reshape(-1,1),(uy_train).reshape(-1,1),phi_1x_train.reshape(-1,1),phi_1y_train.reshape(-1,1),phi_2x_train.reshape(-1,1),phi_2y_train.reshape(-1,1),phi_3x_train.reshape(-1,1),phi_3y_train.reshape(-1,1),phi_4x_train.reshape(-1,1),phi_4y_train.reshape(-1,1),phi_5x_train.reshape(-1,1),phi_5y_train.reshape(-1,1),phi_6x_train.reshape(-1,1),phi_6y_train.reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))

print('X_train.shape: ',X_train.shape)
print('O_train.shape: ',O_train.shape)

@tf.function
def net_f_cartesian(colloc_tensor):
    
    up = model(colloc_tensor)
    # knowns
    A1 = up[:,0] # these are less than 1 based on how the POD is normalized, so there is no need to scale
    A2 = up[:,1]
    A3 = up[:,2]
    A4 = up[:,3]
    A5 = up[:,4]
    A6 = up[:,5]
    ux = up[:,6]*MAX_ux
    uy = up[:,7]*MAX_uy
    phi_1x = up[:,8]*MAX_phi_1x
    phi_1y = up[:,9]*MAX_phi_1y
    phi_2x = up[:,10]*MAX_phi_2x
    phi_2y = up[:,11]*MAX_phi_2y
    phi_3x = up[:,12]*MAX_phi_3x
    phi_3y = up[:,13]*MAX_phi_3y
    phi_4x = up[:,14]*MAX_phi_4x
    phi_4y = up[:,15]*MAX_phi_4y
    phi_5x = up[:,16]*MAX_phi_5x
    phi_5y = up[:,17]*MAX_phi_5y
    phi_6x = up[:,18]*MAX_phi_6x
    phi_6y = up[:,19]*MAX_phi_6y
    uxppuxpp = up[:,20]*MAX_uxppuxpp
    uxppuypp = up[:,21]*MAX_uxppuypp
    uyppuypp = up[:,22]*MAX_uyppuypp
    # unknowns
    p = up[:,23]*MAX_p
    
    # compute the gradients of the quantities
    # gradients of a
    dA1 = tf.gradients(A1, colloc_tensor)[0]
    A1_x = dA1[:,0]/MAX_x
    A1_y = dA1[:,1]/MAX_y
    dA2 = tf.gradients(A2, colloc_tensor)[0]
    A2_x = dA2[:,0]/MAX_x
    A2_y = dA2[:,1]/MAX_y
    dA3 = tf.gradients(A3, colloc_tensor)[0]
    A3_x = dA3[:,0]/MAX_x
    A3_y = dA3[:,1]/MAX_y
    dA4 = tf.gradients(A4, colloc_tensor)[0]
    A4_x = dA4[:,0]/MAX_x
    A4_y = dA4[:,1]/MAX_y
    dA5 = tf.gradients(A5, colloc_tensor)[0]
    A5_x = dA5[:,0]/MAX_x
    A5_y = dA5[:,1]/MAX_y
    dA6 = tf.gradients(A6, colloc_tensor)[0]
    A6_x = dA6[:,0]/MAX_x
    A6_y = dA6[:,1]/MAX_y
    
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

    # mode gradients
    # phi_1x gradient
    dphi_1x = tf.gradients(phi_1x, colloc_tensor)[0]
    phi_1x_x = dphi_1x[:,0]/MAX_x
    phi_1x_y = dphi_1x[:,1]/MAX_y
    # phi_1y gradient
    dphi_1y = tf.gradients(phi_1y, colloc_tensor)[0]
    phi_1y_x = dphi_1y[:,0]/MAX_x
    phi_1y_y = dphi_1y[:,1]/MAX_y
    # phi_2x gradient
    dphi_2x = tf.gradients(phi_2x, colloc_tensor)[0]
    phi_2x_x = dphi_2x[:,0]/MAX_x
    phi_2x_y = dphi_2x[:,1]/MAX_y
    # phi_2y gradient
    dphi_2y = tf.gradients(phi_2y, colloc_tensor)[0]
    phi_2y_x = dphi_2y[:,0]/MAX_x
    phi_2y_y = dphi_2y[:,1]/MAX_y
    # phi_3x gradient
    dphi_3x = tf.gradients(phi_3x, colloc_tensor)[0]
    phi_3x_x = dphi_3x[:,0]/MAX_x
    phi_3x_y = dphi_3x[:,1]/MAX_y
    # phi_3y gradient
    dphi_3y = tf.gradients(phi_3y, colloc_tensor)[0]
    phi_3y_x = dphi_3y[:,0]/MAX_x
    phi_3y_y = dphi_3y[:,1]/MAX_y
    # phi_4x gradient
    dphi_4x = tf.gradients(phi_4x, colloc_tensor)[0]
    phi_4x_x = dphi_4x[:,0]/MAX_x
    phi_4x_y = dphi_4x[:,1]/MAX_y
    # phi_4y gradient
    dphi_4y = tf.gradients(phi_4y, colloc_tensor)[0]
    phi_4y_x = dphi_4y[:,0]/MAX_x
    phi_4y_y = dphi_4y[:,1]/MAX_y
    # phi_5x gradient
    dphi_5x = tf.gradients(phi_5x, colloc_tensor)[0]
    phi_5x_x = dphi_5x[:,0]/MAX_x
    phi_5x_y = dphi_5x[:,1]/MAX_y
    # phi_5y gradient
    dphi_5y = tf.gradients(phi_5y, colloc_tensor)[0]
    phi_5y_x = dphi_5y[:,0]/MAX_x
    phi_5y_y = dphi_5y[:,1]/MAX_y
    # phi_6x gradient
    dphi_6x = tf.gradients(phi_6x, colloc_tensor)[0]
    phi_6x_x = dphi_6x[:,0]/MAX_x
    phi_6x_y = dphi_6x[:,1]/MAX_y
    # phi_6y gradient
    dphi_6y = tf.gradients(phi_6y, colloc_tensor)[0]
    phi_6y_x = dphi_6y[:,0]/MAX_x
    phi_6y_y = dphi_6y[:,1]/MAX_y

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


    # modeled reynolds stress gradients
    uxmuxm_x = 2*A1*phi_1x*phi_1x_x + 2*A2*phi_2x*phi_2x_x + 2*A3*phi_3x*phi_3x_x + 2*A4*phi_4x*phi_4x_x + 2*A5*phi_5x*phi_5x_x + 2*A6*phi_6x*phi_6x_x
    uxmuym_x = A1*(phi_1x*phi_1y_x + phi_1y*phi_1x_x) + A2*(phi_2x*phi_2y_x + phi_2y*phi_2x_x) + A3*(phi_3x*phi_3y_x + phi_3y*phi_3x_x) + A4*(phi_4x*phi_4y_x + phi_4y*phi_4x_x) + A5*(phi_5x*phi_5y_x + phi_5y*phi_5x_x) + A6*(phi_6x*phi_6y_x + phi_6y*phi_6x_x)
    uxmuym_y = A1*(phi_1x*phi_1y_y + phi_1y*phi_1x_y) + A2*(phi_2x*phi_2y_y + phi_2y*phi_2x_y) + A4*(phi_3x*phi_3y_y + phi_3y*phi_3x_y) + A4*(phi_4x*phi_4y_y + phi_4y*phi_4x_y) + A5*(phi_5x*phi_5y_y + phi_5y*phi_5x_y) + A6*(phi_6x*phi_6y_y + phi_6y*phi_6x_y)
    uymuym_y = 2*A1*phi_1y*phi_1y_y + 2*A2*phi_2y*phi_2y_y + 2*A3*phi_3y*phi_3y_y + 2*A4*phi_4y*phi_4y_y + 2*A5*phi_5y*phi_5y_y + 2*A6*phi_6y*phi_6y_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxmuxm_x + uxmuym_y) + (uxppuxpp_x + uxppuypp_y) + p_x - (nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxmuym_x + uymuym_y) + (uxppuypp_x + uyppuypp_y) + p_y - (nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    # we want to impose that Ai is spatially constant, so impose the spatial derivatives as a loss function
    a_loss = tf.square(A1_x)+tf.square(A1_y)+tf.square(A2_x)+tf.square(A2_y)+tf.square(A3_x)+tf.square(A3_y)+tf.square(A4_x)+tf.square(A4_y)+tf.square(A5_x)+tf.square(A5_y)+tf.square(A6_x)+tf.square(A6_y)

    return f_x, f_y, f_mass, a_loss


# create NN
dense_nodes = 75
dense_layers = 10
if useGPU:
    tf_device_string = '/GPU:0'
else:
    tf_device_string = '/CPU:0'

if False:
    # tried to compute a non-linear network, but ran into trouble with gradients
    with tf.device(tf_device_string):
        input_layer = keras.Input(shape=(3,))
        batch_size = tf.shape(input_layer)[0]
        # the order of the split here is very important because the order of the 
        # input vectors determines the order of the derivatives in the physics loss
        xy = tf.slice(input_layer,(batch_size,0),(batch_size,2))
        a = tf.slice(input_layer,(batch_size,2),(batch_size,1))

        # build the xy layers
        dense_type = keras.layers.Dense(dense_nodes,activation='tanh')
        hidden_layer = dense_type(xy)
        for i in range(dense_layers-1):
            hidden_type = keras.layers.Dense(dense_nodes,activation='tanh')
            hidden_layer = hidden_type(hidden_layer)
        xy_output_type = keras.layers.Dense(12,activation='linear')
        xy_output =  xy_output_type(hidden_layer)
        # build the a layers
        a_dense_type = keras.layers.Dense(2,activation='linear')
        hidden_layer_a = a_dense_type(a)
        # concatinate
        # build the output
        # again the order here is important so that this matches up with the loss function wrappers
        output_layer = keras.layers.concatenate((hidden_layer_a,xy_output),1)
        model = keras.Model(inputs=input_layer,outputs=output_layer,name='functional_sequential')
        model.summary()
        keras.utils.plot_model(model,SLURM_TMPDIR+'model.png')

with tf.device(tf_device_string):
    model = keras.Sequential()
    model.add(keras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
    for i in range(dense_layers-1):
        model.add(keras.layers.Dense(dense_nodes, activation='tanh'))
    model.add(keras.layers.Dense(24,activation='linear'))
    model.summary()

# function wrapper, combine data and physics loss
def custom_loss_wrapper(colloc_tensor_f): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_A1 = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # A1
        data_loss_A2 = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # A2
        data_loss_A3 = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # A3
        data_loss_A4 = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # A4
        data_loss_A5 = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # A5
        data_loss_A6 = keras.losses.mean_squared_error(y_true[:,5], y_pred[:,5]) # A6
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,6], y_pred[:,6]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,7], y_pred[:,7]) # v 
        data_loss_phi_1x = keras.losses.mean_squared_error(y_true[:,8], y_pred[:,8]) # phi_1,x
        data_loss_phi_1y = keras.losses.mean_squared_error(y_true[:,9], y_pred[:,9]) # phi_1,y
        data_loss_phi_2x = keras.losses.mean_squared_error(y_true[:,10], y_pred[:,10]) # phi_2,x
        data_loss_phi_2y = keras.losses.mean_squared_error(y_true[:,11], y_pred[:,11]) # phi_2,y
        data_loss_phi_3x = keras.losses.mean_squared_error(y_true[:,12], y_pred[:,12]) # phi_3,x
        data_loss_phi_3y = keras.losses.mean_squared_error(y_true[:,13], y_pred[:,13]) # phi_3,y
        data_loss_phi_4x = keras.losses.mean_squared_error(y_true[:,14], y_pred[:,14]) # phi_4,x
        data_loss_phi_4y = keras.losses.mean_squared_error(y_true[:,15], y_pred[:,15]) # phi_4,y
        data_loss_phi_5x = keras.losses.mean_squared_error(y_true[:,16], y_pred[:,16]) # phi_5,x
        data_loss_phi_5y = keras.losses.mean_squared_error(y_true[:,17], y_pred[:,17]) # phi_5,y
        data_loss_phi_6x = keras.losses.mean_squared_error(y_true[:,18], y_pred[:,18]) # phi_6,x
        data_loss_phi_6y = keras.losses.mean_squared_error(y_true[:,19], y_pred[:,19]) # phi_6,y
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,20], y_pred[:,20]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,21], y_pred[:,21]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,22], y_pred[:,22]) # v''v''


        mx,my,mass,aloss = net_f_cartesian(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(mx))
        physical_loss2 = tf.reduce_mean(tf.square(my))
        physical_loss3 = tf.reduce_mean(tf.square(mass))
        physical_loss4 = tf.reduce_sum(aloss) # the components are already squared
                
        return data_loss_A1 + data_loss_A2 + data_loss_A3 + data_loss_A4 + data_loss_A5 + data_loss_A6 + data_loss_ux + data_loss_uy + data_loss_phi_1x + data_loss_phi_1y + data_loss_phi_2x + data_loss_phi_2y + data_loss_phi_3x + data_loss_phi_3y + data_loss_phi_4x + data_loss_phi_4y + data_loss_phi_5x + data_loss_phi_5y + data_loss_phi_6x + data_loss_phi_6y + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp + (23/3)*physical_loss1 + (23/3)*physical_loss2 + (23/3)*physical_loss3 + 1.0*physical_loss4 # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

    return custom_loss

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
checkpoint_files = get_filepaths_with_glob(SLURM_TMPDIR+'/output/'+job_name+'_output/',job_name+'_ep*.index')
if len(checkpoint_files)>0:
    files_epoch_number = np.zeros([len(checkpoint_files),1],dtype=np.uint)
    # if there are checkpoints, train based on the most recent checkpoint
    for f_indx in range(len(checkpoint_files)):
        re_result = re.search("ep[0-9]*",checkpoint_files[f_indx])
        files_epoch_number[f_indx]=int(checkpoint_files[f_indx][(re_result.start()+2):re_result.end()])
    epochs = np.uint(np.max(files_epoch_number))
    print(SLURM_TMPDIR+'/output/'+job_name+'_output/',job_name+'_ep'+str(epochs))
    model.load_weights(SLURM_TMPDIR+'/output/'+job_name+'_output/'+job_name+'_ep'+str(epochs))
else:
    # if not, we train from the beginning
    epochs = 0

# train the network
d_epochs = 1
X_train = tf.cast(X_train,dtype_train)
O_train = tf.cast(O_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs

if node_name ==LOCAL_NODE:
    # local node training loop, save every epoch for testing
    for e in range(10):
        shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
        temp_X_train = X_train[shuffle_inds,:]
        temp_Y_train = O_train[shuffle_inds,:]
        hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
        epochs = epochs+d_epochs
        model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))

else:
    # compute canada training loop; use time based training
    while True:
        shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
        temp_X_train = X_train[shuffle_inds,:]
        temp_Y_train = O_train[shuffle_inds,:]
        hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
        epochs = epochs+d_epochs
        if epochs>=50:
            keras.backend.set_value(model.optimizer.learning_rate, 0.005)
        if epochs>=100:
            keras.backend.set_value(model.optimizer.learning_rate, 0.001)
            
        if np.mod(epochs,10)==0:
            # save every 10th epoch
            model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))


        # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            # save the last epoch before exiting
            model.save_weights(save_loc+job_name+'_ep'+str(np.uint(epochs)))
            exit()
        last_epoch_time = datetime.now()
