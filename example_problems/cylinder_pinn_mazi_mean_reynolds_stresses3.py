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
import tensorflow.keras as tfkeras
from mpl_toolkits.mplot3d import Axes3D
import h5py
import time
import pickle
import os
import smt
from smt.sampling_methods import LHS
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import h5py
from datetime import datetime

tfkeras.backend.set_floatx('float64')
dtype_train = tf.float64
plt.rcParams['figure.figsize'] = [14, 7]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
#mpl.use('Qt5Agg')
mpl.use('TkAgg')

case = 'JFM'
start_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
save_loc = './mazi_fixed/'+start_timestamp+'_tmp/'
checkpoint_filepath = '/mnt/c/projects/pinns_galerkin_viv/data/mazi_fixed/'+start_timestamp+'_tmp/checkpoint'

PLOT = False
# set constant seed to compare simulations with different hyper parameters


# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)
useGPU=True
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
base_dir = './mazi_fixed/'
meanFieldFile = h5py.File(base_dir+'meanField.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStresses.mat','r')


ux = np.array(meanFieldFile['meanField'][0,:]).transpose()
uy = np.array(meanFieldFile['meanField'][1,:]).transpose()

uxpuxp = np.array(reynoldsStressFile['reynoldsStresses'][0,:]).transpose()
uxpuyp = np.array(reynoldsStressFile['reynoldsStresses'][1,:]).transpose()
uypuyp = np.array(reynoldsStressFile['reynoldsStresses'][2,:]).transpose()

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
MAX_uxpuxp = max(uxpuxp.flatten())
MAX_uxpuyp = max(uxpuyp.flatten())
MAX_uypuyp = max(uypuyp.flatten())

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

if False:
    plt.figure(1)
    plt.scatter(colloc_merged[:,0],colloc_merged[:,1])
    plt.show()

f_colloc_train = colloc_merged*np.array([1/MAX_x,1/MAX_y])




if False:
    fig = plt.figure(1)
    ax1 = plt.subplot(2,1,1)
    plt.contourf(x,y,ux,20, cmap=cm.jet,zorder=1)
    plt.colorbar()
    plt.scatter(x_colloc,y_colloc,0.001,'k',zorder=2)
    
    #plt.clim(-2,6)
    plt.subplot(2,1,2)
    plt.contourf(x,y,uy,20, cmap=cm.jet,zorder=1)
    plt.colorbar()
    plt.scatter(x_colloc,y_colloc,0.001,'k',zorder=2)
    
    #plt.clim(-0.2,0.2)
    plt.show()



# normalize the training data:
x_train = x/MAX_x
y_train = y/MAX_y
ux_train = ux/MAX_ux
uy_train = uy/MAX_uy
uxpuxp_train = uxpuxp/MAX_uxpuxp
uxpuyp_train = uxpuyp/MAX_uxpuyp
uypuyp_train = uypuyp/MAX_uypuyp

# copy the points before reducing the size 
x_all = x_train
y_all = y_train
X_all = np.hstack((x_all.reshape(-1,1),y_all.reshape(-1,1) ))

# reduce the number of points for faster training
if False:
    train_points = np.random.choice(x.size,30000);
    x_train=x_train[train_points]
    y_train=y_train[train_points]
    ux_train=ux_train[train_points]
    uy_train=uy_train[train_points]


O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxpuxp_train).reshape(-1,1),(uxpuyp_train).reshape(-1,1),(uypuyp_train).reshape(-1,1),)) # training data
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1) ))




# b.c. for the pressure
x_bc_p = np.linspace(MIN_x,MIN_x,1)
y_bc_p = np.linspace(MAX_y,MAX_y,1)
p_bc   =   0.02983052962603#np.linspace(0,0,1)
BC_points2 = np.hstack((x_bc_p.reshape(-1,1),y_bc_p.reshape(-1,1)))


@tf.function
def net_f_cartesian(colloc_tensor):
    
    up = model(colloc_tensor)
    
    ux    = up[:,0]*MAX_ux
    uy    = up[:,1]*MAX_uy
    p     = up[:,5]*MAX_p
    uxpuxp = up[:,2]*MAX_uxpuxp
    uxpuyp = up[:,3]*MAX_uxpuyp
    uypuyp = up[:,4]*MAX_uypuyp
    #nut   = pow(up[:,3],2)*MAX_nut
    
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
       
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y
    p_xx = tf.gradients(p_x,colloc_tensor)[0][:,0]/MAX_x
    p_yy = tf.gradients(p_y,colloc_tensor)[0][:,1]/MAX_y

    # gradient reynolds stresses
    uxpuxp_x = tf.gradients(uxpuxp, colloc_tensor)[0][:,0]/MAX_x
    uxpuxp_xx = tf.gradients(uxpuxp_x, colloc_tensor)[0][:,0]/MAX_x
    duxpuyp = tf.gradients(uxpuyp, colloc_tensor)[0]
    uxpuyp_x = duxpuyp[:,0]/MAX_x
    uxpuyp_y = duxpuyp[:,1]/MAX_y
    uxpuyp_xy = tf.gradients(uxpuyp_x,colloc_tensor)[0][:,1]/MAX_y
    uypuyp_y = tf.gradients(uypuyp, colloc_tensor)[0][:,1]/MAX_y
    uypuyp_yy = tf.gradients(uypuyp_y, colloc_tensor)[0][:,1]/MAX_y

    f_x = (ux*ux_x + uy*ux_y) + p_x - (nu_mol)*(ux_xx+ux_yy) + uxpuxp_x + uxpuyp_y #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + p_y - (nu_mol)*(uy_xx+uy_yy) + uxpuyp_x + uypuyp_y#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)

    f_mass = ux_x + uy_y
    f_p = p_xx + p_yy + tf.math.pow(ux_x,tf.constant(2.0,dtype=dtype_train)) + 2*ux_y*uy_x + tf.math.pow(uy_y,tf.constant(2.0,dtype=dtype_train))+uxpuxp_xx+2*uxpuyp_xy+uypuyp_yy

    return f_x, f_y, f_mass, f_p


# create NN
dense_nodes = 30
dense_layers = 10
model_structure_string = 'dense30x10_'
with tf.device('/GPU:0'):
    model = tfkeras.Sequential()
    model.add(tfkeras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
    for i in range(dense_layers-1):
        model.add(tfkeras.layers.Dense(dense_nodes, activation='tanh'))
    model.add(tfkeras.layers.Dense(6,activation='linear'))
    model.summary()


# function for b.c
def BC_fun(colloc_tensor1,BC,var):
    up1 = model(colloc_tensor1)
    #rho_bc_pinn=up1[:,2] # no rescaling since rho_bc is normalised
    f1  = tfkeras.losses.mean_squared_error(up1[:,var], np.squeeze(BC))
    return f1

# function wrapper, combine data and physics loss
def custom_loss_wrapper(colloc_tensor_f,BCs_p): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        
        data_loss1 = tfkeras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss2 = tfkeras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss4 = tfkeras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u'u'   
        data_loss5 = tfkeras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u'v'
        data_loss6 = tfkeras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v'v'

        mx,my,mass,mp = net_f_cartesian(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(mx))
        physical_loss2 = tf.reduce_mean(tf.square(my))
        physical_loss3 = tf.reduce_mean(tf.square(mass))
        physical_loss4 = tf.reduce_mean(tf.square(mp))
        
        #boundary loss
        #f_boundary_t1   = BC_fun(BCs,ut_bc1,2)
        #f_boundary_p = BC_fun(BCs_p,p_bc,2)
        #f_boundary_t2 = BC_fun(BCs_t,ut_bc2,2)
        
        return data_loss1 + data_loss2 + data_loss4 + data_loss5 + data_loss6  + 1*physical_loss1 + 1*physical_loss2 + 1*physical_loss3 + 1*physical_loss4 # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

    return custom_loss

model.compile(optimizer=tfkeras.optimizers.SGD(learning_rate=0.01), loss = custom_loss_wrapper(tf.cast(f_colloc_train,dtype_train),tf.cast(BC_points2,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)

model_checkpoint_callback = tfkeras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tfkeras.callbacks.EarlyStopping(monitor='loss', patience=500)

def plot_pred(x,y,ux,uy,X_train,pred):
    fig = plt.figure(1)
    ax1 = plt.subplot(4,1,1)
    plt.contourf(x,y,ux,20, cmap=cm.jet,zorder=1)
    plt.colorbar()
    #plt.scatter(x_colloc,y_colloc,0.001,'k',zorder=2)
    
    #plt.clim(-2,6)
    plt.subplot(4,1,2)
    plt.contourf(x,y,uy,20, cmap=cm.jet,zorder=1)
    plt.colorbar()
   # plt.scatter(x_colloc,y_colloc,0.001,'k',zorder=2)
    plt.subplot(4,1,3)
    plt.contourf(np.reshape(X_train[:,0],x.shape),np.reshape(X_train[:,1],x.shape),np.reshape(pred[:,0],x.shape),20, cmap=cm.jet,zorder=1)
    plt.colorbar()   
    plt.subplot(4,1,4)
    plt.contourf(np.reshape(X_train[:,0],x.shape),np.reshape(X_train[:,1],x.shape),np.reshape(pred[:,1],x.shape),20, cmap=cm.jet,zorder=1)
    plt.colorbar()
    #plt.clim(-0.2,0.2)
    plt.show()
    return



# this time we randomly shuffle the order of X and O
rng = np.random.default_rng()


save_loc = base_dir+'20230404_test/'

if False:
    model.load_weights(base_dir+'20230328_reynolds_stress/dense50x10_b32_ep250_st2')
    #pred = model.predict(X_all,batch_size=512)
    #h5f = h5py.File('./data/mazi_fixed/20230302_4eq_noConstr/dense30x10_b32_ep500_st3_pred.mat','w')
    #h5f.create_dataset('pred',data=pred)
    #h5f.close()



d_epochs = 1
epochs = 0
stage = 1
X_train = tf.cast(X_train,dtype_train)
O_train = tf.cast(O_train,dtype_train)

for pqr in range(1):
    shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
    temp_X_train = X_train[shuffle_inds,:]
    temp_Y_train = O_train[shuffle_inds,:]
    hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
    epochs = epochs+d_epochs
    model.save_weights(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage))
    pred = model.predict(X_all,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage)+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()

exit()
d_epochs = 50
stage = 2

tfkeras.backend.set_value(model.optimizer.learning_rate, 0.005)
shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
temp_X_train = X_train[shuffle_inds,:]
temp_Y_train = O_train[shuffle_inds,:]
hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
epochs = epochs+d_epochs
model.save_weights(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage))
pred = model.predict(X_all,batch_size=512)
h5f = h5py.File(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage)+'_pred.mat','w')
h5f.create_dataset('pred',data=pred)
h5f.close()

d_epochs = 50
for pqr in range(10):
    shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
    temp_X_train = X_train[shuffle_inds,:]
    temp_Y_train = O_train[shuffle_inds,:]
    hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=d_epochs, callbacks=[early_stop_callback,model_checkpoint_callback])
    epochs = epochs+d_epochs
    model.save_weights(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage))
    pred = model.predict(X_all,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b32_ep'+str(epochs)+'_st'+str(stage)+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()
