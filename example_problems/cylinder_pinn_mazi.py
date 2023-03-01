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

tfkeras.backend.set_floatx('float64')
plt.rcParams['figure.figsize'] = [14, 7]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
#mpl.use('Qt5Agg')
mpl.use('TkAgg')

case = 'JFM'
save_loc = './data/mazi_fixed/tmp/'
checkpoint_filepath = './data/mazi_fixed/tmp/checkpoint'

PLOT = True
# set constant seed to compare simulations with different hyper parameters
np.random.seed(1)
tf.random.set_seed(1)

# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)


# read the data
meanfilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/meanField.mat'
meanFieldFile = h5py.File(meanfilename,'r')
configfilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/configuration.mat'
configFile = h5py.File(configfilename,'r')


ux = np.array(meanFieldFile['meanField'][0,:])
uy = np.array(meanFieldFile['meanField'][1,:])

x = np.array(configFile['X'][0,:])
y = np.array(configFile['X'][1,:])
d = np.array(configFile['cylinderDiameter'])
print('u.shape: ',ux.shape)
print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)



MAX_x = max(x.flatten())
MAX_y = max(y.flatten())
MAX_ux = max(ux.flatten())
MAX_uy = max(uy.flatten())
MIN_x = min(x.flatten())
MIN_y = min(y.flatten())
MIN_ux = min(ux.flatten())
MIN_uy = min(uy.flatten())

MAX_nut= 0.01 # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
MAX_p= 20 # estimated maximum pressure


# reduce the collocation points to 25k
colloc_rand = np.random.choice(x.size,2500);
x_colloc = x[colloc_rand]
y_colloc = x[colloc_rand]

f_colloc_train = np.vstack((x_colloc/MAX_x,y_colloc/MAX_y)).T


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


O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1))) # training data
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1) ))




# b.c. for the pressure
x_bc_p = np.linspace(MIN_x,MIN_x,1)
y_bc_p = np.linspace(MAX_y,MAX_y,1)
p_bc   =   0.02983052962603#np.linspace(0,0,1)
BC_points2 = np.hstack((x_bc_p.reshape(-1,1),y_bc_p.reshape(-1,1) ))




@tf.function
def net_f_cartesian(colloc_tensor):
    
    up = model(colloc_tensor)
    
    ux    = up[:,0]*MAX_ux
    uy    = up[:,1]*MAX_uy
    p     = up[:,2]*MAX_p
    nut   = pow(up[:,3],2)*MAX_nut
    
    
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
    

    # gradient nut
    dnut = tf.gradients(nut, colloc_tensor)[0]
    nut_x = dnut[:,0]/MAX_x
    nut_y = dnut[:,1]/MAX_y
    
    f_x = (ux*ux_x + uy*ux_y) + p_x - (nut)*(ux_xx+ux_yy) - 2*nut_x*ux_x - nut_y*(ux_y+uy_x) #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + p_y - (nut)*(uy_xx+uy_yy) - 2*nut_y*uy_y - nut_x*(ux_y+uy_x) #+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)

    # the original momentum equations from Jakob
    #f_x = (ux*ux_x + uy*ux_y) + p_x - (nu_mol + nut)*(ux_xx+ux_yy) - 2*nut_x*ux_x - nut_y*(ux_y+uy_x) #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    #f_y = (ux*uy_x + uy*uy_y) + p_y - (nu_mol + nut)*(uy_xx+uy_yy) - 2*nut_y*uy_y - nut_x*(ux_y+uy_x) #+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    

    f_mass = ux_x + uy_y

    return f_x, f_y, f_mass

# create NN
dense_nodes = 30
dense_layers = 10
model_structure_string = 'dense30x10_'
with tf.device('/GPU:0'):
    model = tfkeras.Sequential()
    model.add(tfkeras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
    for i in range(dense_layers-1):
        model.add(tfkeras.layers.Dense(dense_nodes, activation='tanh'))
    model.add(tfkeras.layers.Dense(4,activation='linear'))
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
        #data_loss3 = tfkeras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # p        
        
        mx,my,mass = net_f_cartesian(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(mx))
        physical_loss2 = tf.reduce_mean(tf.square(my))
        physical_loss3 = tf.reduce_mean(tf.square(mass))
        
        #boundary loss
        #f_boundary_t1   = BC_fun(BCs,ut_bc1,2)
        f_boundary_p = BC_fun(BCs_p,p_bc,2)
        #f_boundary_t2 = BC_fun(BCs_t,ut_bc2,2)
        
        return data_loss1 + data_loss2  + 1*physical_loss1 + 1*physical_loss2 + 1*physical_loss3 + 0*f_boundary_p # + f_boundary_t1+ f_boundary_t2 

    return custom_loss

model.compile(optimizer=tfkeras.optimizers.Adam(learning_rate=0.01), loss = custom_loss_wrapper(f_colloc_train,BC_points2)) #(...,BC_points1,...,BC_points3)

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

if False: 
    # train network with different learning rates
    hist = model.fit(X_train, O_train, batch_size=32, epochs=5, callbacks=[early_stop_callback,model_checkpoint_callback])
    model.save_weights(save_loc+model_structure_string+'b32_ep5_st1')
    pred = model.predict(X_train,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b32_ep5_st1_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()


    tfkeras.backend.set_value(model.optimizer.learning_rate, 0.005)
    model.load_weights(checkpoint_filepath)
    hist = model.fit(X_train, O_train, batch_size=64, epochs=10, callbacks=[early_stop_callback,model_checkpoint_callback])
    model.save_weights(save_loc+model_structure_string+'b64_ep10_st2')
    pred = model.predict(X_train,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b64_ep10_st2_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()






    tfkeras.backend.set_value(model.optimizer.learning_rate, 0.001)
    hist = model.fit(X_train, O_train, batch_size=128, epochs=25, callbacks=[early_stop_callback,model_checkpoint_callback])
    model.save_weights(save_loc+model_structure_string+'b128_ep25_st3')
    pred = model.predict(X_train,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b128_ep25_st3_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()




    model.load_weights(checkpoint_filepath)
    tfkeras.backend.set_value(model.optimizer.learning_rate, 0.0001)
    hist = model.fit(X_train, O_train, batch_size=128, epochs=50, callbacks=[early_stop_callback,model_checkpoint_callback])
    model.save_weights(save_loc+model_structure_string+'b128_ep50_st4')
    pred = model.predict(X_train,batch_size=512)
    h5f = h5py.File(save_loc+model_structure_string+'b128_ep50_st4_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()

# this time we randomly shuffle the order of X and O
rng = np.random.default_rng(seed=1)
shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))

model.load_weights(checkpoint_filepath)
tfkeras.backend.set_value(model.optimizer.learning_rate, 0.00001)
temp_X_train = X_train[shuffle_inds,:]
temp_Y_train = O_train[shuffle_inds,:]
print(X_train.shape)
print(temp_X_train.shape)
hist = model.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=32, epochs=50, callbacks=[early_stop_callback,model_checkpoint_callback])
model.save_weights(save_loc+model_structure_string+'b32_ep50_st5')
pred = model.predict(X_train,batch_size=512)
h5f = h5py.File(save_loc+model_structure_string+'b32_ep50_st5_pred.mat','w')
h5f.create_dataset('pred',data=pred)
h5f.close()


exit()

model.load_weights(checkpoint_filepath)
tfkeras.backend.set_value(model.optimizer.learning_rate, 0.000001)
hist = model.fit(X_train, O_train, batch_size=512, epochs=100, callbacks=[early_stop_callback,model_checkpoint_callback])
model.save_weights(save_loc+model_structure_string+'b512_ep100_st6')
model.load_weights(checkpoint_filepath)
pred = model.predict(X_train,batch_size=512)
h5f = h5py.File(save_loc+model_structure_string+'b512_ep100_st6_pred.mat','w')
h5f.create_dataset('pred',data=pred)
h5f.close()



loss_history = hist.history['loss']
#%matplotlib inline
#plt.figure()
#plt.plot(loss_history)
#plt.title("Loss History")
#plt.xlabel("Epoch")
#plt.ylabel("Loss")

#plt.show()

# save the model:
model.save(save_loc)




