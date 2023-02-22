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

tfkeras.backend.set_floatx('float64')
plt.rcParams['figure.figsize'] = [14, 7]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
#mpl.use('Qt5Agg')
mpl.use('TkAgg')

case = 'JFM'
save_loc = './tmp/checkpoint_swirl_JFM_all'
checkpoint_filepath = './tmp/checkpoint_swirl_JFM'

PLOT = True
# set constant seed to compare simulations with different hyper parameters
np.random.seed(1)
tf.random.set_seed(1)

# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# read the data
filename = 'C:\projects\pinns_galerkin_viv\jakob_jfm/PIV_für_Jakob/Q10_S62_baseline_mean.mat'
solutFile = scipy.io.loadmat(filename)


x_raw = np.array(solutFile['Y'])
x_raw /= 1000  # change from mm to m
r_0 = np.array(solutFile['X'])
r_raw = r_0 - 0.65  # align coordinate center
r_raw /= 1000  # change from mm to m

#rho_raw=np.array(solutFile['0']['PointData']['rho'])
ux_raw = np.array(solutFile['Vmean'])
ur_raw = np.array(solutFile['Umean'])
ut_raw = np.array(solutFile['Wmean'])



#temperature_raw=np.array(solutFile['0']['PointData']['T'])
# read the reynolds stressen data
filename = 'C:\projects\pinns_galerkin_viv\jakob_jfm/PIV_für_Jakob/Q10_S62_baseline_Reynolds.mat'
solutFile = scipy.io.loadmat(filename)
uxur_raw = np.array(solutFile['UV'])
urut_raw = np.array(solutFile['UW'])
uxut_raw = np.array(solutFile['VW'])

#temperature_raw=np.array(solutFile['0']['PointData']['T'])

if PLOT:
    fig = plt.figure()
    plt.subplot(4,1,1)
    plt.contourf(x_raw,r_raw,ux_raw, cmap=cm.jet)
    plt.colorbar()
    #plt.clim(-2,6)
    plt.subplot(4,1,2)
    plt.contourf(x_raw,r_raw,ur_raw, cmap=cm.jet)
    plt.colorbar()
    #plt.clim(-0.2,0.2)
    plt.subplot(4,1,3)
    plt.contourf(x_raw,r_raw,ut_raw, cmap=cm.jet)
    plt.colorbar()

    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.contourf(x_raw,r_raw,uxur_raw, cmap=cm.jet)
    plt.colorbar()
    #plt.clim(-2,6)
    plt.subplot(3,1,2)
    plt.contourf(x_raw,r_raw,urut_raw, cmap=cm.jet)
    plt.colorbar()
    #plt.clim(-0.2,0.2)
    plt.subplot(3,1,3)
    plt.contourf(x_raw,r_raw,uxut_raw, cmap=cm.jet)
    plt.colorbar()

# cut the data a bit
r_cut=[0,95]

x=x_raw
r=r_raw
ux=ux_raw
ur=ur_raw
ut=ut_raw

uxur=uxur_raw
urut=urut_raw
uxut=uxut_raw

x=x[r>=r_cut[0]]
ux=ux[r>=r_cut[0]]
ur=ur[r>=r_cut[0]]
ut=ut[r>=r_cut[0]]
uxur=uxur[r>=r_cut[0]]
urut=urut[r>=r_cut[0]]
uxut=uxut[r>=r_cut[0]]
r=r[r>=r_cut[0]]

x=x[r<=r_cut[1]]
ux=ux[r<=r_cut[1]]
ur=ur[r<=r_cut[1]]
ut=ut[r<=r_cut[1]]
uxur=uxur[r<=r_cut[1]]
urut=urut[r<=r_cut[1]]
uxut=uxut[r<=r_cut[1]]
r=r[r<=r_cut[1]]



# now we have the field we can train the PINN to learn the density!
x =x[~np.isnan(ux)]
r =r[~np.isnan(ux)]

#rho_raw=np.array(solutFile['0']['PointData']['rho'])
ur =ur[~np.isnan(ux)]
ux =ux[~np.isnan(ux)]
ut =ut[~np.isnan(ux)]
uxur=uxur[~np.isnan(ux)]
urut=urut[~np.isnan(ux)]
uxut=uxut[~np.isnan(ux)]


MAX_x = max(x.flatten())
MAX_r = max(r.flatten())
MAX_ux = max(ux.flatten())
MAX_ur = max(abs(ur.flatten()))
MAX_ut = max(abs(ut.flatten()))

MAX_nut= 0.01 # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
MAX_p= 20 # estimated maximum pressure



# collocation points are the points in the domain plus some points close to
# the reciculation bubble
lb_1 = np.array([0.0, 0])
ub_1 = np.array([0.1675, 0.6])
N_f_colloc_1 = 1000
f_colloc_train_1 = lb_1 + (ub_1-lb_1)*lhs(2,N_f_colloc_1)

lb_2 = np.array([0.17, 0.14])
ub_2 = np.array([0.4,  0.6])
N_f_colloc_2 = 1000
f_colloc_train_2 = lb_2 + (ub_2-lb_2)*lhs(2,N_f_colloc_2)

f_colloc_train_3 = np.vstack((x/MAX_x,r/MAX_r)).T
 
f_colloc_train = np.vstack((f_colloc_train_1,f_colloc_train_2,f_colloc_train_3))

# plt.subplot(3,1,3)
# plt.subplot(3,1,3)
# plt.scatter(f_colloc_train[:,0],f_colloc_train[:,1])



# normalize the training data:
x_train = x/MAX_x
r_train = r/MAX_r
ux_train = ux/MAX_ux
ur_train = ur/MAX_ur    
ut_train = ut/MAX_ut    

O_train = np.hstack(((ux_train).reshape(-1,1),(ur_train).reshape(-1,1))) # training data
X_train = np.hstack((x_train.reshape(-1,1),r_train.reshape(-1,1) ))


# b.c. points for the teta velocity profile 
# b.c. for ut
x_bc1 = np.linspace(0,0.3,10)
r_bc1 = np.linspace(1,1,10)
ut_bc1   =   np.linspace(0,0,10)
BC_points1 = np.hstack((x_bc1.reshape(-1,1),r_bc1.reshape(-1,1) ))


# b.c. for the pressure
x_bc_p = np.linspace(0,0,1)
r_bc_p = np.linspace(1,1,1)
p_bc   =   np.linspace(0,0,1)
BC_points2 = np.hstack((x_bc_p.reshape(-1,1),r_bc_p.reshape(-1,1) ))


# b.c. for ut
second_sensor=~np.add(x/MAX_x>0.508,x/MAX_x<0.5)
x_bc2 = x[second_sensor]/MAX_x
r_bc2 = r[second_sensor]/MAX_r
ut_bc2   =   ut[second_sensor]/MAX_ut
BC_points3 = np.hstack((x_bc2.reshape(-1,1),r_bc2.reshape(-1,1) ))


# create NN
with tf.device('/GPU:0'):
    model = tfkeras.Sequential()
    model.add(tfkeras.layers.Dense(30, activation='tanh', input_shape=(2,)))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(30, activation='tanh'))
    model.add(tfkeras.layers.Dense(5,activation='linear'))
    model.summary()

# define cost function
@tf.function
def net_f(colloc_tensor):
    
    up = model(colloc_tensor)
    r  = colloc_tensor[:,1]*MAX_r
    
    ux = up[:,0]*MAX_ux
    ur = up[:,1]*MAX_ur
    
    ut = up[:,2]*MAX_ut
    nut = up[:,3]*MAX_nut
    p = up[:,4]*MAX_p
    
    
    
    # ux gradient
    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/MAX_x
    ux_r = dux[:,1]/MAX_r
     # second derivatives
    dux_dx = tf.gradients(ux_x, colloc_tensor)[0]
    ux_xx = dux_dx[:,0]/MAX_x
    #ux_xr = dux_dx[:,1]/MAX_r
    dux_dr = tf.gradients(ux_r, colloc_tensor)[0]
    #ux_rx = dux_dr[:,0]/MAX_x
    ux_rr = dux_dr[:,1]/MAX_r
    
    # ur gradient
    dur = tf.gradients(ur, colloc_tensor)[0]
    ur_x = dur[:,0]/MAX_x
    ur_r = dur[:,1]/MAX_r
    # second derivatives
    dur_dx = tf.gradients(ur_x, colloc_tensor)[0]
    ur_xx = dur_dx[:,0]/MAX_x
    #ur_xr = dur_dx[:,1]/MAX_r
    dur_dr = tf.gradients(ur_r, colloc_tensor)[0]
    #ur_rx = dur_dr[:,0]/MAX_x
    ur_rr = dur_dr[:,1]/MAX_r
    
    
    # ut gradient
    dut = tf.gradients(ut, colloc_tensor)[0]
    ut_x = dut[:,0]/MAX_x
    ut_r = dut[:,1]/MAX_r
    # second derivatives
    dut_dx = tf.gradients(ut_x, colloc_tensor)[0]
    ut_xx = dut_dx[:,0]/MAX_x
    #ut_xr = dut_dx[:,1]/MAX_r
    dut_dr = tf.gradients(ut_r, colloc_tensor)[0]
    #ut_rx = dut_dr[:,0]/MAX_x
    ut_rr = dut_dr[:,1]/MAX_r
    
    # p gradient
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_r = dp[:,1]/MAX_r
    
    dnut = tf.gradients(nut, colloc_tensor)[0]
    nut_x = dnut[:,0]/MAX_x
    nut_r = dnut[:,1]/MAX_r
    
    # set of RANS equations without pressure
    f_moment_r = r*ux*ur_x + r*ur*ur_r - ut*ut + r*p_r \
                 - nut*(r*ur_xx + ur_r + r*ur_rr - ur/r)   \
                 - 2*r*nut_r*ur_r  - nut_x*r*(ur_x+ux_r)
    
    f_moment_x = r*ux*ux_x + r*ur*ux_r         + r*p_x \
                 - nut*(r*ux_xx + ux_r + r*ux_rr)      \
                 - 2*r*nut_x*ux_x  - nut_r*r*(ur_x+ux_r)
    
    f_moment_t = r*ux*ut_x + r*ur*ut_r + ur*ut         \
                 - nut*(r*ut_xx + ut_r + r*ut_rr - ut/r)   \
                 - r*nut_x*ut_x    - nut_r*r*(ut_r-ut/r)
    
    f_mass     = r*ux_x + r*ur_r + ur
    
    return f_moment_t, f_moment_x, f_moment_r, f_mass


# function for b.c
def BC_fun(colloc_tensor1,BC,var):
    up1 = model(colloc_tensor1)
    #rho_bc_pinn=up1[:,2] # no rescaling since rho_bc is normalised
    f1  = tfkeras.losses.mean_squared_error(up1[:,var], np.squeeze(BC))
    return f1

# function wrapper, combine data and physics loss
def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def custom_loss(y_true, y_pred):
        
        data_loss1 = tfkeras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss2 = tfkeras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        #data_loss3 = tfkeras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # p        
        
        mt,mx,mr,mass = net_f(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(mt))
        physical_loss2 = tf.reduce_mean(tf.square(mx))
        physical_loss3 = tf.reduce_mean(tf.square(mr))
        physical_loss4 = tf.reduce_mean(tf.square(mass))
        
        #boundary loss
        f_boundary_t1   = BC_fun(BCs,ut_bc1,2)
        f_boundary_p = BC_fun(BCs_p,p_bc,4)
        f_boundary_t2 = BC_fun(BCs_t,ut_bc2,2)
        
        return data_loss1 + data_loss2  + 1*physical_loss1 + 1*physical_loss2 + 1*physical_loss3 + 1*physical_loss4 + f_boundary_t1+ f_boundary_t2 + 1*f_boundary_p

    return custom_loss

model.compile(optimizer=tfkeras.optimizers.Adam(learning_rate=0.01), loss = custom_loss_wrapper(f_colloc_train,BC_points1,BC_points2,BC_points3))

model_checkpoint_callback = tfkeras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tfkeras.callbacks.EarlyStopping(monitor='loss', patience=500)

# train network with different learning rates
hist = model.fit(X_train, O_train, batch_size=32, epochs=5, callbacks=[early_stop_callback,model_checkpoint_callback])


tfkeras.backend.set_value(model.optimizer.learning_rate, 0.005)
model.load_weights(checkpoint_filepath)
hist = model.fit(X_train, O_train, batch_size=32, epochs=10, callbacks=[early_stop_callback,model_checkpoint_callback])


tfkeras.backend.set_value(model.optimizer.learning_rate, 0.001)
model.load_weights(checkpoint_filepath)
hist = model.fit(X_train, O_train, batch_size=64, epochs=25, callbacks=[early_stop_callback,model_checkpoint_callback])


model.load_weights(checkpoint_filepath)
tfkeras.backend.set_value(model.optimizer.learning_rate, 0.0001)
hist = model.fit(X_train, O_train, batch_size=128, epochs=50, callbacks=[early_stop_callback,model_checkpoint_callback])


model.load_weights(checkpoint_filepath)
tfkeras.backend.set_value(model.optimizer.learning_rate, 0.00001)
hist = model.fit(X_train, O_train, batch_size=256, epochs=50, callbacks=[early_stop_callback,model_checkpoint_callback])


model.load_weights(checkpoint_filepath)
tfkeras.backend.set_value(model.optimizer.learning_rate, 0.000001)
hist = model.fit(X_train, O_train, batch_size=512, epochs=100, callbacks=[early_stop_callback,model_checkpoint_callback])
model.load_weights(checkpoint_filepath)



loss_history = hist.history['loss']
#%matplotlib inline
plt.figure()
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()


from LBFGS_example import function_factory
import tensorflow_probability as tfp

# continue with LBFGS steps
func = function_factory(model, custom_loss_wrapper(f_colloc_train,BC_points1,BC_points2,BC_points3), X_train, O_train)

# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

# train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=5000)

# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
func.assign_new_model_parameters(results.position)

# save the model:
model.save(save_loc)




