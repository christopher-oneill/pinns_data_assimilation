

import tensorflow as tf
import tensorflow.keras as keras
import h5py
import tensorflow_probability as tfp

keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')


from pinns_galerkin_viv.lib.LBFGS_example import function_factory
from pinns_galerkin_viv.lib.layers import ResidualLayer
from pinns_galerkin_viv.lib.layers import FourierResidualLayer64
from pinns_galerkin_viv.lib.layers import ProductResidualLayer64
from pinns_galerkin_viv.lib.layers import CubicFourierProductBlock64
from pinns_galerkin_viv.lib.layers import QuarticFourierProductBlock64


from pinns_galerkin_viv.lib.downsample import compute_downsample_inds


def plot_err(epoch,model_sines):
    global X_grid
    global Y_grid
    global i_test
    global MAX_o_train
    global phi_xr_test_grid
    pred_test = model_sines.predict(i_test[:],batch_size=1000)
    pred_test_grid = np.reshape(pred_test,X_grid.shape)
    err_test = phi_xr_test_grid/MAX_o_train-pred_test_grid
    plot.figure(epoch)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,phi_xr_test_grid/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.show(block=False)
    plot.pause(0.01)

plot.ion()


HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
fourierModeFile = h5py.File(base_dir+'fourier_data_DFT.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')

global X_grid
global Y_grid

x = np.array(configFile['X_vec'][0,:])
x_test = x
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
y_test = y

mode_number=5

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

print(phi_xr.shape)
print(x.shape)
global MAX_o_train
global phi_xr_test_grid

ux_grid = np.reshape(ux,X_grid.shape)
phi_xr_test_grid = np.reshape(phi_xr,X_grid.shape)
MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)

x_test = X_grid/MAX_x
y_test = Y_grid/MAX_y

global i_test

i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))

# check the frequency limit of the reference frequency
#plot.figure(1)
#plot.scatter(i_train,o_train)
#plot.scatter(i_train,wave_ref)
#plot.show()
#exit()
supersample_factor = 1
# if we are downsampling and then upsampling, downsample the source data
if supersample_factor>1:
    n_x = np.array(configFile['x_grid']).size
    n_y = np.array(configFile['y_grid']).size
    downsample_inds,n_x_d,n_y_d = compute_downsample_inds(supersample_factor,n_x,n_y)
    x = x[downsample_inds]
    y = y[downsample_inds]
    phi_xr = phi_xr[downsample_inds]
else:
    n_x_d = X_grid.shape[0]
    n_y_d = X_grid.shape[1]

phi_xr_train_grid  = np.reshape(phi_xr,(n_x_d,n_y_d))

x_train = x/MAX_x
y_train = y/MAX_y

X_train_grid = np.reshape(x_train,(n_x_d,n_y_d))
Y_train_grid = np.reshape(y_train,(n_x_d,n_y_d))

i_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
print(i_train.shape)

o_train = phi_xr.reshape(-1,1)
print(o_train.shape)
#o_train = ux_grid[:,100]
MAX_o_train = np.max(o_train)
o_train = o_train/MAX_o_train



def network_loss(y_true,y_pred):
    data_loss = tf.reduce_mean(tf.square(y_true-y_pred)) # u 
    return data_loss


if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1, activation='linear', input_shape=(1,)))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 
if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(FourierLayer(100))
        model_sines.add(keras.layers.Dense(100,activation='linear'))
        model_sines.add(keras.layers.Dense(100,activation='linear'))
        model_sines.add(keras.layers.Dense(100,activation='linear'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 
if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(FourierLayer(100))
        model_sines.add(keras.layers.Dense(100,activation='tanh'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 
if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(FourierLayer(50))
        model_sines.add(resBlock2(50))
        model_sines.add(resBlock2(50))
        model_sines.add(resBlock2(50))
        model_sines.add(resBlock2(50)) 
        model_sines.add(resBlock2(50))
        #model_sines.add(resBlock2(50))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 
if False:
    nodes = 30
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(nodes,activation='linear',input_shape=(2,)))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))      
        model_sines.add(Fourier2ResidualLayer64(nodes))
        model_sines.add(Product2ResidualLayer64(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(Product2ResidualLayer64(nodes))
        model_sines.add(ResidualLayer(nodes))   
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes)) 
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes)) 
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes)) 
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))  
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 

if True:
    nodes = 40
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(nodes,activation='linear',input_shape=(2,)))
        for k in range(3):
            model_sines.add(ResidualLayer(nodes)) 
        model_sines.add(CubicFourierProductBlock64(20))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=tf.losses.mean_absolute_error,jit_compile=False) 

if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1, activation='linear', input_shape=(1,)))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(50,activation='tanh'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 


shuffle_inds = np.array(range((i_train).shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

i_train_shuffle = tf.cast((i_train[shuffle_inds,:])[0,:,:],tf.float64)
o_train_shuffle = tf.cast((o_train[shuffle_inds])[0,:],tf.float64)
print(i_train_shuffle.shape)
print(o_train_shuffle.shape)

LBFGS_steps = 333
LBFGS_epochs = 3*LBFGS_steps

epochs = 0

if True:
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-3)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=30)
    plot_err(500,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=30)
    plot_err(1000,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=30)
    plot_err(1500,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=30)
    plot_err(2000,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-7)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=30)
    plot_err(2500,model_sines)
    pred =  model_sines.predict(i_train,batch_size=1000)

epochs=2500
if True:
    for L_iter in range(10):

        func = function_factory(model_sines, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
        init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        pred =  model_sines.predict(i_train,batch_size=1000)
        epochs = epochs + LBFGS_epochs
        plot_err(epochs,model_sines)



pred_train_grid = np.reshape(pred,(n_x_d,n_y_d))
err_train = phi_xr_train_grid/MAX_o_train - pred_train_grid

pred_test = model_sines.predict(i_test[:],batch_size=1000)
pred_test_grid = np.reshape(pred_test,X_grid.shape)
err_test = phi_xr_test_grid/MAX_o_train-pred_test_grid

print(y.shape)
print(x.shape)
print(pred.shape)

plot.figure(2)
plot.title('Full Resolution')
plot.subplot(3,1,1)
plot.contourf(X_grid,Y_grid,phi_xr_test_grid/MAX_o_train,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,2)
plot.contourf(X_grid,Y_grid,pred_test_grid,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,3)
plot.contourf(X_grid,Y_grid,err_test,levels=21)
plot.set_cmap('bwr')
plot.colorbar()



plot.figure(3)
plot.title('S=8')
plot.subplot(3,1,1)
plot.contourf(X_train_grid,Y_train_grid,phi_xr_train_grid/MAX_o_train,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,2)
plot.contourf(X_train_grid,Y_train_grid,pred_train_grid,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,3)
plot.contourf(X_train_grid,Y_train_grid,err_train,levels=21)
plot.set_cmap('bwr')
plot.colorbar()

plot.show(block=True)

