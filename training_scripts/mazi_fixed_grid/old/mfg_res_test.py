


import tensorflow as tf
import tensorflow.keras as keras
import h5py
import tensorflow_probability as tfp

keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')


from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import FourierResidualLayer64
from pinns_data_assimilation.lib.layers import ProductResidualLayer64
from pinns_data_assimilation.lib.layers import CubicFourierProductBlock64
from pinns_data_assimilation.lib.layers import FourierEmbeddingLayer
from pinns_data_assimilation.lib.layers import AdjustableFourierTransformLayer
from pinns_data_assimilation.lib.layers import QuarticFourierProductBlock64

from pinns_data_assimilation.lib.downsample import compute_downsample_inds



def plot_err(epoch,model_RANS):
    global X_grid
    global Y_grid
    global i_test
    global MAX_o_train
    global o_test_grid
    pred_test = model_RANS.predict(i_test[:],batch_size=1000)
    pred_test_grid = np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],5])

    err_test1 = o_test_grid[:,:,0]/MAX_o_train-pred_test_grid[:,:,0]
    plot.figure(epoch)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,0]/MAX_o_train,levels=21)
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

    err_test2 = o_test_grid[:,:,1]/MAX_o_train-pred_test_grid[:,:,1]
    plot.figure(epoch+1)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,1]/MAX_o_train,levels=21)
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

    err_test3 = o_test_grid[:,:,2]/MAX_o_train-pred_test_grid[:,:,2]
    plot.figure(epoch+2)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,2]/MAX_o_train,levels=21)
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

    err_test4 = o_test_grid[:,:,3]/MAX_o_train-pred_test_grid[:,:,3]
    plot.figure(epoch+3)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,3]/MAX_o_train,levels=21)
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

    err_test5 = o_test_grid[:,:,4]/MAX_o_train-pred_test_grid[:,:,4]
    plot.figure(epoch+4)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,4]/MAX_o_train,levels=21)
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
    plot.show(block=False)
    plot.pause(5)

plot.ion()

HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')

global X_grid
global Y_grid

x = np.array(configFile['X_vec'][0,:])
x_test = x
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
y_test = y

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()



ux_grid = np.reshape(ux,X_grid.shape)
global o_test_grid

o_train = np.hstack((ux.reshape(-1,1),uy.reshape(-1,1),uxux.reshape(-1,1),uxuy.reshape(-1,1),uyuy.reshape(-1,1)))
o_test_grid = np.reshape(o_train,[X_grid.shape[0],X_grid.shape[1],5])
MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)

x_test = X_grid/MAX_x
y_test = Y_grid/MAX_x

global i_test

i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))
i_train = 1.0*i_test

MAX_o_train = np.max(np.abs(o_train.ravel()))
o_train = o_train/MAX_o_train


def network_loss(y_true,y_pred):
    loss_1 = tf.reduce_mean(tf.square(y_true-y_pred)) # u 
    return loss_1

if False:
    nodes = 100
    with tf.device('/CPU:0'):
        model_RANS = keras.Sequential()
        model_RANS.add(keras.layers.Dense(2,activation='linear',input_shape=(2,)))
        model_RANS.add(keras.layers.Dense(nodes,activation='linear'))
        for k in range(9):
            model_RANS.add(ResidualLayer(nodes,activation='elu'))
        model_RANS.add(keras.layers.Dense(5,activation='linear'))
        model_RANS.summary()
        model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=network_loss,jit_compile=False) 


if True:
    nodes = 100
    with tf.device('/CPU:0'):
        model_RANS = keras.Sequential()
        model_RANS.add(keras.layers.Dense(2,activation='linear',input_shape=(2,)))
        model_RANS.add(keras.layers.Dense(nodes,activation='linear'))
        for k in range(9):
            model_RANS.add(keras.layers.Dense(nodes,activation='elu'))
        model_RANS.add(keras.layers.Dense(5,activation='linear'))
        model_RANS.summary()
        model_RANS.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=network_loss,jit_compile=False) 


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
    keras.backend.set_value(model_RANS.optimizer.learning_rate, 1E-3)
    hist = model_RANS.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(500,model_RANS)
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

epochs=2500
if True:
    for L_iter in range(10):

        func = function_factory(model_RANS, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
        init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables)

        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_RANS.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        pred =  model_RANS.predict(i_train,batch_size=1000)
        epochs = epochs + LBFGS_epochs
        plot_err(epochs,model_RANS)



plot.show(block=True)

