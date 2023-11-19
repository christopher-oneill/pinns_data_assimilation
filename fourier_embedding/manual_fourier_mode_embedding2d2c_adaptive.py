

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


def plot_err(epoch,model_sines):
    global X_grid
    global Y_grid
    global i_test
    global MAX_o_train
    global o_test_grid
    pred_test = model_sines.predict(i_test[:],batch_size=1000)
    pred_test_grid = np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],11])

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

    err_test6 = o_test_grid[:,:,5]/MAX_o_train-pred_test_grid[:,:,5]
    plot.figure(epoch+5)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,5]/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,5],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test6,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test7 = o_test_grid[:,:,6]/MAX_o_train-pred_test_grid[:,:,6]
    plot.figure(epoch+6)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,6]/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,6],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test7,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test8 = o_test_grid[:,:,7]/MAX_o_train-pred_test_grid[:,:,7]
    plot.figure(epoch+7)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,7]/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,7],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test8,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    plot.pause(1)

    err_test9 = o_test_grid[:,:,8]/MAX_o_train-pred_test_grid[:,:,8]
    plot.figure(epoch+8)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,8]/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,8],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test9,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    err_test10 = o_test_grid[:,:,9]/MAX_o_train-pred_test_grid[:,:,9]
    plot.figure(epoch+9)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,o_test_grid[:,:,9]/MAX_o_train,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,9],levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,err_test10,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()

    plot.figure(epoch+10)
    plot.contourf(X_grid,Y_grid,100.0/(1+np.exp(-pred_test_grid[:,:,10]))+1,levels=np.linspace(0,101,21))
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.show(block=False)
    plot.pause(5)

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

tau_xx_r = np.array(fourierModeFile['stressModesShortReal'][0,mode_number,:]).transpose()
tau_xx_i = np.array(fourierModeFile['stressModesShortImag'][0,mode_number,:]).transpose()
tau_xy_r = np.array(fourierModeFile['stressModesShortReal'][1,mode_number,:]).transpose()
tau_xy_i = np.array(fourierModeFile['stressModesShortImag'][1,mode_number,:]).transpose()
tau_yy_r = np.array(fourierModeFile['stressModesShortReal'][2,mode_number,:]).transpose()
tau_yy_i = np.array(fourierModeFile['stressModesShortImag'][2,mode_number,:]).transpose()


print(phi_xr.shape)
print(x.shape)


ux_grid = np.reshape(ux,X_grid.shape)
global o_test_grid
o_test_grid = np.reshape(np.hstack((phi_xr.reshape(-1,1),phi_xi.reshape(-1,1),phi_yr.reshape(-1,1),phi_yi.reshape(-1,1),tau_xx_r.reshape(-1,1),tau_xx_i.reshape(-1,1),tau_xy_r.reshape(-1,1),tau_xy_i.reshape(-1,1),tau_yy_r.reshape(-1,1),tau_yy_i.reshape(-1,1))),[X_grid.shape[0],X_grid.shape[1],10])
MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)

x_test = X_grid/MAX_x
y_test = Y_grid/MAX_x

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
    phi_xi = phi_xi[downsample_inds]
    phi_yr = phi_yr[downsample_inds]
    phi_yi = phi_yi[downsample_inds]
    tau_xx_r = tau_xx_r[downsample_inds]
    tau_xx_i = tau_xx_i[downsample_inds]
    tau_xy_r = tau_xy_r[downsample_inds]
    tau_xy_i = tau_xy_i[downsample_inds]
    tau_yy_r = tau_yy_r[downsample_inds]
    tau_yy_i = tau_yy_i[downsample_inds]
else:
    n_x_d = X_grid.shape[0]
    n_y_d = X_grid.shape[1]

o_train_grid  = np.reshape(np.hstack((phi_xr.reshape(-1,1),phi_xi.reshape(-1,1),phi_yr.reshape(-1,1),phi_yi.reshape(-1,1),tau_xx_r.reshape(-1,1),tau_xx_i.reshape(-1,1),tau_xy_r.reshape(-1,1),tau_xy_i.reshape(-1,1),tau_yy_r.reshape(-1,1),tau_yy_i.reshape(-1,1))),(n_x_d,n_y_d,10))

x_train = x/MAX_x
y_train = y/MAX_x

X_train_grid = np.reshape(x_train,(n_x_d,n_y_d))
Y_train_grid = np.reshape(y_train,(n_x_d,n_y_d))

i_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
print(i_train.shape)

o_train = np.hstack((phi_xr.reshape(-1,1),phi_xi.reshape(-1,1),phi_yr.reshape(-1,1),phi_yi.reshape(-1,1),tau_xx_r.reshape(-1,1),tau_xx_i.reshape(-1,1),tau_xy_r.reshape(-1,1),tau_xy_i.reshape(-1,1),tau_yy_r.reshape(-1,1),tau_yy_i.reshape(-1,1)))
print(o_train.shape)
#o_train = ux_grid[:,100]
MAX_o_train = np.max(o_train)
o_train = o_train/MAX_o_train



def network_loss(y_true,y_pred):
    adaptive_loss = 100.0/(1+tf.exp(-y_pred[:,10]))+1
    data_loss = tf.multiply(adaptive_loss,tf.reduce_sum(tf.square(y_true[:,0:10]-y_pred[:,0:10]),1)) # u 
    return 1/(1+tf.reduce_sum(adaptive_loss))-1/(1+101*X_grid.size) + tf.reduce_mean(data_loss)

if False:
    nodes = 60
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(2,activation='linear',input_shape=(2,)))
        model_sines.add(FourierEmbeddingLayer(tf.cast(np.linspace(0,30,300,dtype=np.float64),tf.float64)))
        model_sines.add(keras.layers.Dense(nodes,activation='linear'))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        model_sines.add(ResidualLayer(nodes))
        

        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=network_loss,jit_compile=False) 

if True:
    nodes = 120
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(2,activation='linear',input_shape=(2,)))
        #model_sines.add(AdjustableFourierTransformLayer(60,30))
        model_sines.add(FourierEmbeddingLayer(tf.cast(np.linspace(0,30,120,dtype=np.float64),tf.float64)))
        model_sines.add(keras.layers.Dense(nodes,activation='linear'))
        for k in range(18):
            model_sines.add(ResidualLayer(nodes,activation='elu'))
        model_sines.add(keras.layers.Dense(11,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss=network_loss,jit_compile=False) 

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
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(500,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(1000,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(1500,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-7)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
    plot_err(2000,model_sines)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-8)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=50)
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


plot.show(block=True)

