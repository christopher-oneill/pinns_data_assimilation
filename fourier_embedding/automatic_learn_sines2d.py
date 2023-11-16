

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')

from pinns_galerkin_viv.lib.LBFGS_example import function_factory
from pinns_galerkin_viv.lib.LBFGS_example import function_factory_diff_evo
from pinns_galerkin_viv.lib.layers import ResidualLayer
from pinns_galerkin_viv.lib.layers import FourierResidualLayer64
from pinns_galerkin_viv.lib.layers import ProductResidualLayer64
from pinns_galerkin_viv.lib.layers import CubicFourierProductBlock64
from pinns_galerkin_viv.lib.layers import QuarticFourierProductBlock64





x = np.linspace(-1,1,50,dtype=np.float64)
y = np.linspace(-1,1,50,dtype=np.float64)

x_grid,y_grid = np.meshgrid(x,y)

i_train = np.hstack((x_grid.reshape(-1,1),y_grid.reshape(-1,1)))
print(i_train.shape)

#f = np.sin(0.3*np.pi*x_grid-2*np.pi)*np.sin(4*np.pi*x_grid)*np.sin(3*np.pi*y_grid) # cubic function example
f = np.sin(np.pi*x_grid+np.pi*y_grid-2*np.pi)*np.sin(4*np.pi*x_grid)*np.sin(3*np.pi*y_grid) # quartic function example
MAX_o = np.max(f,axis=(0,1))
o_train = (f.reshape(-1,1))/MAX_o

plot.figure(1)
plot.contourf(x_grid,y_grid,f)
plot.show()



def network_loss(y_true,y_pred):
    data_loss = tf.reduce_mean(tf.square(y_true-y_pred)) # u 
    return data_loss


if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(10, activation='linear', input_shape=(2,)))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 

if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(ResidualLayer(10))
        model_sines.add(ResidualLayer(10))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 
        
if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(20,activation='linear',input_shape=(2,),))
        model_sines.add(Fourier2ResidualLayer64(20))
        model_sines.add(Product2ResidualLayer64(20))
        model_sines.add(Product2ResidualLayer64(20))
        model_sines.add(ResidualLayer(20))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False)

#test_input = np.zeros([2,6])
#test_input[0,:] = np.array([1,2,4,6,8,10])
#test_input[1,:] = np.array([3,6,9,12,15,18])
#test_output = np.multiply(np.reshape(test_input,[test_input.shape[0],1,test_input.shape[1]]),np.reshape(test_input,[test_input.shape[0],test_input.shape[1],1]))
#print(test_input)
#print(test_input.shape)
#print(test_output)
#print(test_output.shape)
#exit()

if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(20,activation='linear',input_shape=(2,),))
        model_sines.add(FourierResidualLayer64(20))
        model_sines.add(ProductResidualLayer64(20))
        model_sines.add(ResidualLayer(20))
        model_sines.add(ProductResidualLayer64(20))
        model_sines.add(ResidualLayer(20))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 

with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(40,activation='linear',input_shape=(2,),))
        model_sines.add(QuarticFourierProductBlock64(20))        
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 

shuffle_inds = np.array(range(i_train.shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

i_train_shuffle = (i_train[shuffle_inds,:])[0,:,:]
o_train_shuffle = (o_train[shuffle_inds])[0,:]

print(i_train.shape)
print(i_train_shuffle.shape)
print(i_train_shuffle.dtype)

print(o_train.shape)
print(o_train_shuffle.shape)
print(o_train_shuffle.dtype)


LBFGS_steps =20*333
LBFGS_epochs = 3*LBFGS_steps

epochs = 0



if False:
    L_iter = 0
    func = function_factory_diff_evo(model_sines, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

    results = tfp.optimizer.differential_evolution_minimize(func,None,init_params,100)
    func.assign_new_model_parameters(results.position)
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
    pred =  model_sines.predict(i_train,batch_size=1000)
    epochs = epochs + LBFGS_epochs
    L_iter = L_iter+1

if False:
    
    #hist = model_sines.fit(tf.cast(i_train_shuffle,tf.float64),tf.cast(o_train_shuffle,tf.float64), batch_size=32, epochs=4000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-3)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=4000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=4000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=2000)
    pred = model_sines.predict(i_train,batch_size=1000)

if True:
    L_iter = 0
    func = function_factory(model_sines, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
    func.assign_new_model_parameters(results.position)
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
    pred =  model_sines.predict(i_train,batch_size=1000)
    epochs = epochs + LBFGS_epochs
    L_iter = L_iter+1


pred = np.reshape(pred,f.shape)*MAX_o

err = f-pred

print(y.shape)
print(x.shape)
print(pred.shape)

plot.figure(2)
plot.subplot(3,1,1)
plot.contourf(x_grid,y_grid,f,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,2)
plot.contourf(x_grid,y_grid,pred,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.subplot(3,1,3)
plot.contourf(x_grid,y_grid,err,levels=21)
plot.set_cmap('bwr')
plot.colorbar()


plot.show()