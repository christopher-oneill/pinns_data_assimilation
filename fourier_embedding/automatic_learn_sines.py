

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')

from pinns_galerkin_viv.lib.LBFGS_example import function_factory
from pinns_galerkin_viv.lib.layers import ResidualLayer
from pinns_galerkin_viv.lib.layers import FourierResidualLayer32

from pinns_galerkin_viv.lib.layers import TruncatedFourierProductBlock64



x = np.linspace(-1,1,1000,dtype=np.float64)
y = np.sin(10*np.pi*x,dtype=np.float64)

MAX_y = np.max(y)

x_train = x


def network_loss(y_true,y_pred):
    data_loss = tf.reduce_mean(tf.square(y_true-y_pred)) # u 
    return data_loss


if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(10, activation='linear', input_shape=(1,)))
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
        model_sines.add(keras.layers.Dense(10,activation='linear',input_shape=(1,)))
        model_sines.add(ResidualLayer(10))
        model_sines.add(ResidualLayer(10))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 
        
if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(10,activation='linear',input_shape=(1,),))
        model_sines.add(FourierResidualLayer32(10))
        model_sines.add(ResidualLayer(10))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 

if True: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(20,activation='linear',input_shape=(1,),))
        model_sines.add(TruncatedFourierProductBlock64(10))
        model_sines.add(ResidualLayer(20))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 


shuffle_inds = np.array(range(x_train.shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

x_train_shuffle = (x_train[shuffle_inds]).transpose()
y_train_shuffle = (y[shuffle_inds]).transpose()

LBFGS_steps =1000
LBFGS_epochs = 3*LBFGS_steps

epochs = 0

if True:
    L_iter = 0
    func = function_factory(model_sines, network_loss, x_train_shuffle[:], y_train_shuffle[:])
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-12)
    func.assign_new_model_parameters(results.position)
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
    pred =  model_sines.predict(x_train,batch_size=1000)
    epochs = epochs + LBFGS_epochs
    L_iter = L_iter+1

if False:
    print(x_train.shape)
    print(x_train_shuffle.shape)
    print(x_train_shuffle.dtype)
    hist = model_sines.fit(tf.cast(x_train_shuffle[:],tf.float64),tf.cast(y_train_shuffle[:],tf.float64), batch_size=32, epochs=10000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-3)
    hist = model_sines.fit(x_train_shuffle[:],y_train_shuffle[:], batch_size=32, epochs=5000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(x_train_shuffle[:],y_train_shuffle[:], batch_size=32, epochs=3000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(x_train_shuffle[:],y_train_shuffle[:], batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(x_train_shuffle[:],y_train_shuffle[:], batch_size=32, epochs=1000)
    pred = model_sines.predict(x_train,batch_size=1000)

err = y-pred[:,0]

print(y.shape)
print(x.shape)
print(pred.shape)

plot.figure(1)
plot.subplot(2,1,1)
plot.scatter(x,y)
plot.scatter(x,pred[:,0])
plot.legend(['raw','NN'])
plot.subplot(2,1,2)
plot.scatter(x,err)
plot.ylabel('NN-raw')

plot.show()