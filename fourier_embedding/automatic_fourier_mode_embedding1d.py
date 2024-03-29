

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import h5py

import numpy as np


keras.backend.set_floatx('float64')


import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')

from pinns_galerkin_viv.lib.LBFGS_example import function_factory
from pinns_galerkin_viv.lib.layers import ResidualLayer
from pinns_galerkin_viv.lib.layers import FourierResidualLayer64


class resBlock(keras.layers.Layer):
    # a simple residual block
    def __init__(self,units):
        super().__init__()
        self.Dense  = keras.layers.Dense(units,activation='tanh')
        self.Linear = keras.layers.Dense(units,activation='linear')    
    
    def call(self,inputs):
        return tf.keras.activations.tanh(self.Linear(self.Dense(inputs))+inputs)
    

    
class QresBlock(keras.layers.Layer):
    # quadratic residual block from:
    # Bu, J., & Karpatne, A. (2021). Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) (pp. 675-683). Society for Industrial and Applied Mathematics.

    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.w1 = tf.Variable(initial_value=self.w_init(shape=(input_shape[-1],self.units),dtype=tf.float64),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w_init(shape=(input_shape[-1],self.units),dtype=tf.float64),trainable=True,name='w2')
        self.b_init = tf.zeros_initializer()
        self.b1 = tf.Variable(initial_value=self.b_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='b1')    
    
    def call(self,inputs):
        self.xw1 = tf.matmul(inputs,self.w1)
        return tf.keras.activations.tanh(tf.multiply(self.xw1,tf.matmul(inputs,self.w2))+self.xw1+self.b1)
    




HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
fourierModeFile = h5py.File(base_dir+'fourier_data_DFT.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')

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

ux_grid = np.reshape(ux,X_grid.shape)
phi_xr_grid = np.reshape(phi_xr,X_grid.shape)
MAX_i = np.max(X_grid)

i_train = X_grid[:,100]/MAX_i

o_train = phi_xr_grid[:,60]
#o_train = ux_grid[:,100]
MAX_o_train = np.max(o_train)
o_train = o_train/MAX_o_train

wave_ref = np.sin(12*2*np.pi*i_train+0*np.pi)

# check the frequency limit of the reference frequency
#plot.figure(1)
#plot.scatter(i_train,o_train)
#plot.scatter(i_train,wave_ref)
#plot.show()
#exit()

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
if True:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(FourierResidualLayer64(10))
        model_sines.add(ResidualLayer(10))
        model_sines.add(ResidualLayer(10))     
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 

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


shuffle_inds = np.array(range(i_train.shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

i_train_shuffle = (i_train[shuffle_inds]).transpose()
o_train_shuffle = (o_train[shuffle_inds]).transpose()


LBFGS_steps =1*3333
LBFGS_epochs = 3*LBFGS_steps

epochs = 0

if True:
    L_iter = 0
    func = function_factory(model_sines, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-12)
    func.assign_new_model_parameters(results.position)
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
    pred =  model_sines.predict(i_train,batch_size=1000)
    epochs = epochs + LBFGS_epochs
    L_iter = L_iter+1

if False:
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-3)
    hist = model_sines.fit(i_train_shuffle[:],o_train_shuffle[:], batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(i_train_shuffle[:],o_train_shuffle[:], batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(i_train_shuffle[:],o_train_shuffle[:], batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(i_train_shuffle[:],o_train_shuffle[:], batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-7)
    hist = model_sines.fit(i_train_shuffle[:],o_train_shuffle[:], batch_size=32, epochs=2000)

    pred = model_sines.predict(i_train,batch_size=1000)


err = o_train-pred[:,0]

print(y.shape)
print(x.shape)
print(pred.shape)

plot.figure(2)
plot.scatter(i_train,o_train)
plot.scatter(i_train,pred[:,0])

plot.figure(3)
plot.scatter(i_train,err)

plot.show()