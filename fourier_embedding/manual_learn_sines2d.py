

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

keras.backend.set_floatx('float64')

import numpy as np

import matplotlib.pyplot as plot

import sys
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.LBFGS_example import function_factory
from pinns_data_assimilation.lib.LBFGS_example import function_factory_diff_evo
from pinns_data_assimilation.lib.layers import ResidualLayer
from pinns_data_assimilation.lib.layers import FourierEmbeddingLayer
from pinns_data_assimilation.lib.layers import AdjustableFourierTransformLayer




x = np.linspace(-1,1,50,dtype=np.float64)
y = np.linspace(-1,1,50,dtype=np.float64)

x_grid,y_grid = np.meshgrid(x,y)

i_train = np.hstack((x_grid.reshape(-1,1),y_grid.reshape(-1,1)))
print(i_train.shape)

#f =  np.sin(0.1*np.pi*x_grid)*np.sin(4*np.pi*x_grid)*np.sin(3*np.pi*y_grid)
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
            model_sines.add(keras.layers.Dense(2,activation='linear',input_shape=(2,),))
            model_sines.add(FourierEmbeddingLayer(tf.cast(np.linspace(0,10,20,dtype=np.float64),tf.float64)))
            model_sines.add(keras.layers.Dense(60,activation='linear'))
            model_sines.add(ResidualLayer(60))  
            model_sines.add(ResidualLayer(60))      
            model_sines.add(keras.layers.Dense(1,activation='linear'))
            model_sines.summary()
            model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=network_loss,jit_compile=False) 

if True:
    with tf.device('/CPU:0'):
            model_sines = keras.Sequential()
            model_sines.add(keras.layers.Dense(2,activation='linear',input_shape=(2,),))
            model_sines.add(AdjustableFourierTransformLayer(60,30))
            model_sines.add(keras.layers.Dense(60,activation='linear'))
            model_sines.add(ResidualLayer(60))  
            model_sines.add(ResidualLayer(60))      
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


LBFGS_steps =333
LBFGS_epochs = 3*LBFGS_steps

epochs = 0



if False:
    
    #hist = model_sines.fit(tf.cast(i_train_shuffle,tf.float64),tf.cast(o_train_shuffle,tf.float64), batch_size=32, epochs=4000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-3)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=1000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-4)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=4000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-5)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=2000)
    keras.backend.set_value(model_sines.optimizer.learning_rate, 1E-6)
    hist = model_sines.fit(i_train_shuffle,o_train_shuffle, batch_size=32, epochs=2000)
    pred = model_sines.predict(i_train,batch_size=1000)

for w in range(10):
    func = function_factory(model_sines, network_loss, tf.cast(i_train_shuffle[:],tf.float64), tf.cast(o_train_shuffle[:],tf.float64))
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
    func.assign_new_model_parameters(results.position)
    init_params = tf.dynamic_stitch(func.idx, model_sines.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
    pred =  model_sines.predict(i_train,batch_size=1000)
    epochs = epochs + LBFGS_epochs


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
plot.contourf(x_grid,y_grid,err/np.max(f.ravel()),levels=21)
plot.set_cmap('bwr')
plot.colorbar()


plot.show()