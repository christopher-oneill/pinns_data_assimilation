

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

import matplotlib.pyplot as plot

x = np.linspace(-1,1,1000)
y = np.sin(10*np.pi*x)

x_train = x/5.0

class FourierResidualLayer(keras.layers.Layer):
    # quadratic residual block from:
    # Bu, J., & Karpatne, A. (2021). Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) (pp. 675-683). Society for Industrial and Applied Mathematics.

    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w1_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*10)

        self.w1 = tf.Variable(initial_value=self.w2_init(shape=(input_shape[-1],self.units)),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w1_init(shape=(self.units,)),trainable=True,name='w2')
    
    def call(self,inputs):
        return tf.multiply(tf.math.cos(tf.matmul(inputs,self.w1)),self.w2)+inputs
    

class resBlock2(keras.layers.Layer):
    # a simple residual block
    def __init__(self,units):
        super().__init__()
        self.units = units
        self.Dense  = keras.layers.Dense(self.units,activation='tanh')
    
      
    def call(self,inputs):
        return self.Dense(inputs)+inputs



if False:
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1, activation='linear', input_shape=(1,)))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(10,activation='tanh'))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 

if False: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(resBlock2(10))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 
        
if True: 
    with tf.device('/CPU:0'):
        model_sines = keras.Sequential()
        model_sines.add(keras.layers.Dense(1,activation='linear',input_shape=(1,)))
        model_sines.add(FourierLayer(10))
        model_sines.add(keras.layers.Dense(1,activation='linear'))
        model_sines.summary()
        model_sines.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss=tf.losses.mean_absolute_error,jit_compile=False) 

shuffle_inds = np.array(range(x_train.shape[0])).transpose()
shuffle_inds = np.random.shuffle(shuffle_inds)

x_train_shuffle = (x_train[shuffle_inds]).transpose()
y_train_shuffle = (y[shuffle_inds]).transpose()

print(x_train.shape)
print(x_train_shuffle.shape)
hist = model_sines.fit(x_train_shuffle[:],y_train_shuffle[:], batch_size=32, epochs=10000)
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
plot.scatter(x,y)
plot.scatter(x,pred[:,0])

plot.figure(2)
plot.scatter(x,err)

plot.show()