import numpy as np
import scipy.io
from scipy import interpolate
from scipy.interpolate import griddata
import tensorflow as tf
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
import cmath

tf.keras.backend.set_floatx('float64')

# set constant seed to compare simulations with different hyper parameters
np.random.seed(1)
tf.random.set_seed(1)

save_loc = './tmp/checkpoint_testbed_all'
checkpoint_filepath = './tmp/checkpoint_testbed'

x = np.linspace(0, 2*np.pi, num=10)
ys = np.sin(x)
yc = np.cos(x)

#plt.figure
#plt.plot(x,ys)
#plt.plot(x,yc)
#plt.show()

# training data
O_train = np.hstack(((ys).reshape(-1,1),(yc).reshape(-1,1)))
X_train = x

y_scale = 1.0
x_scale = 1.0
phys_loss = 10

# collocation points for training physical constraints
f_colloc_train = np.linspace(0, 2*np.pi, num=50)

# create NN
with tf.device('/CPU:0'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='tanh', input_shape=(1,)))
    model.add(tf.keras.layers.Dense(30, activation='tanh'))
    model.add(tf.keras.layers.Dense(30, activation='tanh'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    model.summary()


# define cost function
@tf.function
def net_f(colloc_tensor):
    x = colloc_tensor
    yp = model(x)
    sp = yp[:,0]
    cp = yp[:,1]

    #dsp = tf.gradients(sp, x)[0]
    #dcp = tf.gradients(cp, x)[0]
    dsp = gradient_wrapper(sp, x)
    dcp = gradient_wrapper(cp, x)

    # set residual
    fs = dsp - cp  # d sin = cos
    fc = dcp + sp  # d cos = -sin

    return fs, fc

@tf.function
def net_f_comb(colloc_tensor):
    x = colloc_tensor
    yp = model(x)
    sp = yp[:,0]
    cp = yp[:,1]

    comb = sp + cp
    #comb1 = tf.add(cp, tf.math.negative(sp))

    #dsp = tf.gradients(sp, x)[0]
    #dcp = tf.gradients(cp, x)[0]
    dcomb = gradient_wrapper(comb, x)

    # set residual
    fcomb = dcomb - cp + sp  # d sin = cos

    return fcomb, fcomb

def gradient_wrapper(f,x):
    df = tf.gradients(f, x)[0]
    return df

# function wrapper, combine data and physics loss
def custom_loss_wrapper(colloc_tensor_f):
    def custom_loss(y_true, y_pred):
        data_loss1 = tf.keras.losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])  # sin
        data_loss2 = tf.keras.losses.mean_squared_error(y_true[:, 1], y_pred[:, 1])  # cos

        #ms, mc = net_f(colloc_tensor_f)
        ms, mc = net_f_comb(colloc_tensor_f)
        physical_loss1 = tf.reduce_mean(tf.square(ms))
        physical_loss2 = tf.reduce_mean(tf.square(mc))

        return data_loss1 + data_loss2 + phys_loss* physical_loss1 + phys_loss * physical_loss2

    return custom_loss


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=custom_loss_wrapper(f_colloc_train))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500)

# train network with different learning rates
hist = model.fit(X_train, O_train, batch_size=5, epochs=600, callbacks=[early_stop_callback, model_checkpoint_callback])
loss_history = hist.history['loss']

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0025)
model.load_weights(checkpoint_filepath)
hist = model.fit(X_train, O_train, batch_size=5, epochs=10, callbacks=[early_stop_callback, model_checkpoint_callback])
loss_history += hist.history['loss']

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.005)
model.load_weights(checkpoint_filepath)
hist = model.fit(X_train, O_train, batch_size=10, epochs=10, callbacks=[early_stop_callback, model_checkpoint_callback])
loss_history += hist.history['loss']

#model.load_weights(checkpoint_filepath)
#tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
#hist = model.fit(X_train, O_train, batch_size=128, epochs=50, callbacks=[early_stop_callback, model_checkpoint_callback])

#model.load_weights(checkpoint_filepath)
#tf.keras.backend.set_value(model.optimizer.learning_rate, 0.00001)
#hist = model.fit(X_train, O_train, batch_size=256, epochs=50, callbacks=[early_stop_callback, model_checkpoint_callback])

#model.load_weights(checkpoint_filepath)
#tf.keras.backend.set_value(model.optimizer.learning_rate, 0.000001)
#hist = model.fit(X_train, O_train, batch_size=512, epochs=100, callbacks=[early_stop_callback, model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

#loss_history = hist.history['loss']

# %matplotlib inline
plt.figure()
plt.semilogy(loss_history)
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.figure()
plt.plot(x,ys)
plt.plot(x,yc)
yp = model(x)
plt.plot(x,yp)
plt.show()

