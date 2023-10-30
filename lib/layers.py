
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class FourierResidualLayer64(keras.layers.Layer):
    # quadratic residual block from:
    # Bu, J., & Karpatne, A. (2021). Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) (pp. 675-683). Society for Industrial and Applied Mathematics.

    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w1_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40)

        self.w1 = tf.Variable(initial_value=self.w2_init(shape=(input_shape[-1],self.units),dtype=tf.float64),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w1_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='w2')
    
    def call(self,inputs):
        #inputs = tf.cast(inputs,tf.float64)
        return tf.multiply(tf.math.cos(tf.matmul(inputs,self.w1)),self.w2)+inputs
    

class FourierResidualLayer32(keras.layers.Layer):
    # quadratic residual block from:
    # Bu, J., & Karpatne, A. (2021). Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) (pp. 675-683). Society for Industrial and Applied Mathematics.

    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w1_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40)

        self.w1 = tf.Variable(initial_value=self.w2_init(shape=(input_shape[-1],self.units),dtype=tf.float32),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w1_init(shape=(self.units,),dtype=tf.float32),trainable=True,name='w2')
    
    def call(self,inputs):
        #inputs = tf.cast(inputs,tf.float64)
        return tf.multiply(tf.math.cos(tf.matmul(inputs,self.w1)),self.w2)+inputs
    

class ResidualLayer(keras.layers.Layer):
    # a simple residual block
    def __init__(self,units):
        super().__init__()
        self.units = units
        self.Dense  = keras.layers.Dense(self.units,activation='tanh')
    
      
    def call(self,inputs):
        return self.Dense(inputs)+inputs