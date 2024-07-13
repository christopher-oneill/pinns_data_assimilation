
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import math


class QresBlock(keras.layers.Layer):
    # quadratic residual block from:
    # Bu, J., & Karpatne, A. (2021). Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) (pp. 675-683). Society for Industrial and Applied Mathematics.

    def __init__(self,units,**kwargs):
        super().__init__()
        self.units = units
        self.built=False

        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'activation' in kwargs:
            if type(kwargs['activation'])==str:
                self.activation = tf.keras.activations.deserialize(kwargs['activation'])
            else:
                self.activation = kwargs['activation']
        else:
            self.activation =  tf.keras.activations.tanh
        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=tf.cast(kwargs['w2'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":self.units,
            "activation":self.activation,
            "w1":self.w1.numpy(),
            "w2":self.w2.numpy(),
            "b1":self.b1.numpy(),
            "dtype":self.u_dtype,
        })
        return config

    def build(self, input_shape):
        if self.built==False:
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=w_init(shape=(self.units,),dtype=self.u_dtype),trainable=True,name='b1')    
    
    def call(self,inputs):
        self.xw1 = tf.matmul(inputs,self.w1)
        return self.activation(tf.multiply(self.xw1,tf.matmul(inputs,self.w2))+self.xw1+self.b1)
    

class QresBlock2(keras.layers.Layer):
    # an updated quadratic residual block with skip connection

    def __init__(self,units,**kwargs):
        super().__init__()
        self.units = units
        self.built=False
        if 'dtype' in kwargs:
            self.u_dtype = kwargs['dtype']
        else:
            self.u_dtype = tf.float64
        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=tf.cast(kwargs['w2'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            if (tf.shape(self.w1)[0]==self.units):
                self.Residual = lambda inputs: inputs
            elif (tf.shape(self.w1)[0]<self.units):                                  
                self.Residual = lambda inputs: tf.concat((inputs,tf.zeros((tf.shape(inputs)[0],int(self.units)-tf.shape(self.w1)[0]),self.u_dtype)),axis=1)
            else:
                self.Residual = lambda inputs: inputs[:,0:self.units]
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":self.units,
            "w1":self.w1.numpy(),
            "w2":self.w2.numpy(),
            "b1":self.b1.numpy(),
            "Residual":self.Residual,
            "dtype":self.u_dtype,
        })
        return config

    def build(self, input_shape):
        if self.built==False:
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=w_init(shape=(self.units,),dtype=self.u_dtype),trainable=True,name='b1')
            if (input_shape[1]==self.units):
                self.Residual = lambda inputs: inputs
            elif (input_shape[1]<self.units):                                  
                self.Residual = lambda inputs: tf.concat((inputs,tf.zeros((tf.shape(inputs)[0],int(self.units)-input_shape[1]),self.u_dtype)),axis=1)
            else:
                self.Residual = lambda inputs: inputs[:,0:self.units]
    
    def call(self,inputs):
        self.xw1 = tf.matmul(inputs,self.w1)
        return tf.keras.activations.tanh(tf.multiply(self.xw1,tf.matmul(inputs,self.w2))+self.xw1+self.b1)+self.Residual(inputs)
    
class FourierResidualLayer64(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    # note that for some reason I was having issues with incompatible data types, so I just specified them explicitly. Thus 64. 

    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w1_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40) # initialization of frequencies needs to be customized per problem.

        self.w1 = tf.Variable(initial_value=self.w2_init(shape=(input_shape[-1],),dtype=tf.float64),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w1_init(shape=(input_shape[-1],self.units,),dtype=tf.float64),trainable=True,name='w2')
        self.b1 = tf.Variable(initial_value=self.w1_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='b1')
    
    def call(self,inputs):
        return tf.matmul(tf.math.cos(tf.multiply(inputs,self.w1)+self.b1),self.w2)+inputs
    

class FourierResidualLayer32(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w1_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40) # initialization of frequencies needs to be customized per problem.

        self.w1 = tf.Variable(initial_value=self.w2_init(shape=(input_shape[-1],),dtype=tf.float32),trainable=True,name='w1')
        self.w2 = tf.Variable(initial_value=self.w1_init(shape=(input_shape[-1],self.units,),dtype=tf.float32),trainable=True,name='w2')
        self.b1 = tf.Variable(initial_value=self.w1_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='b1')
    
    def call(self,inputs):
        return tf.matmul(tf.math.cos(tf.multiply(inputs,self.w1)+self.b1),self.w2)+inputs
    

class ResidualLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    # a simple residual block
    def __init__(self,units,activation='tanh',name='Dense',trainable=True,dtype=tf.float64,Dense=None,Residual=None):
        super().__init__()
        self.units = int(units)
        if Dense==None:
            self.Dense  = keras.layers.Dense(self.units,activation=activation,name=name,trainable=trainable,dtype=dtype)
        else:
            self.Dense = Dense


    def build(self,input_shape):
        if (input_shape[1]==self.units):
            self.Residual = lambda inputs: inputs
        elif (input_shape[1]<self.units):                                  
            self.Residual = lambda inputs: tf.concat((inputs,tf.zeros((tf.shape(inputs)[0],int(self.units)-input_shape[1]),tf.float64)),axis=1)
        else:
            self.Residual = lambda inputs: inputs[:,0:self.units]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":self.units,
            "Dense":self.Dense,
            "Residual":self.Residual,
        })
        return config
      
    def call(self,inputs):
        return self.Dense(inputs)+self.Residual(inputs)

class InputPassthroughLayer(keras.layers.Layer):
    # this layer passes through a certain number of inputs from the previous layer, while doing a normal dense layer otherwise
    # this allows the coordinate system to be passed through to deeper depths avoiding the vanishing gradient problem
    # this is similar to a residual layer, but exploits our knowledge that the pinn input is the coordinate system
    def __init__(self,units,npass,**kwargs):
        super().__init__()
        self.units = int(units)
        self.npass = int(npass)
        self.built=False
        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'activation' in kwargs:
            if type(kwargs['activation'])==str:
                self.activation = tf.keras.activations.deserialize(kwargs['activation'])
            else:
                self.activation = kwargs['activation']
        else:
            self.activation =  tf.keras.activations.tanh
        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def build(self,input_shape):
        if self.built==False:
            # check the case where npass is larger than the input dimensionality
            if input_shape[-1]<self.npass:
                self.npass=input_shape[-1]
            # setup weights
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=w_init(shape=(self.units,),dtype=self.u_dtype),trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":self.units,
            "npass":self.npass,
            "activation":self.activation,
            "w1":self.w1.numpy(),
            "b1":self.b1.numpy(),
        })
        return config
      
    def call(self,inputs):
        # the input dimensionality is [nbatch,ninputs]
        # concatenate the first npass inputs then the dense layer output. 
        # thus the output dimensionality will be [nbatch,npass+units] or [nbatch,ninputs+units] if (ninputs<npass)
        return tf.concat((inputs[...,0:self.npass],self.activation(tf.matmul(inputs,self.w1)+self.b1)),axis=1)

class QuadraticInputPassthroughLayer(keras.layers.Layer):
    # this layer passes through a certain number of inputs from the previous layer, while doing a quadratic residual layer otherwise
    # this allows the coordinate system to be passed through to deeper depths avoiding the vanishing gradient problem
    # this is similar to a residual layer, but exploits our knowledge that the pinn input is the coordinate system
    def __init__(self,units,npass,**kwargs):
        super().__init__()
        self.units = int(units)
        self.npass = int(npass)
        self.built=False

        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'activation' in kwargs:
            if type(kwargs['activation'])==str:
                self.activation = tf.keras.activations.deserialize(kwargs['activation'])
            else:
                self.activation = kwargs['activation']
        else:
            self.activation =  tf.keras.activations.tanh
        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=tf.cast(kwargs['w2'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":self.units,
            "npass":self.npass,
            "activation":self.activation,
            "w1":self.w1.numpy(),
            "w2":self.w2.numpy(),
            "b1":self.b1.numpy(),
            "dtype":self.u_dtype,
        })
        return config

    def build(self, input_shape):
        if self.built==False:
            # deal with input passthrough dimensionality
            if input_shape[-1]<self.npass:
                self.npass = input_shape[-1]
            # setup weights
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w1')
            self.w2 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype=self.u_dtype),trainable=True,name='w2')
            self.b1 = tf.Variable(initial_value=w_init(shape=(self.units,),dtype=self.u_dtype),trainable=True,name='b1')    
    
    def call(self,inputs):
        self.xw1 = tf.matmul(inputs,self.w1)
        return tf.concat((inputs[...,0:self.npass],self.activation(tf.multiply(self.xw1,tf.matmul(inputs,self.w2))+self.xw1+self.b1)),1)


    
class ProductResidualLayer64(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    # number of layer weights is factorial with width! so be careful picking too wide a network
    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        
        r = np.arange(input_shape[-1])
        self.mask = tf.reshape(tf.convert_to_tensor((r[:,None]<r),dtype=tf.int64),(input_shape[-1]*input_shape[-1],)) # mask matrix for getting the upper triangular part
        self.w1 = tf.Variable(initial_value=self.w_init(shape=(tf.reduce_sum(self.mask),self.units),dtype=tf.float64),trainable=True,name='w1')
           
    def call(self,inputs):
        self.input_prod = tf.reshape(tf.multiply(tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1]),tf.reshape(inputs,[tf.shape(inputs)[0],1,tf.shape(inputs)[1]])),[tf.shape(inputs)[0],tf.shape(inputs)[1]*tf.shape(inputs)[1]])
        self.input_prod = tf.boolean_mask(self.input_prod,self.mask,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations
        return tf.matmul(self.input_prod,self.w1)+inputs
    

# the idea behind the above layers was that we get products of fourier basis functions. 
# Unfortunately the number of weights is factorial with network width, which is problematic. 
# lets consider a residual layer where only a subset of the basis functions are activated with cosines. 


class CubicFourierProductBlock64(keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units # the number of frequencies and products, the weight matrix will be (units+2*unique(units*units)) x input_shape[-1]

   
    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40) # initialization of frequencies needs to be customized per problem.
        self.user_input_shape = input_shape
        # fourier layer
        self.fw1 = tf.Variable(initial_value=self.w2_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='wf1') # cosine frequency
        self.fw2 = tf.Variable(initial_value=self.w_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='wf2') # cosine ampltiude
        self.fb1 = tf.Variable(initial_value=self.w_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='bf1') # cosine phase

        # product layer 1
        r1 = np.zeros((self.units,self.units),np.int32) # 
        for k in range(self.units):
                for l in range(self.units):
                    if l>=(k):
                        r1[k,l]=1  
        self.product_mask1 = tf.reshape(tf.convert_to_tensor(r1,dtype=tf.int32),(self.units*self.units,)) # mask matrix for getting the upper triangular part
        self.nprod1 = tf.reduce_sum(self.product_mask1)
        # nprod = n^2-nC2
        
        # product layer 2
        r2 = np.zeros((self.nprod1,self.units),dtype=np.int32)
        offset=0
        for j in range(self.units):
            for k in range(self.units):
                for l in range(self.units):
                    if l>=(k+j):
                        r2[offset+k,l]=1
            offset=offset+(self.units-j)
        self.product_mask2 = tf.reshape(tf.convert_to_tensor(r2,dtype=tf.int32),(self.units*self.nprod1,))
        self.nprod2 = tf.reduce_sum(self.product_mask2)
        # output weights
        self.wo = tf.Variable(initial_value=self.w_init(shape=(self.units+self.nprod1+self.nprod2,self.units),dtype=tf.float64))

          
    def call(self,inputs):
        # compute fourier activations
        self.fourier_layer = tf.multiply(tf.math.cos(tf.multiply(inputs[:,0:self.units],self.fw1)+self.fb1),self.fw2)
        # compute the first product layer 
        # linear x linear
        self.product1 = tf.reshape(tf.multiply(tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],self.units,1]),tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],1,self.units])),[tf.shape(inputs)[0],self.units*self.units])
        self.product1 = tf.boolean_mask(self.product1,self.product_mask1,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations
        # compute the second product layer
        # unique quadratic x linear = cubic
        self.product2 = tf.reshape(tf.multiply(tf.reshape(self.product1,[tf.shape(inputs)[0],self.nprod1,1]),tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],1,self.units])),[tf.shape(inputs)[0],self.units*self.nprod1])
        self.product2 = tf.boolean_mask(self.product2,self.product_mask2,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations

        self.combinations = tf.concat((self.fourier_layer,self.product1,self.product2),1)

        #self.output_product = tf.matmul()
        
        return tf.concat((tf.matmul(self.combinations,self.wo),tf.zeros((tf.shape(inputs)[0],tf.shape(inputs)[-1]-self.units,),dtype=tf.float64)),1)+inputs    

class QuarticFourierProductBlock64(keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units # the number of frequencies and products, the weight matrix will be (units+2*unique(units*units)) x input_shape[-1]

   
    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.w2_init = tf.random_uniform_initializer(0,2*np.pi*40) # initialization of frequencies needs to be customized per problem.
        self.user_input_shape = input_shape
        # fourier layer
        self.fw1 = tf.Variable(initial_value=self.w2_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='wf1') # cosine frequency
        self.fw2 = tf.Variable(initial_value=self.w_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='wf2') # cosine ampltiude
        self.fb1 = tf.Variable(initial_value=self.w_init(shape=(self.units,),dtype=tf.float64),trainable=True,name='bf1') # cosine phase

        # product layer 1
        r1 = np.zeros((self.units,self.units),np.int32) # 
        for k in range(self.units):
                for l in range(self.units):
                    if l>=(k):
                        r1[k,l]=1  
        self.product_mask1 = tf.reshape(tf.convert_to_tensor(r1,dtype=tf.int32),(self.units*self.units,)) # mask matrix for getting the upper triangular part
        self.nprod1 = tf.reduce_sum(self.product_mask1)
        # nprod = n^2-nC2
        
        # product layer 2
        r2 = np.zeros((self.nprod1,self.units),dtype=np.int32)
        offset=0
        for j in range(self.units):
            for k in range(self.units):
                for l in range(self.units):
                    if l>=(k+j):
                        r2[offset+k,l]=1
            offset=offset+(self.units-j)
        self.product_mask2 = tf.reshape(tf.convert_to_tensor(r2,dtype=tf.int32),(self.units*self.nprod1,))
        self.nprod2 = tf.reduce_sum(self.product_mask2)

        # product layer 3
        r3 = np.zeros((self.nprod2,self.units),dtype=np.int32)
        offset=0
        for j in range(self.units):
            for k in range(self.units):
                for l in range(self.units):
                    if l>=(k+j):
                        r3[offset+k,l]=1
            offset=offset+(self.units-j)
        self.product_mask3 = tf.reshape(tf.convert_to_tensor(r3,dtype=tf.int32),(self.units*self.nprod2,))
        self.nprod3 = tf.reduce_sum(self.product_mask3)

        # output weights
        self.wo = tf.Variable(initial_value=self.w_init(shape=(self.units+self.nprod1+self.nprod2+self.nprod3,self.units),dtype=tf.float64))

          
    def call(self,inputs):
        # compute fourier activations
        self.fourier_layer = tf.multiply(tf.math.cos(tf.multiply(inputs[:,0:self.units],self.fw1)+self.fb1),self.fw2)
        # compute the first product layer 
        # linear x linear
        self.product1 = tf.reshape(tf.multiply(tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],self.units,1]),tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],1,self.units])),[tf.shape(inputs)[0],self.units*self.units])
        self.product1 = tf.boolean_mask(self.product1,self.product_mask1,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations
        # compute the second product layer
        # unique quadratic x linear = cubic
        self.product2 = tf.reshape(tf.multiply(tf.reshape(self.product1,[tf.shape(inputs)[0],self.nprod1,1]),tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],1,self.units])),[tf.shape(inputs)[0],self.units*self.nprod1])
        self.product2 = tf.boolean_mask(self.product2,self.product_mask2,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations

        self.product3 = tf.reshape(tf.multiply(tf.reshape(self.product2,[tf.shape(inputs)[0],self.nprod2,1]),tf.reshape(self.fourier_layer,[tf.shape(inputs)[0],1,self.units])),[tf.shape(inputs)[0],self.units*self.nprod2])
        self.product3 = tf.boolean_mask(self.product3,self.product_mask3,axis=1) # take only the upper triangular part of the matrix, to eliminate redundant combinations

        self.combinations = tf.concat((self.fourier_layer,self.product1,self.product2,self.product3),1)

        #self.output_product = tf.matmul()
        
        return tf.concat((tf.matmul(self.combinations,self.wo),tf.zeros((tf.shape(inputs)[0],tf.shape(inputs)[-1]-self.units,),dtype=tf.float64)),1)+inputs    

# a more classical fourier embedding layer, passes through the input
class FourierEmbeddingLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    
    def __init__(self,frequency_vector,**kwargs):
        super(FourierEmbeddingLayer,self).__init__()
        self.frequency_vector = tf.reshape(tf.convert_to_tensor(tf.cast(frequency_vector,tf.float64)),(1,1,tf.size(frequency_vector)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "frequency_vector":self.frequency_vector.numpy(),
        })
        return config
          
    def call(self,inputs):
        inp_shape = tf.shape(inputs)
        inp_prod = tf.reshape(tf.multiply(tf.reshape(inputs,(inp_shape[0],inp_shape[-1],1)),self.frequency_vector),(inp_shape[0],inp_shape[-1]*tf.shape(self.frequency_vector)[2]))
        return tf.concat((inputs,tf.cos(inp_prod),tf.sin(inp_prod)),axis=1)

class FourierPassthroughEmbeddingLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    
    def __init__(self,frequency_vector,npass,**kwargs):
        super(FourierPassthroughEmbeddingLayer,self).__init__()
        self.frequency_vector = tf.reshape(tf.convert_to_tensor(tf.cast(frequency_vector,tf.float64)),(1,1,tf.size(frequency_vector)))
        self.npass = int(npass)

    def get_config(self):
        config = super().get_config()
        config.update({
            "frequency_vector":self.frequency_vector.numpy(),
            "npass":self.npass,
        })
        return config
        
    def build(self,input_shape):
        if input_shape[-1]<self.npass:
            self.npass = input_shape[-1]

    def call(self,inputs):
        inp_shape = tf.shape(inputs)
        inp_prod = tf.reshape(tf.multiply(tf.reshape(inputs,(inp_shape[0],inp_shape[-1],1)),self.frequency_vector),(inp_shape[0],inp_shape[-1]*tf.shape(self.frequency_vector)[2]))
        return tf.concat((inputs[...,0:self.npass],tf.cos(inp_prod),tf.sin(inp_prod)),axis=1)

class FourierPassthroughLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    
    def __init__(self,nodes,npass,**kwargs):
        super(FourierPassthroughLayer,self).__init__()
        self.nodes = int(nodes)
        self.npass = int(npass)
        self.built=False
        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "nodes":self.nodes,
            "npass":self.npass,
            "w1":self.w1.numpy(),
            "b1":self.b1.numpy()
        })
        return config
    
    def build(self,input_shape):
        if self.built==False:
            # deal with input passthrough dimensionality
            if input_shape[-1]<self.npass:
                self.npass = input_shape[-1]
            # setup weights
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.nodes),dtype=self.u_dtype),trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=w_init(shape=(self.nodes,),dtype=self.u_dtype),trainable=True,name='b1')  
          
    def call(self,inputs):
        return tf.concat((inputs[...,0:self.npass],tf.cos(tf.matmul(inputs,self.w1)+self.b1)),axis=1)
    

class FourierPassthroughLayer2(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    
    def __init__(self,frequency_vector,npass,**kwargs):
        super(FourierPassthroughLayer2,self).__init__()
        self.frequency_vector = tf.reshape(tf.convert_to_tensor(tf.cast(frequency_vector,tf.float64)),(1,tf.size(frequency_vector)))
        self.npass = int(npass)
        self.built=False
        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "frequency_vector":self.frequency_vector.numpy(),
            "npass":self.npass,
            "w1":self.w1.numpy(),
            "b1":self.b1.numpy()
        })
        return config
    
    def build(self,input_shape):
        if self.built==False:
            # deal with input passthrough dimensionality
            if input_shape[-1]<self.npass:
                self.npass = input_shape[-1]
            # setup weights
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],tf.shape(self.frequency_vector)[1]),dtype=self.u_dtype),trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=w_init(shape=(tf.shape(self.frequency_vector)[1],),dtype=self.u_dtype),trainable=True,name='b1')  
          
    def call(self,inputs):
        return tf.concat((inputs[...,0:self.npass],tf.cos(tf.multiply(self.frequency_vector,tf.matmul(inputs,self.w1)+self.b1))),axis=1)



class FourierPassthroughReductionLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    
    def __init__(self,frequency_vector,npass,**kwargs):
        super(FourierPassthroughReductionLayer,self).__init__()
        self.frequency_vector = tf.reshape(tf.convert_to_tensor(tf.cast(frequency_vector,tf.float64)),(1,tf.size(frequency_vector)))
        self.npass = int(npass)
        self.built=False
        if 'dtype' in kwargs:
            self.u_dtype=kwargs['dtype']
        else:
            self.u_dtype=tf.float64

        if 'activation' in kwargs:
            if type(kwargs['activation'])==str:
                self.activation = tf.keras.activations.deserialize(kwargs['activation'])
            else:
                self.activation = kwargs['activation']
        else:
            self.activation =  tf.keras.activations.tanh
        if 'w1' in kwargs:
            self.w1 = tf.Variable(initial_value=tf.cast(kwargs['w1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=tf.cast(kwargs['b1'],self.u_dtype),dtype=self.u_dtype,trainable=True,name='b1')
            self.built=True

    def get_config(self):
        config = super().get_config()
        config.update({
            "frequency_vector":self.frequency_vector.numpy(),
            "npass":self.npass,
            "activation":self.activation,
            "w1":self.w1.numpy(),
            "b1":self.b1.numpy()
        })
        return config
    
    def build(self,input_shape):
        if self.built==False:
            # deal with input passthrough dimensionality
            if input_shape[-1]<self.npass:
                self.npass = input_shape[-1]
            # setup weights
            w_init = tf.random_normal_initializer()
            self.w1 = tf.Variable(initial_value=w_init(shape=(input_shape[-1],tf.shape(self.frequency_vector)[1]),dtype=self.u_dtype),trainable=True,name='w1')
            self.b1 = tf.Variable(initial_value=w_init(shape=(tf.shape(self.frequency_vector)[1],),dtype=self.u_dtype),trainable=True,name='b1')  
          
    def call(self,inputs):
        return tf.concat((inputs[...,0:self.npass],tf.cos(tf.multiply(self.frequency_vector,self.activation(tf.matmul(inputs,self.w1)+self.b1))),tf.sin(tf.multiply(self.frequency_vector,self.activation(tf.matmul(inputs,self.w1)+self.b1)))),axis=1)

class CylindricalEmbeddingLayer(keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super(CylindricalEmbeddingLayer,self).__init__(*args,**kwargs)
     
    def call(self,inputs):
        inp_shape = tf.shape(inputs)
        r2 = tf.reshape(tf.square(inputs[:,0])+tf.square(inputs[:,1]),(inp_shape[0],1))
        #theta = tf.reshape(tf.atan2(inputs[:,1],inputs[:,0]),(inp_shape[0],1))
        # tf.reshape(tf.multiply(inputs[:,0],inputs[:,1]),(inp_shape[0],1)),tf.reshape(tf.multiply(inputs[:,0],tf.multiply(inputs[:,0],inputs[:,1])),(inp_shape[0],1)),r,tf.square(r)
        return tf.concat((inputs,r2),axis=1)

class AdjustableFourierTransformLayer(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    def __init__(self,units):
        super().__init__()
        self.units = units

    def __init__(self,nfreq,max_freq):
        super().__init__()
        self.nfreq = nfreq
        self.max_freq = max_freq
        self.df = tf.cast(max_freq/(nfreq-1),tf.float64)

    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.b1 = tf.reshape(tf.convert_to_tensor(np.linspace(0,self.max_freq,self.nfreq),dtype=tf.float64),(1,self.nfreq)) # frequency bin centers
        self.w1 = tf.Variable(initial_value=self.w_init(shape=(1,self.nfreq,),dtype=tf.float64),trainable=True,name='w1') # frequency adjustment weight
        
    
    def call(self,inputs):
        inp_shape = tf.shape(inputs)
        freq_vector = self.b1+tf.multiply(self.df,keras.activations.sigmoid(self.w1))
        inp_prod = tf.reshape(tf.multiply(tf.reshape(inputs,(inp_shape[0],inp_shape[-1],1)),freq_vector),(inp_shape[0],inp_shape[-1]*self.nfreq))
        return tf.concat((inputs,tf.cos(2.0*np.pi*inp_prod),tf.sin(2.0*np.pi*inp_prod)),axis=1)
    

class AdjustableFourierTransformLayer2(keras.layers.Layer):
    # Chris O'Neill, University of Calgary 2023
    def __init__(self,units):
        super().__init__()
        self.units = units

    def __init__(self,nfreq,max_freq):
        super().__init__()
        self.nfreq = nfreq
        self.max_freq = max_freq
        self.df = tf.cast(max_freq/(nfreq-1),tf.float64)

    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.b1 = tf.reshape(tf.convert_to_tensor(np.linspace(0,self.max_freq,self.nfreq),dtype=tf.float64),(1,self.nfreq)) # frequency bin centers
        self.w1 = tf.Variable(initial_value=self.w_init(shape=(1,self.nfreq,),dtype=tf.float64),trainable=True,name='w1') # frequency adjustment weight
        
    
    def call(self,inputs):
        inp_shape = tf.shape(inputs)
        freq_vector = self.b1+tf.multiply(self.df,keras.activations.sigmoid(self.w1))
        inp_prod = tf.reshape(tf.multiply(tf.reshape(inputs,(inp_shape[0],inp_shape[-1],1)),freq_vector),(inp_shape[0],inp_shape[-1]*self.nfreq))
        return tf.concat((inputs,tf.cos(2.0*np.pi*inp_prod),tf.sin(2.0*np.pi*inp_prod)),axis=1)
    