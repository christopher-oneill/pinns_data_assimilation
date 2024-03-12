#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.io
from scipy import interpolate
from scipy.interpolate import griddata
import tensorflow as tf
import tensorflow.keras as keras
import h5py
import os
import glob
import sys
import re
import smt
import h5py
from smt.sampling_methods import LHS
from pyDOE import lhs
from datetime import datetime
from datetime import timedelta
import platform

keras.backend.set_floatx('float64')
dtype_train = tf.float64

case = 'JFM'
start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

PLOT = False


assert len(sys.argv)==6

job_number = int(sys.argv[1])
supersample_factor = int(sys.argv[2])
dense_layers = int(sys.argv[3])
dense_nodes = int(sys.argv[4])
duration_hours = int(sys.argv[5])


job_name = 'mfg_adaptive2_mean{:03d}_S{:d}_L{:d}N{:d}'.format(job_number,supersample_factor,dense_layers,dense_nodes)


LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_beluga/sync/'
    HOMEDIR = 'C:/projects/pinns_beluga/sync/'
    sys.path.append('C:/projects/pinns_local/code/')
else:
    # parameters for running on compute canada
    job_duration = timedelta(hours=duration_hours,minutes=0)
    end_time = start_time+job_duration
    
    useGPU=False
    HOMEDIR = '/home/coneill/sync/'
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    sys.path.append(HOMEDIR+'code/')

from pinns_galerkin_viv.lib.downsample import compute_downsample_inds

print("This job is: ",job_name)  

# set the paths
save_loc = HOMEDIR+'output/'+job_name+'_output/'
checkpoint_filepath = save_loc+'checkpoint'
physics_loss_coefficient = 1.0
# set number of cores to compute on 
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

if useGPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    # if we are on the cluster, we need to check we use the right number of gpu, else we should raise an error
    expected_GPU=4
    assert len(physical_devices)==expected_GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# read the data

base_dir = HOMEDIR+'/data/mazi_fixed_grid/'
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxppuxpp = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxppuypp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyppuypp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
x_test = x
y = np.array(configFile['X_vec'][1,:])
y_test = y
d = np.array(configFile['cylinderDiameter'])

# note that we need to get these before we downsample, otherwise we will have inconsistent 
# normalization depending on the supersampling factor which causes chaos elswhere
MAX_x = max(x.flatten())
MAX_y = max(y.flatten())
MAX_ux = max(ux.flatten())
MAX_uy = max(uy.flatten())
MIN_x = min(x.flatten())
MIN_y = min(y.flatten())
MIN_ux = min(ux.flatten())
MIN_uy = min(uy.flatten())
MAX_uxppuxpp = max(uxppuxpp.flatten())
MAX_uxppuypp = max(uxppuypp.flatten())
MAX_uyppuypp = max(uyppuypp.flatten())

# if we are downsampling and then upsampling, downsample the source data
if supersample_factor>1:
    n_x = np.array(configFile['x_grid']).size
    n_y = np.array(configFile['y_grid']).size
    downsample_inds = compute_downsample_inds(supersample_factor,n_x,n_y)
    x = x[downsample_inds]
    y = y[downsample_inds]
    ux = ux[downsample_inds]
    uy = uy[downsample_inds]
    uxppuxpp = uxppuxpp[downsample_inds]
    uxppuypp = uxppuypp[downsample_inds]
    uyppuypp = uyppuypp[downsample_inds]

print('u.shape: ',ux.shape)
print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)

nu_mol = 0.0066667
print('max_x: ',MAX_x)
print('min_x: ',MIN_x)
print('max_y: ',MAX_y)
print('min_y: ',MIN_y)



MAX_p= 1 # estimated maximum pressure, we should 

def colloc_points():
    # reduce the collocation points to 25k
    colloc_limits1 = np.array([[-2.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
    colloc_lhs1 = colloc_sample_lhs1(20000)


    colloc_limits2 = np.array([[-1.0,3.0],[-1.5,1.5]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(10000)

    colloc_merged = np.vstack((colloc_lhs1,colloc_lhs2))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    print(cylinder_inds.shape)
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)
    print('colloc_merged.shape',colloc_merged.shape)

    f_colloc_train = colloc_merged*np.array([1/MAX_x,1/MAX_y])
    return f_colloc_train

f_colloc_train = colloc_points()

# normalize the training data:
x_train = x/MAX_x
y_train = y/MAX_y
ux_train = ux/MAX_ux
uy_train = uy/MAX_uy
uxppuxpp_train = uxppuxpp/MAX_uxppuxpp
uxppuypp_train = uxppuypp/MAX_uxppuypp
uyppuypp_train = uyppuypp/MAX_uyppuypp

# boundary condition points

theta = np.linspace(0,2*np.pi,1000)
ns_BC_x = 0.5*d*np.cos(theta)/MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/MAX_y
ns_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1)))

p_BC_x = np.array([MAX_x,MAX_x])/MAX_x
p_BC_y = np.array([MIN_y,MAX_y])/MAX_y
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

# the order here must be identical to inside the cost functions
O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
X_test = np.hstack((x_test.reshape(-1,1)/MAX_x,y_test.reshape(-1,1)/MAX_y))

print('X_train.shape: ',X_train.shape)
print('X_train.shape: ',X_test.shape)
print('O_train.shape: ',O_train.shape)

  
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
    


@tf.function
def net_f_cartesian(colloc_tensor):
    
    up = model_mean(colloc_tensor)
    # knowns
    ux = up[:,0]*MAX_ux
    uy = up[:,1]*MAX_uy
    uxppuxpp = up[:,2]*MAX_uxppuxpp
    uxppuypp = up[:,3]*MAX_uxppuypp
    uyppuypp = up[:,4]*MAX_uyppuypp
    # unknowns
    p = up[:,5]*MAX_p
    w = 100.0/(1.0+tf.exp(2*2.71828-tf.abs(0.1*up[:,6])))+1.0 # self adaptive weight function must be minimum 1
    
    # compute the gradients of the quantities
    
    # ux gradient
    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/MAX_x
    ux_y = dux[:,1]/MAX_y
    # and second derivative
    ux_xx = tf.gradients(ux_x, colloc_tensor)[0][:,0]/MAX_x
    ux_yy = tf.gradients(ux_y, colloc_tensor)[0][:,1]/MAX_y
    
    # uy gradient
    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/MAX_x
    uy_y = duy[:,1]/MAX_y
    # and second derivative
    uy_xx = tf.gradients(uy_x, colloc_tensor)[0][:,0]/MAX_x
    uy_yy = tf.gradients(uy_y, colloc_tensor)[0][:,1]/MAX_y

    # gradient unmodeled reynolds stresses
    uxppuxpp_x = tf.gradients(uxppuxpp, colloc_tensor)[0][:,0]/MAX_x
    duxppuypp = tf.gradients(uxppuypp, colloc_tensor)[0]
    uxppuypp_x = duxppuypp[:,0]/MAX_x
    uxppuypp_y = duxppuypp[:,1]/MAX_y
    uyppuypp_y = tf.gradients(uyppuypp, colloc_tensor)[0][:,1]/MAX_y

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y


    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxppuxpp_x + uxppuypp_y) + p_x - (nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxppuypp_x + uyppuypp_y) + p_y - (nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    

    return f_x, f_y, f_mass, w

@tf.function
def BC_pressure(BC_points):
    up = model_mean(BC_points)
    # knowns
    # unknowns
    p = up[:,5]*MAX_p
    return tf.square(tf.reduce_mean(p))

def BC_no_slip(BC_points):
    up = model_mean(BC_points)
    # knowns
    ux = up[:,0]*MAX_ux
    uy = up[:,1]*MAX_uy
    uxppuxpp = up[:,2]*MAX_uxppuxpp
    uxppuypp = up[:,3]*MAX_uxppuypp
    uyppuypp = up[:,4]*MAX_uyppuypp
    return tf.reduce_mean(tf.square(ux)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))


# function wrapper, combine data and physics loss
def mean_loss_wrapper(colloc_tensor_f,BC_ns,BC_p): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second

        if physics_loss_coefficient!=0:
            mx,my,mass,w_phys = net_f_cartesian(colloc_tensor_f)
            physical_loss1 = tf.reduce_mean(w_phys*tf.square(mx))
            physical_loss2 = tf.reduce_mean(w_phys*tf.square(my))
            physical_loss3 = tf.reduce_mean(w_phys*tf.square(mass))

            BC_pressure_loss = BC_pressure(BC_p)
            BC_no_slip_loss = BC_no_slip(BC_ns)
            
            w_data = 100.0/(1.0+tf.exp(2*2.71828-tf.abs(0.1*y_pred[:,6])))+1.0  # the weight must be minimum 1

            data_loss_ux = tf.reduce_mean(w_data*tf.square(y_true[:,0]-y_pred[:,0])) # u 
            data_loss_uy = tf.reduce_mean(w_data*tf.square(y_true[:,1]-y_pred[:,1])) # v 
            data_loss_uxppuxpp = tf.reduce_mean(w_data*tf.square(y_true[:,2]-y_pred[:,2])) # u'u' 
            data_loss_uxppuypp = tf.reduce_mean(w_data*tf.square(y_true[:,3]-y_pred[:,3])) # u'v' 
            data_loss_uyppuypp = tf.reduce_mean(w_data*tf.square(y_true[:,4]-y_pred[:,4])) # v'v'
            data_loss = data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp

            adaptive_loss = 1.0/tf.reduce_mean(w_phys)+1.0/tf.reduce_mean(w_data)+1.0/tf.reduce_min(w_phys)+1.0/tf.reduce_min(w_data) # SA pinn adaptive weight function loss

            return  adaptive_loss + data_loss + physics_loss_coefficient*(physical_loss1 + physical_loss2 + physical_loss3 + BC_pressure_loss + BC_no_slip_loss) # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 
        else:

            data_loss_ux = tf.reduce_mean(tf.square(y_true[:,0]-y_pred[:,0])) # u 
            data_loss_uy = tf.reduce_mean(tf.square(y_true[:,1]-y_pred[:,1])) # v 
            data_loss_uxppuxpp = tf.reduce_mean(tf.square(y_true[:,2]-y_pred[:,2])) # u'u' 
            data_loss_uxppuypp = tf.reduce_mean(tf.square(y_true[:,3]-y_pred[:,3])) # u'v' 
            data_loss_uyppuypp = tf.reduce_mean(tf.square(y_true[:,4]-y_pred[:,4])) # v'v' 
            data_loss = data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp

            return data_loss

    return mean_loss




model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))

# this time we randomly shuffle the order of X and O
rng = np.random.default_rng()

# job_name = 'RS3_CC0001_23h'
# we need to check if there are already checkpoints for this job
checkpoint_files = get_filepaths_with_glob(HOMEDIR+'/output/'+job_name+'_output/',job_name+'_ep*_model.h5')
if len(checkpoint_files)>0:
    files_epoch_number = np.zeros([len(checkpoint_files),1],dtype=np.uint)
    # if there are checkpoints, train based on the most recent checkpoint
    for f_indx in range(len(checkpoint_files)):
        re_result = re.search("ep[0-9]*",checkpoint_files[f_indx])
        files_epoch_number[f_indx]=int(checkpoint_files[f_indx][(re_result.start()+2):re_result.end()])
    epochs = np.uint(np.max(files_epoch_number))
    print(HOMEDIR+'/output/'+job_name+'_output/',job_name+'_ep'+str(epochs))
    model_mean = keras.models.load_model(HOMEDIR+'/output/'+job_name+'_output/'+job_name+'_ep'+str(epochs)+'_model.h5',custom_objects={'mean_loss':mean_loss_wrapper(f_colloc_train,ns_BC_vec,p_BC_vec),'QresBlock':QresBlock})
    model_mean.summary()
else:
    # if not, we train from the beginning
    epochs = 0
    if (not os.path.isdir(HOMEDIR+'output/'+job_name+'_output/')):
        os.mkdir(HOMEDIR+'/output/'+job_name+'_output/')

    # create NN
    if useGPU:
        tf_device_string = ['GPU:0']
        for ngpu in range(1,len(physical_devices)):
            tf_device_string.append('GPU:'+str(ngpu))

        strategy = tf.distribute.MirroredStrategy(devices=tf_device_string)
        print('Using devices: ',tf_device_string)
        with strategy.scope():
            model_mean = keras.Sequential()
            model_mean.add(keras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
            for i in range(dense_layers-1):
                model_mean.add(keras.layers.Dense(dense_nodes,activation='tanh'))
            model_mean.add(keras.layers.Dense(7,activation='linear'))
            model_mean.summary()
            model_mean.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = mean_loss_wrapper(tf.cast(f_colloc_train,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)
    else:
        tf_device_string = '/CPU:0'

        with tf.device(tf_device_string):
            model_mean = keras.Sequential()
            model_mean.add(keras.layers.Dense(dense_nodes, activation='tanh', input_shape=(2,)))
            for i in range(dense_layers-1):
                model_mean.add(keras.layers.Dense(dense_nodes,activation='tanh'))
            model_mean.add(keras.layers.Dense(7,activation='linear'))
            model_mean.summary()
            model_mean.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = mean_loss_wrapper(tf.cast(f_colloc_train,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)





# train the network
d_epochs = 1
X_train = tf.cast(X_train,dtype_train)
O_train = tf.cast(O_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs




if node_name ==LOCAL_NODE:
    job_duration = timedelta(hours=22,minutes=30)
    end_time = start_time+job_duration
    LBFGS_steps=333
    LBFGS_epoch = 1000   
else:
    LBFGS_steps=333
    LBFGS_epoch = 1000
 
    # local node training loop, save every epoch for testing

    
    
if True:
    # LBFGS training, compute canada
    from pinns_galerkin_viv.lib.LBFGS_example import function_factory
    import tensorflow_probability as tfp

    L_iter = 0
    f_colloc_train = colloc_points()
    func = function_factory(model_mean, mean_loss_wrapper(f_colloc_train,ns_BC_vec,p_BC_vec), X_train, O_train)
    init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables)
            
    while True:
        if physics_loss_coefficient!=0:
        # randomly select new collocation points every LGFBS step
            f_colloc_train = colloc_points()
            func = function_factory(model_mean, mean_loss_wrapper(f_colloc_train,ns_BC_vec,p_BC_vec), X_train, O_train)
            init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables)
            
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        epochs = epochs + LBFGS_epoch
        L_iter = L_iter+1
            
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
            
        if np.mod(L_iter,1)==0:
            model_mean.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_mean.predict(X_test,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close()
            if physics_loss_coefficient!=0:
                t_mx,t_my,t_mass,t_weight = net_f_cartesian(X_test)
                h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_error.mat','w')
                h5f.create_dataset('mxr',data=t_mx)
                h5f.create_dataset('myr',data=t_my)
                h5f.create_dataset('massr',data=t_mass)
                h5f.create_dataset('weight',data=t_weight)
                h5f.close()

        # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            # save the last epoch before exiting
            model_mean.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_mean.predict(X_test,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close()
            if physics_loss_coefficient!=0:
                t_mx,t_my,t_mass,t_weight = net_f_cartesian(X_test)
                h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_error.mat','w')
                h5f.create_dataset('mxr',data=t_mx)
                h5f.create_dataset('myr',data=t_my)
                h5f.create_dataset('massr',data=t_mass)
                h5f.create_dataset('weight',data=t_weight)
                h5f.close()
            exit()
        last_epoch_time = datetime.now()
  
