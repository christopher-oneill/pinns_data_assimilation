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
nlayers = int(sys.argv[3])
nnodes = int(sys.argv[4])
job_time = int(sys.argv[5])

global job_name
job_name = 'mfg_vdnn_mean{:03d}_S{:d}_L{:d}N{:d}'.format(job_number,supersample_factor,nlayers,nnodes)

job_duration = timedelta(hours=job_time,minutes=0)
end_time = start_time+job_duration

# declare global variables used in the saving function
global model_mean
global save_loc

global X_test
global epochs


LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_narval/sync/'
    HOMEDIR = 'C:/projects/pinns_narval/sync/'
    PROJECTDIR=HOMEDIR
    sys.path.append('C:/projects/pinns_local/code/')
else:
    # parameters for running on compute canada
    
    
    useGPU=False
    HOMEDIR = '/home/coneill/sync/'
    PROJECTDIR = '/home/coneill/projects/def-martinuz/coneill/'
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    sys.path.append(HOMEDIR+'code/')

from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center
from pinns_data_assimilation.lib.layers import ResidualLayer

print("This job is: ",job_name)  

# set the paths
save_loc = PROJECTDIR+'output/'+job_name+'_output/'
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
ux_test = 1.0*ux

uxppuxpp = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxppuypp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyppuypp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
x_test = x
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
y_test = y
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

# create additional points for the extrapolated region in front of the cylinder
x_large = np.linspace(-6,10,500)
y_large = np.linspace(-2.0,2.0,200)
x_grid_large, y_grid_large = np.meshgrid(x_large,y_large)

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
    downsample_inds, ndx,ndy = compute_downsample_inds_center(supersample_factor,X_grid[:,0],Y_grid[0,:].transpose())
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
    colloc_limits1 = np.array([[-6.0,10.0],[-2.0,2.0]])
    colloc_sample_lhs1 = LHS(xlimits=colloc_limits1)
    colloc_lhs1 = colloc_sample_lhs1(30000)


    colloc_limits2 = np.array([[-1.0,3.0],[-1.5,1.5]])
    colloc_sample_lhs2 = LHS(xlimits=colloc_limits2)
    colloc_lhs2 = colloc_sample_lhs2(10000)

    colloc_merged = np.vstack((np.stack((x,y),axis=1),colloc_lhs1,colloc_lhs2))


    # remove points inside the cylinder
    c1_loc = np.array([0,0],dtype=np.float64)
    cylinder_inds = np.less(np.power(np.power(colloc_merged[:,0]-c1_loc[0],2)+np.power(colloc_merged[:,1]-c1_loc[1],2),0.5*d),0.5)
    colloc_merged = np.delete(colloc_merged,cylinder_inds[0,:],axis=0)

    f_colloc_train = colloc_merged*np.array([1/MAX_x,1/MAX_x])
    return f_colloc_train

f_colloc_train = colloc_points()

class UserScalingParameters(object):
    pass

ScalingParameters = UserScalingParameters()
ScalingParameters.fs = 10.0
ScalingParameters.MAX_x = np.max(x.flatten())
ScalingParameters.MAX_y = ScalingParameters.MAX_x # we scale based on the largest spatial dimension
ScalingParameters.MAX_ux = MAX_ux # we scale based on the max of the whole output array
ScalingParameters.MAX_uy = MAX_uy
ScalingParameters.MIN_x = np.min(x.flatten())
ScalingParameters.MIN_y = np.min(y.flatten())
ScalingParameters.MIN_ux = np.min(ux.flatten())
ScalingParameters.MIN_uy = np.min(uy.flatten())
ScalingParameters.MAX_uxppuxpp = MAX_uxppuxpp
ScalingParameters.MAX_uxppuypp = MAX_uxppuypp
ScalingParameters.MAX_uyppuypp = MAX_uyppuypp
ScalingParameters.nu_mol = 0.0066667
ScalingParameters.MAX_p= MAX_p # estimated maximum pressure, we should
ScalingParameters.batch_size = f_colloc_train.shape[0]
ScalingParameters.physics_loss_coefficient = 1.0
ScalingParameters.boundary_loss_coefficient = 1.0
ScalingParameters.data_loss_coefficient = 1.0


# normalize the training data:
x_train = x/MAX_x
y_train = y/MAX_x
ux_train = ux/MAX_ux
uy_train = uy/MAX_uy
uxppuxpp_train = uxppuxpp/MAX_uxppuxpp
uxppuypp_train = uxppuypp/MAX_uxppuypp
uyppuypp_train = uyppuypp/MAX_uyppuypp

# boundary condition points

theta = np.linspace(0,2*np.pi,1000)
ns_BC_x = 0.5*d*np.cos(theta)/MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/MAX_x
ns_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1)))

p_BC_x = np.array([MAX_x,MAX_x])/MAX_x
p_BC_y = np.array([MIN_y,MAX_y])/MAX_x
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

inlet_BC_x = -6.0*np.ones([500,1])/MAX_x
inlet_BC_y = np.linspace(MIN_y,MAX_y,500)/MAX_x
inlet_BC_vec = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

inlet_BC_x = -2.0*np.ones([500,1])/MAX_x
inlet_BC_y = np.linspace(-2.0,2.0,500)/MAX_x
inlet_BC_vec2 = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

# the order here must be identical to inside the cost functions
O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
X_test = np.hstack((x_test.reshape(-1,1)/MAX_x,y_test.reshape(-1,1)/MAX_x))
X_large = np.hstack((x_grid_large.reshape(-1,1)/MAX_x,y_grid_large.reshape(-1,1)/MAX_x))

print('X_train.shape: ',X_train.shape)
print('X_train.shape: ',X_test.shape)
print('O_train.shape: ',O_train.shape)



from pinns_data_assimilation.lib.navier_stokes_cartesian import RANS_reynolds_stress_cartesian
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_pressure
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_reynolds_stress_no_slip
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet
from pinns_data_assimilation.lib.navier_stokes_cartesian import BC_RANS_inlet2

# we need to write the physics wrapper and save wrappers custom to this script

def RANS_reynolds_stress_loss_wrapper(model_RANS,colloc_points,BC_ns,BC_p,BC_inlet,BC_inlet2): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''
        
        data_loss = tf.reduce_mean(data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp)

        if (model_RANS.ScalingParameters.physics_loss_coefficient==0):
            physics_loss = 0.0
        else:
            if (model_RANS.ScalingParameters.batch_size==colloc_points.shape[0]):
                # all colloc points
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,colloc_points)
            else:
                # random selection of collocation points with batch size
                rand_colloc_points = np.random.choice(colloc_points.shape[0],model_RANS.ScalingParameters.batch_size)
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,tf.gather(colloc_points,rand_colloc_points))
            physical_loss1 = tf.reduce_mean(tf.square(mx))
            physical_loss2 = tf.reduce_mean(tf.square(my))
            physical_loss3 = tf.reduce_mean(tf.square(mass))
            physics_loss = tf.reduce_mean(physical_loss1 + physical_loss2 + physical_loss3)
        
        if (model_RANS.ScalingParameters.boundary_loss_coefficient==0):
            boundary_loss = 0.0
        else:
            BC_pressure_loss = BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) # scaled to compensate the reduce sum on other BCs
            BC_no_slip_loss = BC_RANS_reynolds_stress_no_slip(model_RANS,BC_ns)
            BC_inlet_loss = BC_RANS_inlet(model_RANS,BC_inlet)
            BC_inlet_loss2 = BC_RANS_inlet2(model_RANS,BC_inlet2)
                       
            boundary_loss = (BC_pressure_loss + BC_no_slip_loss + BC_inlet_loss + BC_inlet_loss2)

        return  model_RANS.ScalingParameters.data_loss_coefficient*data_loss + model_RANS.ScalingParameters.physics_loss_coefficient*physics_loss + model_RANS.ScalingParameters.boundary_loss_coefficient*boundary_loss
        
        
    return mean_loss

def save_custom():
    global model_mean
    global save_loc
    global job_name
    global X_test
    global epochs

    model_mean.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
    pred = model_mean.predict(X_test,batch_size=32)
    h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
    h5f.create_dataset('pred',data=pred)
    h5f.close()

    if physics_loss_coefficient!=0:
        t_mx,t_my,t_mass = RANS_reynolds_stress_cartesian(model_mean,X_test)
        h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_error.mat','w')
        h5f.create_dataset('mxr',data=t_mx)
        h5f.create_dataset('myr',data=t_my)
        h5f.create_dataset('massr',data=t_mass)
        h5f.close()







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
checkpoint_files = get_filepaths_with_glob(PROJECTDIR+'output/'+job_name+'_output/',job_name+'_ep*_model.h5')

if len(checkpoint_files)>0:
    files_epoch_number = np.zeros([len(checkpoint_files),1],dtype=np.uint)
    # if there are checkpoints, train based on the most recent checkpoint
    for f_indx in range(len(checkpoint_files)):
        re_result = re.search("ep[0-9]*",checkpoint_files[f_indx])
        files_epoch_number[f_indx]=int(checkpoint_files[f_indx][(re_result.start()+2):re_result.end()])
    epochs = np.uint(np.max(files_epoch_number))
    print(PROJECTDIR+'/output/'+job_name+'_output/',job_name+'_ep'+str(epochs))
    model_mean = keras.models.load_model(PROJECTDIR+'/output/'+job_name+'_output/'+job_name+'_ep'+str(epochs)+'_model.h5',custom_objects={'mean_loss':RANS_reynolds_stress_loss_wrapper(None,f_colloc_train,ns_BC_vec,p_BC_vec,inlet_BC_vec,inlet_BC_vec2),'ResidualLayer':ResidualLayer})
    # we need to compile again after loading once we can populate the loss function with the model object
    model_mean.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss = RANS_reynolds_stress_loss_wrapper(model_mean,tf.cast(f_colloc_train,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train),tf.cast(inlet_BC_vec,dtype_train),tf.cast(inlet_BC_vec2,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)
    model_mean.ScalingParameters = ScalingParameters
    model_mean.summary()
else:
    # if not, we train from the beginning
    epochs = 0
    if (not os.path.isdir(PROJECTDIR+'output/'+job_name+'_output/')):
        os.mkdir(PROJECTDIR+'/output/'+job_name+'_output/')

    # create NN
    dense_nodes = nnodes
    dense_layers = nlayers
    if useGPU:
        exit()        
    else:
        tf_device_string = '/CPU:0'

        with tf.device(tf_device_string):
            model_mean = keras.Sequential()
            model_mean.add(keras.layers.Dense(dense_nodes, activation='linear', input_shape=(2,)))
            model_mean.add(keras.layers.Dense(dense_nodes, activation='linear'))
            for i in range(dense_layers-2):
                model_mean.add(ResidualLayer(dense_nodes,activation='swish'))
            model_mean.add(keras.layers.Dense(6,activation='linear'))
            model_mean.summary()
            model_mean.ScalingParameters = ScalingParameters
            model_mean.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss = RANS_reynolds_stress_loss_wrapper(model_mean,tf.cast(f_colloc_train,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train),tf.cast(inlet_BC_vec,dtype_train),tf.cast(inlet_BC_vec2,dtype_train)),jit_compile=False) #(...,BC_points1,...,BC_points3)





# train the network
d_epochs = 100
X_train = tf.cast(X_train,dtype_train)
O_train = tf.cast(O_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs




if node_name ==LOCAL_NODE:
    LBFGS_steps=1
    LBFGS_epoch = 1000   
else:
    LBFGS_steps=333
    LBFGS_epoch = 1000
 
    # local node training loop, save every epoch for testing

model_mean.ScalingParameters.physics_loss_coefficient = 1.0
if True:
    # LBFGS training, compute canada
    from pinns_data_assimilation.lib.LBFGS_example import function_factory
    import tensorflow_probability as tfp

    L_iter = 0
    f_colloc_train = colloc_points()
    func = function_factory(model_mean, RANS_reynolds_stress_loss_wrapper(model_mean,f_colloc_train,ns_BC_vec,p_BC_vec,inlet_BC_vec,inlet_BC_vec2), X_train, O_train)
    init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables)
            
    while True:
        if model_mean.ScalingParameters.physics_loss_coefficient!=0:
        # randomly select new collocation points every LGFBS step
            f_colloc_train = colloc_points()
            func = function_factory(model_mean, RANS_reynolds_stress_loss_wrapper(model_mean,f_colloc_train,ns_BC_vec,p_BC_vec,inlet_BC_vec,inlet_BC_vec2), X_train, O_train)
            init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables)
            
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps,tolerance=1E-16)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_mean.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        epochs = epochs + LBFGS_epoch
        L_iter = L_iter+1
            
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
            
        if np.mod(L_iter,10)==0:
            save_custom()

        # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            save_custom()
            exit()
        last_epoch_time = datetime.now()
  
