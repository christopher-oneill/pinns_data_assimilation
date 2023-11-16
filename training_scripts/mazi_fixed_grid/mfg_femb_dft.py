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
import re
import smt
import h5py
from smt.sampling_methods import LHS
from pyDOE import lhs
from datetime import datetime
from datetime import timedelta
import platform
import sys

keras.backend.set_floatx('float64')
dtype_train = tf.float64

case = 'JFM'
start_time = datetime.now()
start_timestamp = datetime.strftime(start_time,'%Y%m%d%H%M%S')

node_name = platform.node()

PLOT = False


assert len(sys.argv)==7

mode_number = int(sys.argv[1])
supersample_factor = int(sys.argv[2])
job_number = int(sys.argv[3])
arg_nodes = int(sys.argv[4])
arg_layers = int(sys.argv[5])
job_hours = int(sys.argv[6])

job_name = 'mfg_femb_dft{:d}_S{:d}_j{:03d}'.format(mode_number,supersample_factor,job_number)


LOCAL_NODE = 'DESKTOP-AMLVDAF'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    job_duration = timedelta(hours=job_hours,minutes=0)
    end_time = start_time+job_duration
    useGPU=False    
    SLURM_TMPDIR='C:/projects/pinns_narval/sync/'
    
    HOMEDIR = 'C:/projects/pinns_narval/sync/'
    PROJECTDIR = HOMEDIR
    sys.path.append('C:/projects/pinns_local/code/')
    # set number of cores to compute on 
else:
    # parameters for running on compute canada
    job_duration = timedelta(hours=job_hours,minutes=0)
    end_time = start_time+job_duration
    print("This job is: ",job_name)
    useGPU=False
    HOMEDIR = '/home/coneill/sync/'
    PROJECTDIR = '/home/coneill/projects/def-martinuz/coneill/'
    SLURM_TMPDIR=os.environ["SLURM_TMPDIR"]
    sys.path.append(HOMEDIR+'code/')
    # set number of cores to compute on 

from pinns_galerkin_viv.lib.downsample import compute_downsample_inds
from pinns_galerkin_viv.lib.layers import ResidualLayer
from pinns_galerkin_viv.lib.layers import QresBlock
from pinns_galerkin_viv.lib.layers import FourierResidualLayer64
from pinns_galerkin_viv.lib.layers import ProductResidualLayer64

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)
    

# set the paths
save_loc = PROJECTDIR+'output/'+job_name+'_output/'
checkpoint_filepath = save_loc+'checkpoint'


if useGPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    # if we are on the cluster, we need to check we use the right number of gpu, else we should raise an error
    expected_GPU=4
    assert len(physical_devices)==expected_GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'

fourierModeFile = h5py.File(base_dir+'fourier_data_DFT.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')

reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxppuxpp = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxppuypp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyppuypp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

#psi_r = np.real(np.array(fourierModeFile['pressureModesShort'][mode_number,:])).transpose()
#psi_i = np.imag(np.array(fourierModeFile['pressureModesShort'][mode_number,:])).transpose()

tau_xx_r = np.array(fourierModeFile['stressModesShortReal'][0,mode_number,:]).transpose()
tau_xx_i = np.array(fourierModeFile['stressModesShortImag'][0,mode_number,:]).transpose()
tau_xy_r = np.array(fourierModeFile['stressModesShortReal'][1,mode_number,:]).transpose()
tau_xy_i = np.array(fourierModeFile['stressModesShortImag'][1,mode_number,:]).transpose()
tau_yy_r = np.array(fourierModeFile['stressModesShortReal'][2,mode_number,:]).transpose()
tau_yy_i = np.array(fourierModeFile['stressModesShortImag'][2,mode_number,:]).transpose()

fs = 10.0 #np.array(configFile['fs'])
omega = np.array(fourierModeFile['fShort'][0,mode_number])*2*np.pi


print(configFile['X_vec'].shape)
x = np.array(configFile['X_vec'][0,:])
x_test = x
y = np.array(configFile['X_vec'][1,:])
y_test = y
d = np.array(configFile['cylinderDiameter'])

# create a dummy object to contain all the scaling parameters
class UserScalingParameters(object):
    pass

MeanScalingParameters = UserScalingParameters()
MeanScalingParameters.fs = fs
MeanScalingParameters.MAX_x = np.max(x.flatten())
MeanScalingParameters.MAX_y = np.max(y.flatten())
MeanScalingParameters.MAX_ux = np.max(ux.flatten())
MeanScalingParameters.MAX_uy = np.max(uy.flatten())
MeanScalingParameters.MIN_x = np.min(x.flatten())
MeanScalingParameters.MIN_y = np.min(y.flatten())
MeanScalingParameters.MIN_ux = np.min(ux.flatten())
MeanScalingParameters.MIN_uy = np.min(uy.flatten())
MeanScalingParameters.MAX_uxppuxpp = np.max(uxppuxpp.flatten())
MeanScalingParameters.MAX_uxppuypp = np.max(uxppuypp.flatten())
MeanScalingParameters.MAX_uyppuypp = np.max(uyppuypp.flatten())

FourierScalingParameters = UserScalingParameters()
FourierScalingParameters.physics_loss_coefficient = 0.0
FourierScalingParameters.omega = omega
FourierScalingParameters.MAX_x = np.max(x.flatten())
FourierScalingParameters.MAX_y = np.max(y.flatten())
FourierScalingParameters.MIN_x = np.min(x.flatten())
FourierScalingParameters.MIN_y = np.min(y.flatten())
FourierScalingParameters.MAX_phi_xr = np.max(phi_xr.flatten())
FourierScalingParameters.MAX_phi_xi = np.max(phi_xi.flatten())
FourierScalingParameters.MAX_phi_yr = np.max(phi_yr.flatten())
FourierScalingParameters.MAX_phi_yi = np.max(phi_yi.flatten())
FourierScalingParameters.MAX_tau_xx_r = np.max(tau_xx_r.flatten())
FourierScalingParameters.MAX_tau_xx_i = np.max(tau_xx_i.flatten())
FourierScalingParameters.MAX_tau_xy_r = np.max(tau_xy_r.flatten())
FourierScalingParameters.MAX_tau_xy_i = np.max(tau_xy_i.flatten())
FourierScalingParameters.MAX_tau_yy_r = np.max(tau_yy_r.flatten())
FourierScalingParameters.MAX_tau_yy_i = np.max(tau_yy_i.flatten())

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
    phi_xr = phi_xr[downsample_inds]
    phi_xi = phi_xi[downsample_inds]
    phi_yr = phi_yr[downsample_inds]
    phi_yi = phi_yi[downsample_inds]
    tau_xx_r = tau_xx_r[downsample_inds]
    tau_xx_i = tau_xx_i[downsample_inds]
    tau_xy_r = tau_xy_r[downsample_inds]
    tau_xy_i = tau_xy_i[downsample_inds]
    tau_yy_r = tau_yy_r[downsample_inds]
    tau_yy_i = tau_yy_i[downsample_inds]



MeanScalingParameters.nu_mol = 0.0066667
FourierScalingParameters.nu_mol = 0.0066667



print('max_x: ',MeanScalingParameters.MAX_x)
print('min_x: ',MeanScalingParameters.MIN_x)
print('max_y: ',MeanScalingParameters.MAX_y)
print('min_y: ',MeanScalingParameters.MIN_y)

MeanScalingParameters.MAX_p= 1 # estimated maximum pressure, we should 
FourierScalingParameters.MAX_psi= 0.1 # chosen based on abs(max(psi))

def colloc_points():
    # reduce the collocation points to 25k
    colloc_limits1 = np.array([[-6.0,10.0],[-2.0,2.0]])
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

    f_colloc_train = colloc_merged*np.array([1/MeanScalingParameters.MAX_x,1/MeanScalingParameters.MAX_y])
    return f_colloc_train

f_colloc_train = colloc_points()

# normalize the training data:
x_train = x/MeanScalingParameters.MAX_x
y_train = y/MeanScalingParameters.MAX_y
ux_train = ux/MeanScalingParameters.MAX_ux
uy_train = uy/MeanScalingParameters.MAX_uy
uxppuxpp_train = uxppuxpp/MeanScalingParameters.MAX_uxppuxpp
uxppuypp_train = uxppuypp/MeanScalingParameters.MAX_uxppuypp
uyppuypp_train = uyppuypp/MeanScalingParameters.MAX_uyppuypp
phi_xr_train = phi_xr/FourierScalingParameters.MAX_phi_xr
phi_xi_train = phi_xi/FourierScalingParameters.MAX_phi_xi
phi_yr_train = phi_yr/FourierScalingParameters.MAX_phi_yr
phi_yi_train = phi_yi/FourierScalingParameters.MAX_phi_yi

tau_xx_r_train = tau_xx_r/FourierScalingParameters.MAX_tau_xx_r
tau_xx_i_train = tau_xx_i/FourierScalingParameters.MAX_tau_xx_i
tau_xy_r_train = tau_xy_r/FourierScalingParameters.MAX_tau_xy_r
tau_xy_i_train = tau_xy_i/FourierScalingParameters.MAX_tau_xy_i
tau_yy_r_train = tau_yy_r/FourierScalingParameters.MAX_tau_yy_r
tau_yy_i_train = tau_yy_i/FourierScalingParameters.MAX_tau_yy_i

# the order here must be identical to inside the cost functions
O_train = np.hstack(((ux_train).reshape(-1,1),(uy_train).reshape(-1,1),(uxppuxpp_train).reshape(-1,1),(uxppuypp_train).reshape(-1,1),(uyppuypp_train).reshape(-1,1),)) # training data
F_train = np.hstack(((phi_xr_train).reshape(-1,1),(phi_xi_train).reshape(-1,1),(phi_yr_train).reshape(-1,1),(phi_yi_train).reshape(-1,1),(tau_xx_r_train).reshape(-1,1),(tau_xx_i_train).reshape(-1,1),(tau_xy_r_train).reshape(-1,1),(tau_xy_i_train).reshape(-1,1),(tau_yy_r_train).reshape(-1,1),(tau_yy_i_train).reshape(-1,1))) # training data
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))
X_test = np.hstack((x_test.reshape(-1,1)/FourierScalingParameters.MAX_x,y_test.reshape(-1,1)/FourierScalingParameters.MAX_y))
# the order here must be identical to inside the cost functions


# boundary condition points
theta = np.linspace(0,2*np.pi,1000)
ns_BC_x = 0.5*d*np.cos(theta)/FourierScalingParameters.MAX_x # we beed to normalize the boundary conditions as well
ns_BC_y = 0.5*d*np.sin(theta)/FourierScalingParameters.MAX_y
ns_BC_vec = np.hstack((ns_BC_x.reshape(-1,1),ns_BC_y.reshape(-1,1)))

p_BC_x = np.array([FourierScalingParameters.MAX_x,FourierScalingParameters.MAX_x])/FourierScalingParameters.MAX_x
p_BC_y = np.array([FourierScalingParameters.MIN_y,FourierScalingParameters.MAX_y])/FourierScalingParameters.MAX_y
p_BC_vec = np.hstack((p_BC_x.reshape(-1,1),p_BC_y.reshape(-1,1)))

inlet_BC_x = -6.0*np.ones([500,1])/FourierScalingParameters.MAX_x
inlet_BC_y = np.linspace(FourierScalingParameters.MIN_y,FourierScalingParameters.MAX_y,500)/FourierScalingParameters.MAX_y
inlet_BC_vec = np.hstack((inlet_BC_x.reshape(-1,1),inlet_BC_y.reshape(-1,1)))

print('X_train.shape: ',X_train.shape)
print('O_train.shape: ',O_train.shape)



# import the physics

# mean model functions
from pinns_galerkin_viv.lib.physics_cartesian import net_f_mean_cartesian
from pinns_galerkin_viv.lib.physics_cartesian import mean_cartesian
from pinns_galerkin_viv.lib.physics_cartesian import mean_loss_wrapper
# mean model BCs
from pinns_galerkin_viv.lib.physics_cartesian import BC_mean_pressure
from pinns_galerkin_viv.lib.physics_cartesian import BC_mean_no_slip



# fourier model functions
from pinns_galerkin_viv.lib.physics_cartesian import net_f_fourier_cartesian
from pinns_galerkin_viv.lib.physics_cartesian import net_f_fourier_cartesian_batch
from pinns_galerkin_viv.lib.physics_cartesian import fourier_loss_wrapper
# fourier model BCs
# mean model BCs
from pinns_galerkin_viv.lib.physics_cartesian import BC_fourier_pressure_outlet
from pinns_galerkin_viv.lib.physics_cartesian import BC_fourier_no_slip
from pinns_galerkin_viv.lib.physics_cartesian import BC_fourier_inlet

# create the NNs

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))


# load the saved mean model
global model_mean
with tf.device('/CPU:0'):
    if (supersample_factor == 1):
        model_mean = keras.models.load_model(PROJECTDIR+'/output/mfg_mean008_output/mfg_mean008_ep54000_model.h5',custom_objects={'mean_loss':mean_loss_wrapper(None,f_colloc_train,ns_BC_vec,p_BC_vec),'QresBlock':QresBlock})
    elif (supersample_factor == 8):
        model_mean = keras.models.load_model(PROJECTDIR+'/output/mfg_mean008_output/mfg_mean008_ep54000_model.h5',custom_objects={'mean_loss':mean_loss_wrapper(None,f_colloc_train,ns_BC_vec,p_BC_vec),'QresBlock':QresBlock})
    model_mean.trainable=False

# append the scaling parameters to the model
model_mean.MeanScalingParameters = MeanScalingParameters

# get the values for the mean_data tensor
mean_data = mean_cartesian(model_mean,f_colloc_train)
mean_data_test = mean_cartesian(model_mean,X_train)
mean_data_test_grid = mean_cartesian(model_mean,X_test)

# clear the session, we will now create the fourier model
# tf.keras.backend.clear_session()

# check if the model has been created before, if so load it
def get_filepaths_with_glob(root_path: str, file_regex: str):
    return glob.glob(os.path.join(root_path, file_regex))
# we need to check if there are already checkpoints for this job
checkpoint_files = get_filepaths_with_glob(PROJECTDIR+'output/'+job_name+'_output/',job_name+'_ep*_model.h5')
if len(checkpoint_files)>0:
    files_epoch_number = np.zeros([len(checkpoint_files),1],dtype=np.uint)
    # if there are checkpoints, train based on the most recent checkpoint
    for f_indx in range(len(checkpoint_files)):
        re_result = re.search("ep[0-9]*",checkpoint_files[f_indx])
        files_epoch_number[f_indx]=int(checkpoint_files[f_indx][(re_result.start()+2):re_result.end()])
    epochs = np.uint(np.max(files_epoch_number))

        
    print(PROJECTDIR+'output/'+job_name+'_output/',job_name+'_ep'+str(epochs))
    model_fourier = keras.models.load_model(PROJECTDIR+'output/'+job_name+'_output/'+job_name+'_ep'+str(epochs)+'_model.h5',custom_objects={'custom_loss':fourier_loss_wrapper(None,f_colloc_train,mean_data,ns_BC_vec,p_BC_vec,inlet_BC_vec),'FourierResidualLayer64':FourierResidualLayer64,'ProductResidualLayer64':ProductResidualLayer64,'ResidualLayer':ResidualLayer})
    model_fourier.FourierScalingParameters = FourierScalingParameters # assign the scaling parameters
    model_fourier.compile(optimizer=keras.optimizers.SGD(learning_rate=1E-12), loss = fourier_loss_wrapper(model_fourier,tf.cast(f_colloc_train,dtype_train),tf.cast(mean_data,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train),tf.cast(inlet_BC_vec,dtype_train)),jit_compile=False)
    model_fourier.summary()
else:
    # if not, we train from the beginning
    epochs = 0
    FourierScalingParameters.physics_loss_coefficient = 0.0
    if (not os.path.isdir(PROJECTDIR+'output/'+job_name+'_output/')):
        os.mkdir(PROJECTDIR+'output/'+job_name+'_output/')

    # create the fourier model
    fourier_nodes = arg_nodes
    fourier_layers = arg_layers
    if useGPU:
        tf_device_string = ['GPU:0']
        for ngpu in range(1,len(physical_devices)):
            tf_device_string.append('GPU:'+str(ngpu))
                
        strategy = tf.distribute.MirroredStrategy(devices=tf_device_string)
        print('Using devices: ',tf_device_string)
        with strategy.scope():
            model_fourier = keras.Sequential()
            model_fourier.add(keras.layers.Dense(fourier_nodes, activation='linear', input_shape=(2,)))
            for i in range(fourier_layers-5):
                model_fourier.add(ResidualLayer(fourier_nodes))
            model_fourier.add(FourierResidualLayer64(fourier_nodes))
            model_fourier.add(ProductResidualLayer64(fourier_nodes))
            model_fourier.add(ResidualLayer(fourier_nodes))
            model_fourier.add(ProductResidualLayer64(fourier_nodes))
            model_fourier.add(ResidualLayer(fourier_nodes))    
            model_fourier.add(keras.layers.Dense(12,activation='linear'))
            model_fourier.FourierScalingParameters = FourierScalingParameters # assign the scaling parameters
            model_fourier.summary()
            model_fourier.compile(optimizer=keras.optimizers.SGD(learning_rate=1E-12), loss = fourier_loss_wrapper(model_fourier,tf.cast(f_colloc_train,dtype_train),tf.cast(mean_data,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train),tf.cast(inlet_BC_vec,dtype_train)),jit_compile=False) 
    else:
        with tf.device('/CPU:0'):
            model_fourier = keras.Sequential()
            model_fourier.add(keras.layers.Dense(fourier_nodes, activation='linear', input_shape=(2,)))
            for i in range(fourier_layers-5):
                model_fourier.add(ResidualLayer(fourier_nodes))
            model_fourier.add(FourierResidualLayer64(fourier_nodes))
            model_fourier.add(ProductResidualLayer64(fourier_nodes))
            model_fourier.add(ResidualLayer(fourier_nodes))
            model_fourier.add(ProductResidualLayer64(fourier_nodes))
            model_fourier.add(ResidualLayer(fourier_nodes))    
            model_fourier.add(keras.layers.Dense(12,activation='linear'))
            model_fourier.FourierScalingParameters = FourierScalingParameters # assign the scaling parameters
            model_fourier.summary()
            model_fourier.compile(optimizer=keras.optimizers.SGD(learning_rate=1E-12), loss = fourier_loss_wrapper(model_fourier,tf.cast(f_colloc_train,dtype_train),tf.cast(mean_data,dtype_train),tf.cast(ns_BC_vec,dtype_train),tf.cast(p_BC_vec,dtype_train),tf.cast(inlet_BC_vec,dtype_train)),jit_compile=False) 

# set the training call back
fourier_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
fourier_early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=500)

# setup the training data
# this time we randomly shuffle the order of X and O
rng = np.random.default_rng()
# train the network
d_epochs = 1
X_train = tf.cast(X_train,dtype_train)
F_train = tf.cast(F_train,dtype_train)
last_epoch_time = datetime.now()
average_epoch_time=timedelta(minutes=10)
start_epochs = epochs

shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
temp_X_train = X_train[shuffle_inds,:]
temp_Y_train = F_train[shuffle_inds,:]



       # compute canada LGFBS loop
BACKPROP_flag = False
if BACKPROP_flag:
    model_fourier.FourierScalingParameters.batch_size=32
    
    # compute canada training loop; use time based training
    while BACKPROP_flag:
        if np.mod(epochs,10)==0:
            shuffle_inds = rng.shuffle(np.arange(0,X_train.shape[1]))
            temp_X_train = X_train[shuffle_inds,:]
            temp_Y_train = F_train[shuffle_inds,:]
        hist = model_fourier.fit(temp_X_train[0,:,:],temp_Y_train[0,:,:], batch_size=model_fourier.FourierScalingParameters.batch_size, epochs=d_epochs, callbacks=[fourier_early_stop_callback,fourier_checkpoint_callback])
        epochs = epochs+d_epochs

        if epochs>500:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-15)
        if epochs>1000:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-16)
        if epochs>1500:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-17)
        if epochs>2000:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-18)
        if epochs>2500:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-19)

        if epochs>3000:
            model_fourier.FourierScalingParameters.physics_loss_coefficient=1.0
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-15)

        if epochs>3500:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-16)
        if epochs>4000:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-17)
        if epochs>4500:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-18)
        if epochs>5000:
            keras.backend.set_value(model_fourier.optimizer.learning_rate, 1E-19)
        if epochs>5500:
            BACKPROP_flag=False

        if np.mod(epochs,500)==0:
            # save every 10th epoch
            model_fourier.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_fourier.predict(X_train,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close()
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_error.mat','w')
            f_xr, f_xi, f_yr, f_yi, f_mr, f_mi = net_f_fourier_cartesian_batch(model_fourier,X_test,mean_data_test_grid,512)
            h5f.create_dataset('mxr_grid',data=f_xr)
            h5f.create_dataset('mxi_grid',data=f_xi)
            h5f.create_dataset('myr_grid',data=f_yr)
            h5f.create_dataset('myi_grid',data=f_yi)
            h5f.create_dataset('massr_grid',data=f_mr)
            h5f.create_dataset('massi_grid',data=f_mi)
            h5f.close()


        # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            # save the last epoch before exiting
            model_fourier.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_fourier.predict(X_train,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.close()
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_error.mat','w')
            f_xr, f_xi, f_yr, f_yi, f_mr, f_mi = net_f_fourier_cartesian_batch(model_fourier,X_test,mean_data_test_grid,256)
            h5f.create_dataset('mxr_grid',data=f_xr)
            h5f.create_dataset('mxi_grid',data=f_xi)
            h5f.create_dataset('myr_grid',data=f_yr)
            h5f.create_dataset('myi_grid',data=f_yi)
            h5f.create_dataset('massr_grid',data=f_mr)
            h5f.create_dataset('massi_grid',data=f_mi)
            h5f.close()
            exit()
            
        last_epoch_time = datetime.now()



LBFGS_steps=333
LBFGS_epoch = 1000


from pinns_galerkin_viv.lib.LBFGS_example import function_factory
import tensorflow_probability as tfp

L_iter = 0
func = function_factory(model_fourier, fourier_loss_wrapper(model_fourier,f_colloc_train,mean_data,ns_BC_vec,p_BC_vec,inlet_BC_vec), X_train, F_train)
init_params = tf.dynamic_stitch(func.idx, model_fourier.trainable_variables)

if BACKPROP_flag==False:
    while True:
        if model_fourier.FourierScalingParameters.physics_loss_coefficient!=0:
            # each L iter, we randomize the colocation points for robustness
            f_colloc_train = colloc_points()
            # get the values for the mean_data tensor
            
            mean_data = mean_cartesian(model_mean,f_colloc_train)
            func = function_factory(model_fourier, fourier_loss_wrapper(model_fourier,f_colloc_train,mean_data,ns_BC_vec,p_BC_vec,inlet_BC_vec), X_train, F_train)
            init_params = tf.dynamic_stitch(func.idx, model_fourier.trainable_variables)
            
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_steps)
        func.assign_new_model_parameters(results.position)
        init_params = tf.dynamic_stitch(func.idx, model_fourier.trainable_variables) # we need to reasign the parameters otherwise we start from the beginning each time
        epochs = epochs +LBFGS_epoch
        L_iter = L_iter+1
                
            # after training, the final optimized parameters are still in results.position
            # so we have to manually put them back to the model
                
        if np.mod(L_iter,10)==0:
            model_fourier.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_fourier.predict(X_train,batch_size=32)
            pred_grid = model_fourier.predict(X_test,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.create_dataset('pred_grid',data=pred_grid)
            h5f.close()
            if model_fourier.FourierScalingParameters.physics_loss_coefficient!=0:
                mxr_grid,mxi_grid,myr_grid,myi_grid,massr_grid,massi_grid = net_f_fourier_cartesian_batch(model_fourier,X_test,mean_data_test_grid,256)
                h5f.create_dataset('mxr_grid',data=mxr_grid)
                h5f.create_dataset('mxi_grid',data=mxi_grid)
                h5f.create_dataset('myr_grid',data=myr_grid)
                h5f.create_dataset('myi_grid',data=myi_grid)
                h5f.create_dataset('massr_grid',data=massr_grid)
                h5f.create_dataset('massi_grid',data=massi_grid)
                h5f.close()

                # check if we should exit
        average_epoch_time = (average_epoch_time+(datetime.now()-last_epoch_time))/2
        if (datetime.now()+average_epoch_time)>end_time:
            # if there is not enough time to complete the next epoch, exit
            print("Remaining time is insufficient for another epoch, exiting...")
            model_fourier.save(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_model.h5')
            pred = model_fourier.predict(X_train,batch_size=32)
            pred_grid = model_fourier.predict(X_test,batch_size=32)
            h5f = h5py.File(save_loc+job_name+'_ep'+str(np.uint(epochs))+'_pred.mat','w')
            h5f.create_dataset('pred',data=pred)
            h5f.create_dataset('pred_grid',data=pred_grid)
            h5f.close()
            if model_fourier.FourierScalingParameters.physics_loss_coefficient!=0:
                mxr_grid,mxi_grid,myr_grid,myi_grid,massr_grid,massi_grid = net_f_fourier_cartesian_batch(model_fourier,X_test,mean_data_test_grid,1024)
                h5f.create_dataset('mxr_grid',data=mxr_grid)
                h5f.create_dataset('mxi_grid',data=mxi_grid)
                h5f.create_dataset('myr_grid',data=myr_grid)
                h5f.create_dataset('myi_grid',data=myi_grid)
                h5f.create_dataset('massr_grid',data=massr_grid)
                h5f.create_dataset('massi_grid',data=massi_grid)
                h5f.close()
            exit()
        last_epoch_time = datetime.now()



   
   
   
