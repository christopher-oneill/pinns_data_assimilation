
import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp

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





# functions
def find_highest_numbered_file(path_prefix, number_pattern, suffix):
    # Get the directory path and file prefix
    directory, file_prefix = os.path.split(path_prefix)
    
    # Compile the regular expression pattern
    pattern = re.compile(f'{file_prefix}({number_pattern}){suffix}')
    
    # Initialize variables to track the highest number and file path
    highest_number = 0
    highest_file_path = None
    
    # Iterate over the files in the directory
    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            file_number = int(match.group(1))
            if file_number > highest_number:
                highest_number = file_number
                highest_file_path = os.path.join(directory, file)
    
    return highest_file_path, highest_number

def extract_matching_integers(prefix, number_pattern, suffix):
    # Get the directory path and file name prefix
    directory, file_prefix = os.path.split(prefix)
    
    # Compile the regular expression pattern
    pattern = re.compile(f'{file_prefix}({number_pattern}){suffix}')
    
    # Initialize a NumPy array to store the matching integers
    matching_integers = np.array([], dtype=int)
    
    # Iterate over the files in the directory

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            matching_integers = np.append(matching_integers, int(match.group(1)))
    
    return matching_integers

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dummy_loss():
    return 1.0

@tf.function
def gradients_mean(colloc_tensor):
    
    up = model_mean(colloc_tensor)
    # knowns
    ux = up[:,0]*MAX_ux
    uy = up[:,1]*MAX_uy
    # unknowns
    p = up[:,5]*MAX_p
    
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

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y

    return ux, ux_x, ux_y, ux_xx, ux_yy, uy_x, uy,  uy_y, uy_xx, uy_yy, p, p_x, p_y


@tf.function
def gradients_fourier(model,colloc_tensor):
  
    up = model(colloc_tensor)
    # velocity fourier coefficients
    phi_xr = up[:,0]*MAX_phi_xr
    phi_xi = up[:,1]*MAX_phi_xi
    phi_yr = up[:,2]*MAX_phi_yr
    phi_yi = up[:,3]*MAX_phi_yi

    # unknowns, pressure fourier modes
    psi_r = up[:,10]*MAX_psi
    psi_i = up[:,11]*MAX_psi
    # compute the gradients of the quantities
    
    # phi_xr gradient
    dphi_xr = tf.gradients(phi_xr, colloc_tensor)[0]
    phi_xr_x = dphi_xr[:,0]/MAX_x
    phi_xr_y = dphi_xr[:,1]/MAX_y
    # and second derivative
    phi_xr_xx = tf.gradients(phi_xr_x, colloc_tensor)[0][:,0]/MAX_x
    phi_xr_yy = tf.gradients(phi_xr_y, colloc_tensor)[0][:,1]/MAX_y

    # phi_xi gradient
    dphi_xi = tf.gradients(phi_xi, colloc_tensor)[0]
    phi_xi_x = dphi_xi[:,0]/MAX_x
    phi_xi_y = dphi_xi[:,1]/MAX_y
    # and second derivative
    phi_xi_xx = tf.gradients(phi_xi_x, colloc_tensor)[0][:,0]/MAX_x
    phi_xi_yy = tf.gradients(phi_xi_y, colloc_tensor)[0][:,1]/MAX_y

    # phi_yr gradient
    dphi_yr = tf.gradients(phi_yr, colloc_tensor)[0]
    phi_yr_x = dphi_yr[:,0]/MAX_x
    phi_yr_y = dphi_yr[:,1]/MAX_y
    # and second derivative
    phi_yr_xx = tf.gradients(phi_yr_x, colloc_tensor)[0][:,0]/MAX_x
    phi_yr_yy = tf.gradients(phi_yr_y, colloc_tensor)[0][:,1]/MAX_y
    
    # phi_yi gradient
    dphi_yi = tf.gradients(phi_yi, colloc_tensor)[0]
    phi_yi_x = dphi_yi[:,0]/MAX_x
    phi_yi_y = dphi_yi[:,1]/MAX_y
    # and second derivative
    phi_yi_xx = tf.gradients(phi_yi_x, colloc_tensor)[0][:,0]/MAX_x
    phi_yi_yy = tf.gradients(phi_yi_y, colloc_tensor)[0][:,1]/MAX_y

    # pressure gradients
    dpsi_r = tf.gradients(psi_r, colloc_tensor)[0]
    psi_r_x = dpsi_r[:,0]/MAX_x
    psi_r_y = dpsi_r[:,1]/MAX_y
    dpsi_i = tf.gradients(psi_i, colloc_tensor)[0]
    psi_i_x = dpsi_i[:,0]/MAX_x
    psi_i_y = dpsi_i[:,1]/MAX_y


    return phi_xr, phi_xr_x, phi_xr_y, phi_xr_xx, phi_xr_yy, phi_xi, phi_xi_x, phi_xi_y, phi_xi_xx, phi_xi_yy, phi_yr, phi_yr_x, phi_yr_y, phi_yr_xx, phi_yr_yy, phi_yi, phi_yi_x, phi_yi_y, phi_yi_xx, phi_yi_yy, psi_r, psi_r_x, psi_r_y, psi_i, psi_i_x, psi_i_y 

def gradients_mean_fd(X_grid,Y_grid, ux, uy, p):
    ux_grid = np.reshape(ux,X_grid.shape)
    uy_grid = np.reshape(uy,X_grid.shape)
    p_grid = np.reshape(p,X_grid.shape)

    ux_x = np.gradient(ux_grid,X_grid,axis=0)
    ux_y = np.gradient(ux_grid,Y_grid,axis=1)
    ux_xx = np.gradient(ux_x,X_grid,axis=0)
    ux_yy = np.gradient(ux_y,Y_grid,axis=1)

    uy_x = np.gradient(uy_grid,X_grid,axis=0)
    uy_y = np.gradient(uy_grid,Y_grid,axis=1)
    uy_xx = np.gradient(uy_x,X_grid,axis=0)
    uy_yy = np.gradient(uy_y,Y_grid,axis=1)

    p_x = np.gradient(p_grid,X_grid,axis=0)
    p_y = np.gradient(p_grid,Y_grid,axis=1)  

    return np.reshape(ux_x,[X_grid.size,1]), np.reshape(ux_y,[X_grid.size,1]), np.reshape(ux_xx,[X_grid.size,1]), np.reshape(ux_yy,[X_grid.size,1]), np.reshape(uy_x,[X_grid.size,1]), np.reshape(uy_y,[X_grid.size,1]), np.reshape(uy_xx,[X_grid.size,1]), np.reshape(uy_yy,[X_grid.size,1]), np.reshape(p_x,[X_grid.size,1]), np.reshape(p_y,[X_grid.size,1])

def gradients_fourier_fd(X_grid, Y_grid, phi_xr, phi_xi, phi_yr, phi_yi, psi_r, psi_i):
    # for comparison, compute the gradients using standard finite differences
    phi_x = np.reshape(np.complex64(phi_xr+1j*phi_xi),X_grid.shape)
    phi_y = np.reshape(np.complex64(phi_yr+1j*phi_yi),X_grid.shape)
    psi = np.reshape(np.complex64(psi_r+1j*psi_i),X_grid.shape)

    phi_x_x = np.gradient(phi_x,X_grid,axis=0)
    phi_x_y = np.gradient(phi_x,Y_grid,axis=1)
    phi_x_xx = np.gradient(phi_x_x,X_grid,axis=0)
    phi_x_yy = np.gradient(phi_x_y,Y_grid,axis=1)

    phi_y_x = np.gradient(phi_y,X_grid,axis=0)
    phi_y_y = np.gradient(phi_y,Y_grid,axis=1)
    phi_y_xx = np.gradient(phi_y_x,X_grid,axis=0)
    phi_y_yy = np.gradient(phi_y_y,Y_grid,axis=1)

    psi_x = np.gradient(psi,X_grid,axis=0)
    psi_y = np.gradient(psi,Y_grid,axis=1)

    return np.reshape(phi_x,[X_grid.size,1]), np.reshape(phi_x_x,[X_grid.size,1]), np.reshape(phi_x_y,[X_grid.size,1]), np.reshape(phi_x_xx,[X_grid.size,1]), np.reshape(phi_x_yy,[X_grid.size,1]), np.reshape(phi_y,[X_grid.size,1]), np.reshape(phi_y_x,[X_grid.size,1]), np.reshape(phi_y_y,[X_grid.size,1]), np.reshape(phi_y_xx,[X_grid.size,1]), np.reshape(phi_y_yy,[X_grid.size,1]), np.reshape(psi_r,[X_grid.size,1]), np.reshape(psi_x,[X_grid.size,1]), np.reshape(psi_y,[X_grid.size,1])

def galerkin_projection_fd():

    return 1


def galerkin_projection(ux, ux_x, ux_y, ux_xx, ux_yy, uy, uy_x, uy_y, uy_xx, uy_yy, p, p_x, p_y, phi_xr, phi_xr_x, phi_xr_y, phi_xr_xx, phi_xr_yy, phi_xi, phi_xi_x, phi_xi_y, phi_xi_xx, phi_xi_yy, phi_yr, phi_yr_x, phi_yr_y, phi_yr_xx, phi_yr_yy, phi_yi, phi_yi_x, phi_yi_y, phi_yi_xx, phi_yi_yy, psi_r, psi_r_x, psi_r_y, psi_i, psi_i_x, psi_i_y):
    nx = phi_xr.shape[0]
    nk = phi_xr.shape[1]

    Fk =  np.zeros([nx,nk],dtype=np.cdouble)
    Lkl = np.zeros([nx,nk,nk],dtype=np.cdouble)
    Qklm = np.zeros([nx,nk,nk,nk],np.cdouble)

    for k in range(nk):
        # phi_k
        phi_k_x = phi_xr[:,k]+1j*phi_xi[:,k]
        phi_k_y = phi_yr[:,k]+1j*phi_yi[:,k]
        Fk[:,k] = -phi_k_x*np.cdouble(2*ux*ux_x+ux*uy_y+uy*ux_y)- phi_k_y*np.cdouble(ux*uy_x+uy*ux_x+2*uy*uy_y) - phi_k_x*np.cdouble(p_x) - phi_k_y*np.cdouble(p_y) + np.cdouble(nu_mol)*(phi_k_x*np.cdouble(ux_xx+ux_yy)+phi_k_y*np.cdouble(uy_xx+uy_yy))

        for l in range(nk):
            #phi_l
            phi_l_x = phi_xr[:,l]+1j*phi_xi[:,l]
            phi_l_y = phi_yr[:,l]+1j*phi_yi[:,l]
            # derivatives of phi_l
            phi_l_x_x = phi_xr_x[:,l] + 1j*phi_xi_x[:,l]
            phi_l_x_y = phi_xr_y[:,l] + 1j*phi_xi_y[:,l]
            phi_l_y_x = phi_yr_x[:,l] + 1j*phi_yi_x[:,l]
            phi_l_y_y = phi_yr_y[:,l] + 1j*phi_yi_y[:,l]
            phi_l_x_xx = phi_xr_xx[:,l] + 1j*phi_xi_xx[:,l]
            phi_l_x_yy = phi_xr_yy[:,l] + 1j*phi_xi_yy[:,l]
            phi_l_y_xx = phi_yr_xx[:,l] + 1j*phi_yi_xx[:,l]
            phi_l_y_yy = phi_yr_yy[:,l] + 1j*phi_yi_yy[:,l]
            # psi derivatives
            psi_l_x = psi_r_x[:,l]+1j*psi_i_x[:,l]
            psi_l_y = psi_r_y[:,l]+1j*psi_i_y[:,l]

            Lkl[:,k,l] = (-phi_k_x*(phi_l_x*np.cdouble(ux_x)+phi_l_y*np.cdouble(ux_y)) -phi_k_y*(phi_l_x*np.cdouble(uy_x)+phi_l_y*np.cdouble(uy_y))
                          -phi_k_x*(np.cdouble(ux)*phi_l_x_x+np.cdouble(uy)*phi_l_x_y)-phi_k_y*(np.cdouble(ux)*phi_l_y_x+np.cdouble(uy)*phi_l_y_y)
                          -phi_k_x*psi_l_x-phi_k_y*psi_l_y
                          +np.cdouble(nu_mol)*(phi_k_x*(phi_l_x_xx+phi_l_x_yy)+phi_k_y*(phi_l_y_xx+phi_l_y_yy)))
    
            for m in range(nk):
                # derivatives of phi_m
                phi_m_x_x = phi_xr_x[:,m] + 1j*phi_xi_x[:,m]
                phi_m_x_y = phi_xr_y[:,m] + 1j*phi_xi_y[:,m]
                phi_m_y_x = phi_yr_x[:,m] + 1j*phi_yi_x[:,m]
                phi_m_y_y = phi_yr_y[:,m] + 1j*phi_yi_y[:,m]

                Qklm[:,k,l,m] = -phi_k_x*(phi_l_x*phi_m_x_x + phi_l_y*phi_m_x_y) -phi_k_y*(phi_l_x*phi_m_y_x + phi_l_y*phi_m_y_y)

    return Fk, Lkl, Qklm


base_dir = 'C:/projects/pinns_beluga/sync/'
data_dir = base_dir+'data/mazi_fixed_grid_wake/'
output_base_dir = base_dir+'output/'

configFile = h5py.File(data_dir+'configuration.mat','r')
meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
fourierModeFile = h5py.File(data_dir+'fourierDataShort.mat','r')

ux_mean = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy_mean = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
MAX_ux = np.max(ux_mean.flatten())
MAX_uy = np.max(uy_mean.flatten())

nu_mol = 0.0066667
MAX_p = 1.0
MAX_psi= 0.1 # chosen based on abs(max(psi))

x = np.array(configFile['X_vec'][0,:])
X_grid_temp = np.array(configFile['X_grid'])
X_grid =np.reshape(x,X_grid_temp.shape[::-1])
print(X_grid.shape)
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.reshape(y,X_grid.shape)


MAX_x = np.max(x.flatten())
MAX_y = np.max(y.flatten())
# normalize the training data:
x_train = x/MAX_x
y_train = y/MAX_y
# note that the order here needs to be the same as the split inside the network!
X_train = np.hstack((x_train.reshape(-1,1),y_train.reshape(-1,1)))

mean_model_filestr, mean_model_epochs = find_highest_numbered_file(output_base_dir+'mfgw_mean003_output/mfgw_mean003_ep','[0-9]*','_model.h5')
model_mean = keras.models.load_model(mean_model_filestr,custom_objects ={'custom_loss':dummy_loss})

ux, ux_x, ux_y, ux_xx, ux_yy, uy, uy_x, uy_y, uy_xx, uy_yy, p, p_x, p_y = gradients_mean(X_train)


fourier_mode_cases = ['mfgw_fourier8_002','mfgw_fourier9_001','mfgw_fourier10_002','mfgw_fourier11_001','mfgw_fourier20_001','mfgw_fourier21_001']
fourier_mode_numbers = [8,9,10,11,20,21]

# create the derivative arrays
fft_phi_xr_2 = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

phi_xr = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xr_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xr_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xr_xx = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xr_yy = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

phi_xi = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xi_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xi_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xi_xx = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_xi_yy = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

phi_yr = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yr_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yr_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yr_xx = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yr_yy = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

phi_yi = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yi_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yi_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yi_xx = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
phi_yi_yy = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

psi_r  =np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
psi_r_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
psi_r_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
psi_i = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
psi_i_x = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])
psi_i_y = np.zeros([X_train.shape[0],len(fourier_mode_numbers)])

import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

fourier_models = []
for k in range(len(fourier_mode_numbers)):
    fourier_mode_filestr, fourier_mode_epoch_number = find_highest_numbered_file(output_base_dir+fourier_mode_cases[k]+'_output/'+fourier_mode_cases[k]+'_ep','[0-9]*','_model.h5')

    mode_number = fourier_mode_numbers[k]-1
    fft_phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
    fft_phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
    fft_phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
    fft_phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

    MAX_phi_xr = np.max(fft_phi_xr.flatten())
    MAX_phi_xi = np.max(fft_phi_xi.flatten())
    MAX_phi_yr = np.max(fft_phi_yr.flatten())
    MAX_phi_yi = np.max(fft_phi_yi.flatten())   
    
    print(fourier_mode_cases[k])    
    print(output_base_dir+fourier_mode_cases[k]+'_output/'+fourier_mode_cases[k]+'_ep'+str(fourier_mode_epoch_number)+'_model.h5')
    model_fourier = keras.models.load_model(output_base_dir+fourier_mode_cases[k]+'_output/'+fourier_mode_cases[k]+'_ep'+str(fourier_mode_epoch_number)+'_model.h5',custom_objects={'custom_loss':dummy_loss,})
    

    fft_phi_xr_2[:,k] = fft_phi_xr
    phi_xr[:,k], phi_xr_x[:,k], phi_xr_y[:,k], phi_xr_xx[:,k], phi_xr_yy[:,k], phi_xi[:,k], phi_xi_x[:,k], phi_xi_y[:,k], phi_xi_xx[:,k], phi_xi_yy[:,k], phi_yr[:,k], phi_yr_x[:,k], phi_yr_y[:,k], phi_yr_xx[:,k], phi_yr_yy[:,k], phi_yi[:,k], phi_yi_x[:,k], phi_yi_y[:,k], phi_yi_xx[:,k], phi_yi_yy[:,k], psi_r[:,k], psi_r_x[:,k], psi_r_y[:,k], psi_i[:,k], psi_i_x[:,k], psi_i_y[:,k] = gradients_fourier(model_fourier,X_train)
    
    keras.backend.clear_session()
    del model_fourier


Fk, Lkl, Qklm = galerkin_projection(ux, ux_x, ux_y, ux_xx, ux_yy, uy, uy_x, uy_y, uy_xx, uy_yy, p, p_x, p_y, phi_xr, phi_xr_x, phi_xr_y, phi_xr_xx, phi_xr_yy, phi_xi, phi_xi_x, phi_xi_y, phi_xi_xx, phi_xi_yy, phi_yr, phi_yr_x, phi_yr_y, phi_yr_xx, phi_yr_yy, phi_yi, phi_yi_x, phi_yi_y, phi_yi_xx, phi_yi_yy, psi_r, psi_r_x, psi_r_y, psi_i, psi_i_x, psi_i_y)

Qklm_grid = np.reshape(Qklm,[X_grid.shape[0],X_grid.shape[1],Qklm.shape[1],Qklm.shape[2],Qklm.shape[3]])



x_lim_vec = [0.5,10.0]
y_lim_vec = [-2.0,2.0]
if False:
    fig1 = plot.figure(1)
    for k in range(6):
        ax = fig1.add_subplot(6,1,k+1)
        plot.axis('equal')
        plot.contourf(X_grid,Y_grid,np.reshape(phi_xr[:,k],X_grid.shape),levels=21)
        plot.set_cmap('bwr')
        plot.colorbar()
        ax=plot.gca()
        ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
        ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
        plot.ylabel('y/D')
    plot.show()

if True:

    #f1_levels = np.linspace(-MAX_ux,MAX_ux,21)
    fig1 = plot.figure(1)
    for l in range(6):
        for m in range(6):
            ax = fig1.add_subplot(6,6,6*l+m+1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,np.real(Qklm_grid[:,:,1,l,m]),levels=21)
            plot.set_cmap('bwr')
            plot.colorbar()
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if m==0:
                plot.ylabel('y/D')
            if l==5:
                plot.xlabel('x/D')

    fig2 = plot.figure(2)
    for l in range(6):
        for m in range(6):
            ax = fig2.add_subplot(6,6,6*l+m+1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,np.imag(Qklm_grid[:,:,1,l,m]),levels=21)
            plot.set_cmap('bwr')
            plot.colorbar()
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if m==0:
                plot.ylabel('y/D')
            if l==5:
                plot.xlabel('x/D')




    plot.show()

