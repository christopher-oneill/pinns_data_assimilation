

import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp
import os
import re

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


# script

base_dir = 'C:/projects/pinns_beluga/sync/'
data_dir = base_dir+'data/mazi_fixed_grid_wake/'
output_base_dir = base_dir+'output/'


meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
configFile = h5py.File(data_dir+'configuration.mat','r')
meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
fourierModeFile = h5py.File(data_dir+'fourierDataShort.mat','r')
fluctuatingVelocityFile = h5py.File(data_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(data_dir+'fluctuatingPressure.mat','r')

ux_mean = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy_mean = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]

time = configFile['time']
print(time.shape)
nT = np.size(time)
nX = np.size(configFile['X_vec'],1)

x = np.array(configFile['X_vec'][0,:])
X_grid_temp = np.array(configFile['X_grid'])
X_grid =np.reshape(x,X_grid_temp.shape[::-1])
print(X_grid.shape)
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.reshape(y,X_grid.shape)

velocityField = np.array(fluctuatingVelocityFile['fluctuatingVelocity']).transpose()
pressureField = np.array(fluctuatingPressureFile['fluctuatingPressure']).transpose()
for t in range(nT):
    velocityField[:,t,0] = velocityField[:,t,0] + ux_mean
    velocityField[:,t,1] = velocityField[:,t,1] + uy_mean
    pressureField[:,t] = pressureField[:,t] + p


MAX_ux_mean = np.max(ux_mean)
MAX_uy_mean = np.max(uy_mean)
MAX_p_mean = 1 # estimated maximum pressure

MAX_u = np.max(np.power(np.power(velocityField[:,:,0],2.0)+np.power(velocityField[:,:,1],2.0),0.5).flatten())


reconstructedVelocity = np.zeros([nX,nT,2])
reconstructedPressure = np.zeros([nX,nT])
PINNreconstructedVelocity = np.zeros([nX,nT,2])
PINNreconstructedPressure = np.zeros([nX,nT])

# load the mean field

mean_field_case = 'mfgw_mean003'
meanPredFilename,mean_epoch_number = find_highest_numbered_file(output_base_dir+mean_field_case+'_output/'+mean_field_case+'_ep','[0-9]*','_pred.mat')
meanPredFile =  h5py.File(meanPredFilename,'r')

ux_mean_pred = np.array(meanPredFile['pred'][:,0])*MAX_ux_mean
uy_mean_pred = np.array(meanPredFile['pred'][:,1])*MAX_uy_mean
p_mean_pred = np.array(meanPredFile['pred'][:,5])*MAX_p_mean

for t in range(nT):   
    reconstructedVelocity[:,t,0] = ux_mean_pred
    reconstructedVelocity[:,t,1] = uy_mean_pred
    reconstructedPressure[:,t] = p_mean_pred
    PINNreconstructedVelocity[:,t,0] = ux_mean_pred
    PINNreconstructedVelocity[:,t,1] = uy_mean_pred
    PINNreconstructedPressure[:,t] = p_mean_pred



# load the fourier modes

fourier_mode_cases = ['mfgw_fourier8_002','mfgw_fourier9_001','mfgw_fourier10_002','mfgw_fourier11_001','mfgw_fourier20_001','mfgw_fourier21_001']
fourier_mode_numbers = [8,9,10,11,20,21]

fourier_mode_structs = []
for k in range(len(fourier_mode_numbers)):
    fourier_mode_filestr, fourier_mode_epoch_number = find_highest_numbered_file(output_base_dir+fourier_mode_cases[k]+'_output/'+fourier_mode_cases[k]+'_ep','[0-9]*','_pred.mat')
    fourier_mode_structs.append(h5py.File(fourier_mode_filestr,'r'))
    fs = np.array(configFile['fs'])

    mode_number = fourier_mode_numbers[k]-1
    omega = (1/(fs*fs))*np.array(fourierModeFile['fShort'][0,mode_number])*2*np.pi
    phase_vector = np.exp(1j*omega*time)

    phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
    phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
    phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
    phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

    psi_r = np.array(fourierModeFile['pressureModesShortReal'][mode_number,:]).transpose()
    psi_i = np.array(fourierModeFile['pressureModesShortImag'][mode_number,:]).transpose()

    MAX_psi= 0.1 # chosen based on abs(max(psi))

    MAX_phi_xr = np.max(phi_xr.flatten())
    MAX_phi_xi = np.max(phi_xi.flatten())
    MAX_phi_yr = np.max(phi_yr.flatten())
    MAX_phi_yi = np.max(phi_yi.flatten())

    MAX_psi_r = np.max(psi_r.flatten())
    MAX_psi_i = np.max(psi_i.flatten())
    
    phi_xr_pred = np.array(fourier_mode_structs[k]['pred'][:,0])*MAX_phi_xr
    phi_xi_pred = np.array(fourier_mode_structs[k]['pred'][:,1])*MAX_phi_xi
    phi_yr_pred = np.array(fourier_mode_structs[k]['pred'][:,2])*MAX_phi_yr
    phi_yi_pred = np.array(fourier_mode_structs[k]['pred'][:,3])*MAX_phi_yi

    psi_r_pred = np.array(fourier_mode_structs[k]['pred'][:,10])*MAX_psi
    psi_i_pred = np.array(fourier_mode_structs[k]['pred'][:,11])*MAX_psi
    
    mode_velocity_x = np.real(np.matmul(np.reshape(phi_xr_pred+1j*phi_xi_pred,[phi_xr_pred.size,1]),phase_vector))
    mode_velocity_y = np.real(np.matmul(np.reshape(phi_yr_pred+1j*phi_yi_pred,[phi_yr_pred.size,1]),phase_vector))
    mode_pressure = np.real(np.matmul(np.reshape(psi_r_pred+1j*psi_i_pred,[phi_xr_pred.size,1]),phase_vector))



    PINNreconstructedVelocity[:,:,0] = PINNreconstructedVelocity[:,:,0] + mode_velocity_x
    PINNreconstructedVelocity[:,:,1] = PINNreconstructedVelocity[:,:,1] + mode_velocity_y
    PINNreconstructedPressure = PINNreconstructedPressure + mode_pressure

for k in range(1,21):
    mode_number = k-1
    omega = (1/(fs*fs))*np.array(fourierModeFile['fShort'][0,mode_number])*2*np.pi
    phase_vector = np.exp(1j*omega*time)

    phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
    phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
    phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
    phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

    psi_r = np.array(fourierModeFile['pressureModesShortReal'][mode_number,:]).transpose()
    psi_i = np.array(fourierModeFile['pressureModesShortImag'][mode_number,:]).transpose()
    
    mode_velocity_x = 2.0*np.real(np.matmul(np.reshape(phi_xr+1j*phi_xi,[phi_xr.size,1]),phase_vector))
    mode_velocity_y = 2.0*np.real(np.matmul(np.reshape(phi_yr+1j*phi_yi,[phi_yr.size,1]),phase_vector))
    mode_pressure = np.real(np.matmul(np.reshape(psi_r+1j*psi_i,[phi_xr.size,1]),phase_vector))

    reconstructedVelocity[:,:,0] = reconstructedVelocity[:,:,0] + mode_velocity_x
    reconstructedVelocity[:,:,1] = reconstructedVelocity[:,:,1] + mode_velocity_y
    reconstructedPressure = reconstructedPressure + mode_pressure

velocityFieldGrid = np.reshape(velocityField,[np.shape(X_grid)[0],np.shape(X_grid)[1],nT,2])
reconstructedVelocityGrid = np.reshape(reconstructedVelocity,[np.shape(X_grid)[0],np.shape(X_grid)[1],nT,2])

 
grad_x = np.gradient(velocityFieldGrid,X_grid[:,0],axis=0)
grad_y = np.gradient(velocityFieldGrid,Y_grid[0,:],axis=1)

gradR_x = np.gradient(reconstructedVelocityGrid,X_grid[:,0],axis=0)
gradR_y = np.gradient(reconstructedVelocityGrid,Y_grid[0,:],axis=1)

vorticity = grad_x[:,:,:,0]-grad_y[:,:,:,1]
vorticityR = gradR_x[:,:,:,0]-gradR_y[:,:,:,1]

x_lim_vec = [0.5,10.0]
y_lim_vec = [-2.0,2.0]
f1_levels = np.linspace(-MAX_u,MAX_u,21)
fig = plot.figure(1)
ax = fig.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(X_grid,Y_grid,velocityFieldGrid[:,:,100,0],levels=f1_levels)
plot.set_cmap('bwr')
plot.colorbar()
ax=plot.gca()
ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
plot.ylabel('y/D')
fig.add_subplot(3,1,2)
plot.contourf(X_grid,Y_grid,reconstructedVelocityGrid[:,:,100,0],levels=f1_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
ax=plot.gca()
ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
plot.axis('equal')
fig.add_subplot(3,1,3)
plot.contourf(X_grid,Y_grid,(velocityFieldGrid[:,:,100,0]-reconstructedVelocityGrid[:,:,100,0])/MAX_u,levels=21)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
plot.xlabel('x/D')
plot.axis('equal')
ax=plot.gca()
ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])

plot.show()