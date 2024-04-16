

import numpy as np
import h5py

import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.dft import idft

figures_dir = 'C:/projects/paper_figures/mean_field/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

cases_list_mean = ['mfg_fbc003_001_S0/mfg_fbc003_001_S0_ep72927_pred.mat','mfg_fbc003_001_S2/mfg_fbc003_001_S2_ep74925_pred.mat','mfg_fbc003_001_S4/mfg_fbc003_001_S4_ep86913_pred.mat','mfg_fbc003_001_S8/mfg_fbc003_001_S8_ep101898_pred.mat','mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_pred.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_pred.mat']
cases_list_f0 = ['mfg_fbcf007_f0_S0_j001_output/mfg_fbcf007_f0_S0_j001_ep153846_pred.mat','mfg_fbcf007_f0_S2_j001_output/mfg_fbcf007_f0_S2_j001_ep265734_pred.mat','mfg_fbcf007_f0_S4_j001_output/mfg_fbcf007_f0_S4_j001_ep164835_pred.mat','mfg_fbcf007_f0_S8_j001_output/mfg_fbcf007_f0_S8_j001_ep167832_pred.mat','mfg_fbcf007_f0_S16_j001_output/mfg_fbcf007_f0_S16_j001_ep175824_pred.mat','mfg_fbcf007_f0_S32_j001_output/mfg_fbcf007_f0_S32_j001_ep164835_pred.mat']
cases_list_f1 = ['mfg_fbcf007_f1_S0_j001_output/mfg_fbcf007_f1_S0_j001_ep153846_pred.mat','mfg_fbcf007_f1_S2_j001_output/mfg_fbcf007_f1_S2_j001_ep163836_pred.mat','mfg_fbcf007_f1_S4_j001_output/mfg_fbcf007_f1_S4_j001_ep164835_pred.mat','mfg_fbcf007_f1_S8_j001_output/mfg_fbcf007_f1_S8_j001_ep164835_pred.mat','mfg_fbcf007_f1_S16_j001_output/mfg_fbcf007_f1_S16_j001_ep164835_pred.mat','mfg_fbcf007_f1_S32_j001_output/mfg_fbcf007_f1_S32_j001_ep161838_pred.mat']
cases_list_f2 = ['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_pred.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_pred.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_pred.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_pred.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_pred.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_pred.mat']
cases_supersample_factor = [0,2,4,8,16,32]

# get the constants for all the modes
class UserScalingParameters(object):
    pass
ScalingParameters = UserScalingParameters()
ScalingParameters.mean = UserScalingParameters()
ScalingParameters.f0 = UserScalingParameters()
ScalingParameters.f1 = UserScalingParameters()
ScalingParameters.f2 = UserScalingParameters()

# load the reference data
base_dir = data_dir



meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()
p = np.array(meanPressureFile['meanPressure'])

ScalingParameters.mean.MAX_x = 10.0
ScalingParameters.mean.MAX_y = 10.0
ScalingParameters.mean.MAX_ux = np.max(ux)
ScalingParameters.mean.MAX_uy = np.max(uy)
ScalingParameters.mean.MAX_uxux = np.max(uxux)
ScalingParameters.mean.MAX_uxuy = np.max(uxuy)
ScalingParameters.mean.MAX_uyuy = np.max(uyuy)
ScalingParameters.mean.MAX_p = 1.0


for mode_number in [0,1,2]:
    fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))

    phi_xi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,1]))

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_i = np.array(np.imag(fourierModeFile['pressureModes'][:,mode_number]))

    tau_xx_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xx_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xy_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_xy_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_yy_r = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,2]))
    tau_yy_i = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,2]))

    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi


# now load the modes

for c in [0]:
    pred_mean_file = h5py.File(output_dir+cases_list[c],'r')   
    ux_pred = np.array(pred_mean_file['pred'][:,0])*MAX_ux)
    uy_pred.append(np.array(predFile['pred'][:,1])*MAX_uy)
    p_pred.append(np.array(predFile['pred'][:,5])*MAX_p)


L_dft=4082
fs = 10
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])




