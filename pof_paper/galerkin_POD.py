
import sys
import numpy as np
import h5py

HOMEDIR = 'C:/projects/pinns_narval/sync/'
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.decomposition import POD



base_dir = HOMEDIR+'/data/mazi_fixed_grid/'


meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')


fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')

um_x = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
um_y = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
pm = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

p_t  = np.array(fluctuatingPressureFile['fluctuatingPressure']).transpose()
p_t = p_t + np.reshape(pm,[p_t.shape[0],1])
t = np.array(configFile['time']).transpose()

u_t = np.array(fluctuatingVelocityFile['fluctuatingVelocity']).transpose()

POD(u_t)