

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plot

HOMEDIR = 'C:/projects/pinns_narval/sync/'
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.decomposition import POD

base_dir = HOMEDIR+'/data/mazi_fixed_grid/'


meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')


x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')

um_x = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
um_y = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
pm = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

p_t  = np.array(fluctuatingPressureFile['fluctuatingPressure']).transpose()
p_t = p_t + np.reshape(pm,[p_t.shape[0],1])
t = np.array(configFile['time']).transpose()

u_t = np.array(fluctuatingVelocityFile['fluctuatingVelocity']).transpose()

print('Fluctuating field shape',u_t.shape)

Phi, Lambda, Ak = POD(u_t)

print('Phi.shape',Phi.shape)
print('Lambda.shape',Lambda.shape)
print('Ak.shape',Ak.shape)

n_r = 6
err_L2 = np.zeros([n_r+1,2])
r_m = np.arange(2,(n_r+1)*4,4)

for m in range(r_m.shape[0]):
    # reconstruct the field
    u_t_r = np.matmul(Phi[:,0:r_m[m]],Ak[:,0:r_m[m]].transpose())
    u_t_r = np.stack((u_t_r[0:u_t.shape[0],:],u_t_r[u_t.shape[0]:u_t_r.shape[0],:]),axis=2)

    print('Reconstructed field shape:',u_t_r.shape)
    err_L2[m,:] = np.mean(np.mean(np.abs(u_t-u_t_r),axis=0),axis=0)

plot.figure(1)
plot.plot(r_m,err_L2[:,0])
plot.plot(r_m,err_L2[:,1])
plot.yscale('log')
plot.show()