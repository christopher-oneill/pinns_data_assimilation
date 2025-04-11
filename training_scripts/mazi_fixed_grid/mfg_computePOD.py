

import numpy as np
import sys
import h5py
import platform
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

node_name = platform.node()

LOCAL_NODE = 'DESKTOP-L3FA8HC'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    import matplotlib.colors as mplcolors
    useGPU=False    
    HOMEDIR = 'F:/projects/pinns_narval/sync/'
    sys.path.append('F:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists
from pinns_data_assimilation.lib.decomposition import POD

# read the data

base_dir = HOMEDIR+'data/mazi_fixed_grid/'
time_data_dir = 'F:/projects/fixed_cylinder/grid/data/'

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')

fs = 10.0

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

ux_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uy_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()

L_dft=4082

ux_ref = ux_ref[:,0:L_dft]
uy_ref = uy_ref[:,0:L_dft]

# downsample the data, if needed

if True:

    # concatenate into a single array along the space axis
    U_pod = np.concatenate((ux_ref,uy_ref),axis=0)

    Phi, Lambda, Ak = POD(U_pod)

    Phi = Phi[:,0:16]
    Ak = Ak[:,0:16]

    save_filestring = 'POD_data_m16.mat'
    h5f = h5py.File(base_dir+save_filestring,'w')
    h5f.create_dataset('Phi',data=Phi)
    h5f.create_dataset('Lambda',data=Lambda)
    h5f.create_dataset('Ak',data=Ak)
    h5f.close()


if False:
    plot.figure(iS)
    plot.plot(Ak[:,0])
    plot.xlim([0,20])

    plot.figure(2)
    plot.contourf(np.reshape(Phi[0:X_grid.size,0],[X_grid.shape[0],X_grid.shape[1]]))


    plot.show()


