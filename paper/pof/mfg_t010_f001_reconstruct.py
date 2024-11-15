

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot

import sys
sys.path.append('F:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.dft import idft

figures_dir = 'F:/projects/paper_figures/t010/reconstruction/'
rec_dir = 'F:/projects/paper_figures/t010/data/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

from pinns_data_assimilation.lib.file_util import find_highest_numbered_file

cases_supersample_factor = [0,2,4,8,16,32]
cases_frequency = [0,1,2,3,4,5]

cases_list_mean = []
for ik in range(len(cases_supersample_factor)):
    file_path,file_number = find_highest_numbered_file(output_dir+'mfg_t010_001_S'+str(cases_supersample_factor[ik])+'/mfg_t010_001_S'+str(cases_supersample_factor[ik])+'_ep','[0-9]*','_pred.mat')
    cases_list_mean.append('mfg_t010_001_S'+str(cases_supersample_factor[ik])+'/mfg_t010_001_S'+str(cases_supersample_factor[ik])+'_ep'+str(file_number)+'_pred.mat')

cases_list_f = []
for ij in range(len(cases_frequency)):
    temp_cases_list = []
    temp_phys_list = []
    for ik in range(len(cases_supersample_factor)):
        file_path,file_number = find_highest_numbered_file(output_dir+'mfg_t010_f001_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_output/mfg_t010_f001_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_ep','[0-9]*','_pred.mat')
        temp_cases_list.append('mfg_t010_f001_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_output/mfg_t010_f001_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_ep'+str(file_number)+'_pred.mat')
    cases_list_f.append(temp_cases_list)




# get the constants for all the modes
class UserScalingParameters(object):
    pass
ScalingParameters = UserScalingParameters()
ScalingParameters.mean = UserScalingParameters()
ScalingParameters.f =[]


# load the reference data
base_dir = data_dir

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')


x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

X_grid_plot = X_grid
Y_grid_plot = Y_grid
X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)


ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

uxt = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uyt = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()
pt = np.array(fluctuatingPressureFile['fluctuatingPressure'][:,:]).transpose()

uxt = uxt+np.reshape(ux,[ux.shape[0],1])
uyt = uyt+np.reshape(uy,[uy.shape[0],1])
pt = pt+np.reshape(p,[uy.shape[0],1])

uxt = np.reshape(uxt,[X_grid.shape[0],X_grid.shape[1],uxt.shape[1]])
uyt = np.reshape(uyt,[X_grid.shape[0],X_grid.shape[1],uyt.shape[1]])
pt = np.reshape(pt,[X_grid.shape[0],X_grid.shape[1],pt.shape[1]])

ScalingParameters.mean.fs = 10.0
ScalingParameters.mean.MAX_x = 10.0
ScalingParameters.mean.MAX_y = 10.0
ScalingParameters.mean.MAX_ux = np.max(ux.flatten())
ScalingParameters.mean.MAX_uy = np.max(uy.flatten())
ScalingParameters.mean.MAX_p = 1.0
ScalingParameters.mean.nx = ux.shape[0]

phi_x_ref = []
phi_y_ref = []
psi_ref = []

for mode_number in [0,1,2,3,4,5,6,7]:
    ScalingParameters.f.append(UserScalingParameters())
    fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_x_ref.append(np.complex128(phi_xr)+1j*np.complex128(phi_xi))
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_y_ref.append(np.complex128(phi_yr)+1j*np.complex128(phi_yi))

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_i = np.array(np.imag(fourierModeFile['pressureModes'][:,mode_number]))
    psi_ref.append(np.complex128(psi_r)+1j*np.complex128(psi_i))

    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi

    ScalingParameters.f[mode_number].MAX_x = 20.0
    ScalingParameters.f[mode_number].MAX_y = 20.0 # we use the larger of the two spatial scalings
    ScalingParameters.f[mode_number].MAX_phi_xr = np.max(phi_xr.flatten())
    ScalingParameters.f[mode_number].MAX_phi_xi = np.max(phi_xi.flatten())
    ScalingParameters.f[mode_number].MAX_phi_yr = np.max(phi_yr.flatten())
    ScalingParameters.f[mode_number].MAX_phi_yi = np.max(phi_yi.flatten())
    ScalingParameters.f[mode_number].MAX_psi= 0.2*np.power((omega_0/omega),2.0) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
    ScalingParameters.f[mode_number].omega = omega
    ScalingParameters.f[mode_number].f = np.array(fourierModeFile['modeFrequencies'][mode_number])

L_dft=4082
fs = 10
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

# reconstruct with the reference fourier modes
# allocate the memory for the reconstructed quantities
ux_rec_ref = np.zeros([ScalingParameters.mean.nx,L_dft])
uy_rec_ref = np.zeros([ScalingParameters.mean.nx,L_dft])
p_rec_ref = np.zeros([ScalingParameters.mean.nx,L_dft])

# add the mean field
ux_rec_ref = ux_rec_ref + np.reshape(ux,[ScalingParameters.mean.nx,1])
uy_rec_ref = uy_rec_ref + np.reshape(uy,[ScalingParameters.mean.nx,1])
p_rec_ref = p_rec_ref + np.reshape(p,[ScalingParameters.mean.nx,1])

for c in [0,1,2,3,4,5,6,7]:
    ux_i,t_i = idft(np.reshape(phi_x_ref[c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
    ux_rec_ref = ux_rec_ref+2.0*ux_i

    uy_i,t_i = idft(np.reshape(phi_y_ref[c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
    uy_rec_ref = uy_rec_ref+2.0*uy_i

    p_i,t_i = idft(np.reshape(psi_ref[c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
    p_rec_ref = p_rec_ref+2.0*p_i

    h5f = h5py.File(rec_dir+'rec_fourier_c'+str(c)+'.h5','w')
    h5f.create_dataset('ux',data=ux_rec_ref)
    h5f.create_dataset('uy',data=uy_rec_ref)
    h5f.create_dataset('p',data=p_rec_ref)
    h5f.close()


ux_pred = []
uy_pred = []
p_pred = []

phi_x_pred = []
phi_y_pred = []
psi_pred = []

# now load the modes
for s in range(len(cases_supersample_factor)):
    phi_x_pred.append([])
    phi_y_pred.append([])
    psi_pred.append([])
    
    pred_mean_file = h5py.File(output_dir+cases_list_mean[s],'r')   
    ux_pred.append(np.array(pred_mean_file['pred'][:,0])*ScalingParameters.mean.MAX_ux)
    uy_pred.append(np.array(pred_mean_file['pred'][:,1])*ScalingParameters.mean.MAX_uy)
    p_pred.append(np.array(pred_mean_file['pred'][:,5])*ScalingParameters.mean.MAX_p)

    for c in [0,1,2,3,4,5]:
        pred_f_file = h5py.File(output_dir+cases_list_f[c][s],'r')
        #print(output_dir+cases_list_f[c][s])
        phi_x_pred[s].append(np.array(pred_f_file['pred'][:,0],dtype=np.complex128)*ScalingParameters.f[c].MAX_phi_xr+np.array(pred_f_file['pred'][:,1],dtype=np.complex128)*ScalingParameters.f[c].MAX_phi_xi*1j)
        phi_y_pred[s].append(np.array(pred_f_file['pred'][:,2],dtype=np.complex128)*ScalingParameters.f[c].MAX_phi_yr+np.array(pred_f_file['pred'][:,3],dtype=np.complex128)*ScalingParameters.f[c].MAX_phi_yi*1j)
        psi_pred[s].append(np.array(pred_f_file['pred'][:,10],dtype=np.complex128)*ScalingParameters.f[c].MAX_psi+np.array(pred_f_file['pred'][:,11],dtype=np.complex128)*ScalingParameters.f[c].MAX_psi*1j)







# perform the reconstruction
for s in range(len(cases_supersample_factor)):
    # allocate the memory for the reconstructed quantities
    ux_rec = np.zeros([ScalingParameters.mean.nx,L_dft])
    uy_rec = np.zeros([ScalingParameters.mean.nx,L_dft])
    p_rec = np.zeros([ScalingParameters.mean.nx,L_dft])

    # add the mean field
    ux_rec = ux_rec + np.reshape(ux_pred[s],[ScalingParameters.mean.nx,1])
    uy_rec = uy_rec + np.reshape(uy_pred[s],[ScalingParameters.mean.nx,1])
    p_rec = p_rec + np.reshape(p_pred[s],[ScalingParameters.mean.nx,1])

    for c in [0,1,2,3,4,5]:
        ux_i,t_i = idft(np.reshape(phi_x_pred[s][c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
        ux_rec = ux_rec+2.0*ux_i

        uy_i,t_i = idft(np.reshape(phi_y_pred[s][c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
        uy_rec = uy_rec+2.0*uy_i

        p_i,t_i = idft(np.reshape(psi_pred[s][c],[ScalingParameters.mean.nx,1]),f=ScalingParameters.f[c].f,t=np.reshape(t,[1,t.size]),fs=ScalingParameters.mean.fs)
        p_rec = p_rec+2.0*p_i

        h5f = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.h5','w')
        h5f.create_dataset('ux',data=ux_rec)
        h5f.create_dataset('uy',data=uy_rec)
        h5f.create_dataset('p',data=p_rec)
        h5f.close()

    
if False:
    # plots for debugging the reconstruction
    
    ux_rec = np.reshape(ux_rec,[X_grid.shape[0],X_grid.shape[1],L_dft])
    uy_rec = np.reshape(uy_rec,[X_grid.shape[0],X_grid.shape[1],L_dft])
    p_rec = np.reshape(p_rec,[X_grid.shape[0],X_grid.shape[1],L_dft])
    ux_rec_ref = np.reshape(ux_rec_ref,[X_grid.shape[0],X_grid.shape[1],L_dft])
    uy_rec_ref = np.reshape(uy_rec_ref,[X_grid.shape[0],X_grid.shape[1],L_dft])
    p_rec_ref = np.reshape(p_rec_ref,[X_grid.shape[0],X_grid.shape[1],L_dft])

    temp_ux_ref = uxt[:,:,1000]
    temp_ux_ref[cylinder_mask]=np.NaN
    temp_ux_rec = ux_rec[:,:,1000]
    temp_ux_rec[cylinder_mask]=np.NaN
    temp_ux_rec_ref = ux_rec_ref[:,:,1000]
    temp_ux_rec_ref[cylinder_mask]=np.NaN
    temp_MAX = np.max([np.nanmax(np.abs(temp_ux_ref)),np.nanmax(np.abs(temp_ux_ref))])
    temp_levels = np.linspace(-temp_MAX,temp_MAX,21)
    plot.figure(1)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_ux_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_ux_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_ux_ref-temp_ux_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    plot.figure(2)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_ux_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_ux_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_ux_ref-temp_ux_rec_ref,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    plot.figure(3)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_ux_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_ux_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_ux_rec_ref-temp_ux_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    temp_uy_ref = uyt[:,:,1000]
    temp_uy_ref[cylinder_mask]=np.NaN
    temp_uy_rec = uy_rec[:,:,1000]
    temp_uy_rec[cylinder_mask]=np.NaN
    temp_uy_rec_ref = uy_rec_ref[:,:,1000]
    temp_uy_rec_ref[cylinder_mask]=np.NaN
    temp_MAX = np.max([np.nanmax(np.abs(temp_uy_ref)),np.nanmax(np.abs(temp_uy_ref))])
    temp_levels = np.linspace(-temp_MAX,temp_MAX,21)
    plot.figure(21)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_uy_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_uy_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_uy_ref-temp_uy_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    plot.figure(22)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_uy_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_uy_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_uy_ref-temp_uy_rec_ref,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    plot.figure(23)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_uy_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_uy_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_uy_rec_ref-temp_uy_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    temp_p_ref = pt[:,:,1000]
    temp_p_ref[cylinder_mask]=np.NaN
    temp_p_rec = p_rec[:,:,1000]
    temp_p_rec[cylinder_mask]=np.NaN
    temp_p_rec_ref = p_rec_ref[:,:,1000]
    temp_p_rec_ref[cylinder_mask]=np.NaN
    temp_MAX = np.max([np.nanmax(np.abs(temp_p_ref)),np.nanmax(np.abs(temp_p_ref))])
    temp_levels = np.linspace(-temp_MAX,temp_MAX,21)
    plot.figure(31)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_p_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_p_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_p_ref-temp_p_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()

    plot.figure(32)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_p_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_p_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_p_ref-temp_p_rec_ref,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()
    
    plot.figure(33)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,temp_p_rec_ref,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,temp_p_rec,levels=temp_levels,cmap= matplotlib.colormaps['bwr'])
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,temp_p_rec_ref-temp_p_rec,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
    plot.colorbar()



    
    plot.show()
