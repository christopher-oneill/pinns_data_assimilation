
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

figures_dir = 'C:/projects/paper_figures/reconstruction/'
rec_dir = 'C:/projects/paper_figures/data/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

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

ux_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uy_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()
p_ref = np.array(fluctuatingPressureFile['fluctuatingPressure'][:,:]).transpose()

L_dft=4082

ux_ref = np.reshape(ux_ref,[X_grid.shape[0],X_grid.shape[1],ux_ref.shape[1]])
uy_ref = np.reshape(uy_ref,[X_grid.shape[0],X_grid.shape[1],uy_ref.shape[1]])
p_ref = np.reshape(p_ref,[X_grid.shape[0],X_grid.shape[1],p_ref.shape[1]])

fs=10.0
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

# crop the fluctuating fields to the first 4082 so they are the same as the fourier data
ux_ref = ux_ref[:,:,0:L_dft]
uy_ref = uy_ref[:,:,0:L_dft]
p_ref = p_ref[:,:,0:L_dft]

ux_ref[cylinder_mask,:] = np.NaN
uy_ref[cylinder_mask,:] = np.NaN
p_ref[cylinder_mask,:] = np.NaN

MAX_ux_ref = np.nanmax(np.abs(ux_ref.ravel()))
MAX_uy_ref = np.nanmax(np.abs(uy_ref.ravel()))
MAX_p_ref = np.nanmax(np.abs(p_ref.ravel()))

# compute reference reynolds stresses 
uxux_ref = np.mean(np.multiply(ux_ref,ux_ref),axis=2)
uxuy_ref = np.mean(np.multiply(ux_ref,uy_ref),axis=2)
uyuy_ref = np.mean(np.multiply(uy_ref,uy_ref),axis=2)

MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))

ind_MAX_uxux_ref = np.nanargmax(np.abs(uxux_ref.ravel()))
ind_MAX_uxuy_ref = np.nanargmax(np.abs(uxuy_ref.ravel()))
ind_MAX_uyuy_ref = np.nanargmax(np.abs(uyuy_ref.ravel()))

inds_dft = np.array([ind_MAX_uxux_ref,ind_MAX_uxuy_ref,ind_MAX_uyuy_ref])

# get the modes used in the fouriermodes file

fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
f_exported_modes = np.array(fourierModeFile['modeFrequencies'])

## plot of the spectrum of the flow 

from pinns_data_assimilation.lib.dft import dft

ux_modes,f_ux_modes = dft((np.reshape(ux_ref,[X_grid.shape[0]*X_grid.shape[1],ux_ref.shape[2]]))[inds_dft,:])
uy_modes,f_uy_modes = dft((np.reshape(uy_ref,[X_grid.shape[0]*X_grid.shape[1],ux_ref.shape[2]]))[inds_dft,:])
p_modes,f_p_modes = dft((np.reshape(p_ref,[X_grid.shape[0]*X_grid.shape[1],ux_ref.shape[2]]))[inds_dft,:])

# keep single sided spectrum, divide by L_DFT to normalize
half_index = int(L_dft/2)
f_ux_modes = f_ux_modes[0:half_index]
# scale the non-dimensional frequncies by the sampling rate
f_ux_modes = f_ux_modes*fs
ux_modes = ux_modes[:,0:half_index]/L_dft
uy_modes = uy_modes[:,0:half_index]/L_dft
p_modes = p_modes[:,0:half_index]/L_dft

# get which modes were exported
ind_exported_modes = np.zeros([6,1],dtype=np.int64)
for i in range(ind_exported_modes.shape[0]):
    ind_exported_modes[i,0]=np.argwhere(f_ux_modes==f_exported_modes[i])


ux_mode_inds = ind_exported_modes

shift=1E4
line_width=0.5
x_ticks = [0.0, 0.18382353, 2*0.18382353, 3*0.18382353, 4*0.18382353, 5*0.18382353, 6*0.18382353,7*0.18382353,8*0.18382353,1.7]
x_tick_labels = [ format(w, ".2f") for w in x_ticks]


## plot the reconstruction error 
# load the reference fourier reconstructions
uxux_rec_ref = []
uxuy_rec_ref = []
uyuy_rec_ref = []

rec_mode_vec =[0,1,2,3,4,5]
cases_supersample_factor = [0,2,4,8,16,32]

mean_err_ux_p = []
mean_err_uy_p = []
mean_err_p_p = []

p95_err_ux_p = []
p95_err_uy_p = []
p95_err_p_p = []

max_err_ux_p = []
max_err_uy_p = []
max_err_p_p = []

max_err_ux_t = []
inds_max_err_ux_t = []
ux_ref_t = []
ux_rec_t = []

mean_rec_err_uxux = []
mean_rec_err_uxuy = []
mean_rec_err_uyuy = []

p95_rec_err_uxux = []
p95_rec_err_uxuy = []
p95_rec_err_uyuy = []

max_rec_err_uxux = []
max_rec_err_uxuy = []
max_rec_err_uyuy = []
if True:

    for c in rec_mode_vec:
        recFile = h5py.File(rec_dir+'rec_fourier_c'+str(c)+'.h5','r')
        ux_rec_ref = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        uy_rec_ref = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        p_rec_ref = np.reshape(np.array(recFile['p']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        ux_rec_ref_m = np.reshape(np.mean(ux_rec_ref,axis=2),[ux_rec_ref.shape[0],ux_rec_ref.shape[1],1])
        uy_rec_ref_m = np.reshape(np.mean(uy_rec_ref,axis=2),[uy_rec_ref.shape[0],uy_rec_ref.shape[1],1])
        p_rec_ref_m = np.reshape(np.mean(p_rec_ref,axis=2),[p_rec_ref.shape[0],p_rec_ref.shape[1],1])

        ux_p = ux_rec_ref-ux_rec_ref_m
        uy_p = uy_rec_ref-uy_rec_ref_m
        p_p = p_rec_ref-p_rec_ref_m

        # instantaneous errors of the reconstructions

        mean_err_ux_p.append(np.nanmean(np.abs(ux_ref-ux_p)/MAX_ux_ref,(0,1,2)))
        mean_err_uy_p.append(np.nanmean(np.abs(uy_ref-uy_p)/MAX_uy_ref,(0,1,2)))
        mean_err_p_p.append(np.nanmean(np.abs(p_ref-p_p)/MAX_p_ref,(0,1,2)))

        p95_err_ux_p.append(np.nanpercentile(np.abs(ux_ref-ux_p)/MAX_ux_ref,95,(0,1,2)))
        p95_err_uy_p.append(np.nanpercentile(np.abs(uy_ref-uy_p)/MAX_uy_ref,95,(0,1,2)))
        p95_err_p_p.append(np.nanpercentile(np.abs(p_ref-p_p)/MAX_p_ref,95,(0,1,2)))

        max_err_ux_p.append(np.nanmax(np.abs(ux_ref-ux_p)/MAX_ux_ref,(0,1,2)))
        max_err_uy_p.append(np.nanmax(np.abs(uy_ref-uy_p)/MAX_uy_ref,(0,1,2)))
        max_err_p_p.append(np.nanmax(np.abs(p_ref-p_p)/MAX_p_ref,(0,1,2)))

        max_err_ux_t.append(np.nanmax(np.abs(ux_ref-ux_p)/MAX_ux_ref,(0,1)))
        inds_max_err_ux_t.append(np.nanargmax(np.reshape(np.abs(ux_ref-ux_p)/MAX_ux_ref,[ux_ref.shape[0]*ux_ref.shape[1],ux_ref.shape[2]]),axis=0))
        ux_ref_t.append(np.reshape(ux_ref,[ux_ref.shape[0]*ux_ref.shape[1],ux_ref.shape[2]])[35462,:])
        ux_rec_t.append(np.reshape(ux_p,[ux_ref.shape[0]*ux_ref.shape[1],ux_ref.shape[2]])[35462,:])

        # compute the stresses for the given truncation level
        uxux_rec_ref.append(np.mean(np.multiply(ux_p,ux_p),axis=2))
        uxuy_rec_ref.append(np.mean(np.multiply(ux_p,uy_p),axis=2))
        uyuy_rec_ref.append(np.mean(np.multiply(uy_p,uy_p),axis=2))
        uxux_rec_ref[c][cylinder_mask]=np.NaN
        uxuy_rec_ref[c][cylinder_mask]=np.NaN
        uyuy_rec_ref[c][cylinder_mask]=np.NaN

        mean_rec_err_uxux.append(np.nanmean((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel()))
        mean_rec_err_uxuy.append(np.nanmean((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel()))
        mean_rec_err_uyuy.append(np.nanmean((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel()))

        p95_rec_err_uxux.append(np.nanpercentile((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel(),95))
        p95_rec_err_uxuy.append(np.nanpercentile((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel(),95))
        p95_rec_err_uyuy.append(np.nanpercentile((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel(),95))

        max_rec_err_uxux.append(np.nanmax((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel()))
        max_rec_err_uxuy.append(np.nanmax((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel()))
        max_rec_err_uyuy.append(np.nanmax((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel()))


    # generate the contour profiles
    profile_locations = np.linspace(-1.5,9.5,22) # spacing of 0.5
    X_line_locations = X_grid_plot[:,0]
    X_distance_matrix = np.power(np.power(np.reshape(X_line_locations,[X_line_locations.size,1])-np.reshape(profile_locations,[1,profile_locations.size]),2.0),0.5)
    # find index of closest data line
    line_inds = np.argmin(X_distance_matrix,axis=0)
    profile_locations = X_grid_plot[line_inds,:]
    point_locations = Y_grid_plot[0,:]

    profile_x_offset = np.array([0,0,0])
    profile_x_scale = np.array([0.5,0.5,0.5])

if False:
    # additional plots to compare why 6 or modes are needed.

    max_err_ux_t = np.array(max_err_ux_t)
    print(np.nanmax(max_err_ux_t,1))
    print(inds_max_err_ux_t[2])

    plot.figure()
    plot.contourf(X_grid_plot,Y_grid_plot,uxuy_ref,levels=21)
    plot.plot((X_grid_plot.ravel())[inds_max_err_ux_t[2]],(Y_grid_plot.ravel())[inds_max_err_ux_t[2]],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    plot.plot((X_grid_plot.ravel())[35462],(Y_grid_plot.ravel())[35462],markersize=4,linewidth=0,color='r',marker='.',fillstyle='full',markeredgecolor='none')

    x_test = (X_grid_plot.ravel())[35462]
    y_test = (Y_grid_plot.ravel())[35462]


    ref_mode,ref_f = dft(ux_ref_t[0])
    plot.figure()
    plot.plot(ref_f[0:int(ref_f.shape[0]/2)],(np.abs(ref_mode))[0:int(ref_f.shape[0]/2)]/L_dft)
    plot.yscale('log')

    plot.figure()
    plot.plot(t,ux_ref_t[2])
    plot.plot(t,ux_rec_t[2])

    plot.figure()
    plot.plot(t,(ux_ref_t[2]-ux_rec_t[2])/MAX_ux_ref)

    plot.figure()
    plot.plot(t,ux_ref_t[5])
    plot.plot(t,ux_rec_t[5])

    plot.figure()
    plot.plot(t,(ux_ref_t[5]-ux_rec_t[5])/MAX_ux_ref)

x_test = (X_grid_plot.ravel())[35462]
y_test = (Y_grid_plot.ravel())[35462]

# load the raw data
raw_data_dir = 'C:/projects/pinns_local/data/mazi_fixed/'
raw_data_file = h5py.File(raw_data_dir+'raw_data.mat','r')
print(raw_data_file.keys())

x_raw = np.array(raw_data_file['x']).transpose()
print(x_raw.shape)
y_raw = np.array(raw_data_file['y']).transpose()
ux_raw = (np.array(raw_data_file['ux']).transpose())
ux_raw_long = 1.0*(ux_raw)
ux_raw = ux_raw[:,0:L_dft]
print(ux_raw.shape)
ux_p_raw = ux_raw - np.reshape(np.mean(ux_raw,axis=1),[ux_raw.shape[0],1])

x_ind_raw = np.argmin(np.power(np.power(x_raw-x_test,2.0)+np.power(y_raw-y_test,2.0),0.5))

temp_ux_raw = np.reshape(ux_raw[x_ind_raw,:],(L_dft,))
mean_temp_ux_raw = np.mean(temp_ux_raw,axis=0)
temp_ux_raw = temp_ux_raw - mean_temp_ux_raw
ux_raw_mode,raw_f = dft(temp_ux_raw)

if False:
    plot.figure()
    plot.plot(t,temp_ux_raw)

    plot.figure()
    plot.plot(ref_f[0:int(ref_f.shape[0]/2)],(np.abs(ref_mode))[0:int(ref_f.shape[0]/2)]/L_dft)
    plot.plot(ref_f[0:int(ref_f.shape[0]/2)],(np.abs(ux_raw_mode))[0:int(ref_f.shape[0]/2)]/L_dft)
    plot.yscale('log')

# to confirm, let's reconstruct the whole DNS and verify the maximum error for each mode number

ux_raw_mode,raw_f = dft(ux_p_raw)
ux_raw_mode = ux_raw_mode[:,0:int(L_dft/2)]/L_dft
raw_f = raw_f[0:int(L_dft/2)]*fs

raw_rec_inds = np.argsort(ux_raw_mode[x_ind_raw,:])[::-1]
print(raw_rec_inds[0:8])

ux_raw_rec = np.zeros(ux_raw.shape)

nrec_raw = 8
raw_rec_error_mean = np.zeros((nrec_raw,1))
raw_rec_error_max = np.zeros((nrec_raw,1))
raw_rec_nmodes = np.arange(nrec_raw,dtype=np.int64)

from pinns_data_assimilation.lib.dft import idft
print(ux_raw_mode[:,raw_rec_inds[0]].shape)
print(raw_f[raw_rec_inds[0]].shape)
for i in raw_rec_nmodes:
    ux_i,t_i = idft(np.reshape(ux_raw_mode[:,raw_rec_inds[i]],[ux_raw.shape[0],1]),f=raw_f[raw_rec_inds[i]],t=np.reshape(t,[1,t.size]),fs=fs)
    ux_raw_rec = ux_raw_rec+2.0*ux_i
    raw_rec_error_mean[i] = np.mean(np.abs(ux_raw_rec-ux_p_raw),axis=(0,1))
    raw_rec_error_max[i] = np.max(np.abs(ux_raw_rec-ux_p_raw),axis=(0,1))
    
    plot.figure()
    plot.plot(t,ux_p_raw[x_ind_raw,:])
    plot.plot(t,ux_raw_rec[x_ind_raw,:])

mean_err_ux_p = np.array(mean_err_ux_p)
max_err_ux_p = np.array(max_err_ux_p)

plot.figure()
plot.plot(raw_rec_nmodes-0.1,raw_rec_error_mean,linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
plot.plot(raw_rec_nmodes-0.1,raw_rec_error_max,linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
plot.plot(rec_mode_vec,mean_err_ux_p,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
plot.plot(rec_mode_vec,max_err_ux_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
plot.yscale('log')


dft_lengths = np.arange(4050,4099,dtype=np.int64)
mode1_amp = np.zeros((dft_lengths.shape[0],1))
mode2_amp = np.zeros((dft_lengths.shape[0],1))
mode3_amp = np.zeros((dft_lengths.shape[0],1))
mode4_amp = np.zeros((dft_lengths.shape[0],1))
mode5_amp = np.zeros((dft_lengths.shape[0],1))
mode6_amp = np.zeros((dft_lengths.shape[0],1))

sig_raw_long = np.reshape(ux_raw_long[x_ind_raw,:],(ux_raw_long.shape[1],))
for k in range(dft_lengths.shape[0]):
    temp_mode,temp_f = dft(sig_raw_long[0:dft_lengths[k]]-np.mean(sig_raw_long[0:dft_lengths[k]]))
    temp_inds_max = np.argsort(np.abs(temp_mode))[::-1]
    mode1_amp[k]=np.abs(temp_mode[temp_inds_max[0]])
    mode2_amp[k]=np.abs(temp_mode[temp_inds_max[1]])
    mode3_amp[k]=np.abs(temp_mode[temp_inds_max[2]])
    mode4_amp[k]=np.abs(temp_mode[temp_inds_max[3]])
    mode5_amp[k]=np.abs(temp_mode[temp_inds_max[4]])
    mode6_amp[k]=np.abs(temp_mode[temp_inds_max[5]])

fig,axs = plot.subplots(6,1)
axs[0].plot(dft_lengths,mode1_amp)
axs[1].plot(dft_lengths,mode2_amp)
axs[2].plot(dft_lengths,mode3_amp)
axs[3].plot(dft_lengths,mode4_amp)
axs[4].plot(dft_lengths,mode5_amp)
axs[5].plot(dft_lengths,mode6_amp)
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')
axs[3].set_yscale('log')
axs[4].set_yscale('log')
axs[5].set_yscale('log')

plot.show()

