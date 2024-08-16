
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('F:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

figures_dir = 'F:/projects/paper_figures/reconstruction/'
rec_dir = 'F:/projects/paper_figures/data/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

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
ind_exported_modes = np.zeros([8,1],dtype=np.int64)
for i in range(ind_exported_modes.shape[0]):
    ind_exported_modes[i,0]=np.argwhere(f_ux_modes==f_exported_modes[i])


ux_mode_inds = ind_exported_modes

shift=1E4
line_width=0.5
x_ticks = [0.0, 0.18382353, 2*0.18382353, 3*0.18382353, 4*0.18382353, 5*0.18382353, 6*0.18382353,7*0.18382353,8*0.18382353,1.7]
x_tick_labels = [ format(w, ".2f") for w in x_ticks]

# plot the spectra
fig,axs = plot.subplots(3,1)
fig.set_size_inches(3.37,5.5)
plot.subplots_adjust(left=0.08,top=0.95,right=0.95,bottom=0.08)
uxux_plot, = axs[0].plot(f_ux_modes,np.abs(ux_modes[0,:]*shift),linewidth=line_width,color='k')
uxuy_plot, =axs[0].plot(f_ux_modes,np.abs(ux_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
uyuy_plot, =axs[0].plot(f_ux_modes,np.abs(ux_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
axs[0].plot(f_ux_modes[ux_mode_inds],np.abs(ux_modes[0,ux_mode_inds])*shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[0].plot(f_ux_modes[ux_mode_inds],np.abs(ux_modes[1,ux_mode_inds]),linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[0].plot(f_ux_modes[ux_mode_inds],np.abs(ux_modes[2,ux_mode_inds])/shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
selected_modes_f = f_ux_modes[ux_mode_inds]
selected_modes_a = np.abs(ux_modes[0,ux_mode_inds])*shift
for m in range(selected_modes_f.shape[0]):
    axs[0].text(selected_modes_f[m]+0.05,0.5*selected_modes_a[m],str(m+1),fontsize=8)

axs[0].text(1.2,np.abs(ux_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
axs[0].text(1.2,np.abs(ux_modes[1,1])*1E2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
axs[0].text(1.2,np.abs(ux_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)

axs[0].xaxis.set_tick_params(labelbottom=False)
axs[0].yaxis.set_tick_params(labelleft=False)
axs[0].yaxis.set_tick_params(left=False)
axs[0].set_ylabel('$abs(\Phi_x)$',fontsize=8)
axs[0].set_yscale('log')
axs[0].set_xlim(0,1.7)
axs[0].set_ylim(1E-10,1E4)
axs[0].set_xticks(x_ticks)
#axs[0].legend([uxux_plot,uxuy_plot,uyuy_plot],['at max($\overline{u\'_xu\'_x}$)','at max($\overline{u\'_xu\'_y}$)','at max($\overline{u\'_yu\'_y}$)'],fontsize=8)
axs[0].text(-0.1,shift*0.5,'(a)',fontsize=8)

axs[1].plot(f_ux_modes,np.abs(uy_modes[0,:])*shift,linewidth=line_width,color='k')
axs[1].plot(f_ux_modes,np.abs(uy_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
axs[1].plot(f_ux_modes,np.abs(uy_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
axs[1].xaxis.set_tick_params(labelbottom=False)
axs[1].yaxis.set_tick_params(labelleft=False)
axs[1].yaxis.set_tick_params(left=False)
axs[1].set_ylabel('$abs(\Phi_y)$',fontsize=8)
axs[1].plot(f_ux_modes[ux_mode_inds],np.abs(uy_modes[0,ux_mode_inds])*shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[1].plot(f_ux_modes[ux_mode_inds],np.abs(uy_modes[1,ux_mode_inds]),linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[1].plot(f_ux_modes[ux_mode_inds],np.abs(uy_modes[2,ux_mode_inds])/shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[1].text(1.2,np.abs(uy_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
axs[1].text(1.2,np.abs(uy_modes[1,1])*1E2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
axs[1].text(1.2,np.abs(uy_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)
axs[1].set_yscale('log')
axs[1].set_xlim(0,1.7)
axs[1].set_ylim(1E-10,1E4)
axs[1].set_xticks(x_ticks)
axs[1].text(-0.1,shift*0.5,'(b)',fontsize=8)

axs[2].plot(f_ux_modes,np.abs(p_modes[0,:])*shift,linewidth=line_width,color='k')
axs[2].plot(f_ux_modes,np.abs(p_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
axs[2].plot(f_ux_modes,np.abs(p_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
axs[2].plot(f_ux_modes[ux_mode_inds],np.abs(p_modes[0,ux_mode_inds])*shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[2].plot(f_ux_modes[ux_mode_inds],np.abs(p_modes[1,ux_mode_inds]),linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[2].plot(f_ux_modes[ux_mode_inds],np.abs(p_modes[2,ux_mode_inds])/shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs[2].text(1.2,np.abs(p_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
axs[2].text(1.2,np.abs(p_modes[1,1])*2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
axs[2].text(1.2,np.abs(p_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)
axs[2].set_yscale('log')
axs[2].set_xlabel('Frequency (St)',fontsize=8)
axs[2].set_ylabel('$abs(\Psi)$',fontsize=8)
axs[2].yaxis.set_tick_params(labelleft=False)
axs[2].yaxis.set_tick_params(left=False)
axs[2].set_xlim(0,1.7)
axs[2].set_ylim(1E-10,1E4)
axs[2].set_xticks(x_ticks)
axs[2].set_xticklabels(x_tick_labels,fontsize=8)
axs[2].text(-0.1,shift*0.5,'(c)',fontsize=8)

plot.savefig(figures_dir+'ref_spectrum.pdf')
plot.savefig(figures_dir+'ref_spectrum.png',dpi=300)
plot.close(fig)       

# plot the spectra
fig,axs = plot.subplots(1,1)
fig.set_size_inches(3.37,2.5)
plot.subplots_adjust(left=0.08,top=0.95,right=0.95,bottom=0.15)

uxuy_plot, =axs.plot(f_ux_modes,np.abs(ux_modes[1,:])*shift,linewidth=line_width,linestyle='-',color='k')
axs.plot(f_ux_modes[ux_mode_inds],np.abs(ux_modes[1,ux_mode_inds])*shift,linewidth=0.0,marker='o',color='k',markersize=1.0)
axs.plot(f_ux_modes,np.abs(uy_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
axs.plot(f_ux_modes[ux_mode_inds],np.abs(uy_modes[1,ux_mode_inds]),linewidth=0.0,marker='o',color='k',markersize=1.0)
axs.plot(f_ux_modes,np.abs(p_modes[1,:])/shift,linewidth=line_width,linestyle='-',color='k')
axs.plot(f_ux_modes[ux_mode_inds],np.abs(p_modes[1,ux_mode_inds])/shift,linewidth=0.0,marker='o',color='k',markersize=1.0)

selected_modes_f = f_ux_modes[ux_mode_inds]
selected_modes_a = np.abs(ux_modes[1,ux_mode_inds])*shift
for m in range(selected_modes_f.shape[0]):
    axs.text(selected_modes_f[m]+0.05,0.5*selected_modes_a[m],str(m+1),fontsize=8)

axs.text(1.6,np.abs(ux_modes[1,1]*shift*1E1),'$\Phi_x$',fontsize=8)
axs.text(1.6,np.abs(uy_modes[1,1])*1E1,'$\Phi_y$',fontsize=8)
axs.text(1.6,np.abs(p_modes[1,1]/shift)*1E1,'$\psi$',fontsize=8)

axs.yaxis.set_tick_params(labelleft=False)
axs.yaxis.set_tick_params(left=False)
axs.set_ylabel('$abs(\psi),abs(\Phi_y),abs(\Phi_x)$',fontsize=8)
axs.set_yscale('log')
axs.set_xlim(0,1.7)
axs.set_ylim(1E-10,1E4)
axs.set_xticks(x_ticks)
axs.set_xticklabels(x_tick_labels,fontsize=8)
axs.set_xlabel('Frequency (St)',fontsize=8)

plot.savefig(figures_dir+'ref_spectrum_short.pdf')
plot.savefig(figures_dir+'ref_spectrum_short.png',dpi=300)
plot.close(fig)       



## plot the reconstruction error 
# load the reference fourier reconstructions
uxux_rec_ref = []
uxuy_rec_ref = []
uyuy_rec_ref = []

rec_mode_vec =[0,1,2,3,4,5,6,7]
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
    s=0 # needed so no points appear in the reference comparison plotss
    for c in rec_mode_veF:

        if True:

            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
            MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
            MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
            
            levels_uxux = 1.1*np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
            levels_uxuy = 1.1*np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
            levels_uyuy = 1.1*np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)
            
            levels_err = np.geomspace(1E-3,1,21)
            ticks_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

            if cases_supersample_factor[s]>0:
                linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

                x_downsample = x[linear_downsample_inds]
                y_downsample = y[linear_downsample_inds]
                valid_inds = (np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d).ravel()
                x_downsample = x_downsample[valid_inds]
                y_downsample = y_downsample[valid_inds]

            # quadrant 1

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            # (1,(1,1))
            ax = plot.Subplot(fig,inner[0][0])
            ux_plot = ax.contourf(X_grid,Y_grid,uxux_ref,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{DNS}$',fontsize=8)
            ax.text(-1.75,1.5,'(aa)',fontsize=8,color='k')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec_ref[c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{FMD}$',fontsize=8)
            ax.text(-1.75,1.5,'(ab)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs((uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{FMD}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=8,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.1e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_ref,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{DNS}$',fontsize=8)
            ax.text(-1.75,1.5,'(ba)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec_ref[c],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{FMD}$',fontsize=8)
            ax.text(-1.75,1.5,'(bb)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{FMD}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=8,color='k')
            ax.text(-1.75,1.5,'(bc)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_ref,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{DNS}$',fontsize=8)
            ax.text(-1.75,1.5,'(ca)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec_ref[c],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{FMD}$',fontsize=8)
            ax.text(-1.75,1.5,'(cb)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=8)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{FMD}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=8,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxux_ref[line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxux_rec_ref[c][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.text(-1.75,1.5,'(da)',fontsize=8,color='k')
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{x}}_{DNS}$','$\overline{u\'_{x}u\'_{x}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=8,framealpha=0.0)
            fig.add_subplot(ax)
        
            ax = plot.Subplot(fig,inner[3][3])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxuy_ref[line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxuy_rec_ref[c][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(-1.75,1.5,'(db)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{y}}_{DNS}$','$\overline{u\'_{x}u\'_{y}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=8,framealpha=0.0)
            fig.add_subplot(ax)

            ax = plot.Subplot(fig,inner[3][6])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uyuy_ref[line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uyuy_rec_ref[c][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=8)
            ax.text(-1.75,1.5,'(dc)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{y}u\'_{y}}_{DNS}$','$\overline{u\'_{y}u\'_{y}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=8,framealpha=0.0)
            fig.add_subplot(ax)

            # for now empty ...

            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_recref_c'+str(c)+'.pdf')
            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_recref_c'+str(c)+'.png',dpi=300)
            plot.close('all')




if True:

    mean_rec_err_uxux = np.array(mean_rec_err_uxux)
    mean_rec_err_uxuy = np.array(mean_rec_err_uxux)
    mean_rec_err_uyuy = np.array(mean_rec_err_uyuy)

    p95_rec_err_uxux = np.array(p95_rec_err_uxux)
    p95_rec_err_uxuy = np.array(p95_rec_err_uxux)
    p95_rec_err_uyuy = np.array(p95_rec_err_uyuy)

    max_rec_err_uxux = np.array(max_rec_err_uxux)
    max_rec_err_uxuy = np.array(max_rec_err_uxux)
    max_rec_err_uyuy = np.array(max_rec_err_uyuy)

    error_x_tick_labels = ['1','2','3','4','5','6','7','8']
    plot_mode_vec = np.array([1,2,3,4,5,6,7,8])
    error_y_tick_labels = ['1E-4','1E-3','1E-2','1E-1']
    # compare the different number of modes by Restress error
    fig,axs = plot.subplots(3,1)
    fig.set_size_inches(3.37,5.5)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.09)

    mean_plt,=axs[0].plot(plot_mode_vec,mean_rec_err_uxux,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    p95_plt,=axs[0].plot(plot_mode_vec,p95_rec_err_uxux,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    max_plt,=axs[0].plot(plot_mode_vec,max_rec_err_uxux,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[0].set_xticks(plot_mode_vec)
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[0].set_yscale('log')
    axs[0].set_ylim(5E-5,5E-1)
    axs[0].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=8)
    axs[0].set_ylabel("Relative Error",fontsize=8)
    axs[0].set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs[0].legend([mean_plt,p95_plt,max_plt],['Mean','95th Percentile','Max'],fontsize=8)
    axs[0].grid('on')
    axs[0].text(-0.5,0.5,'(a)',fontsize=10)

    axs[1].plot(plot_mode_vec,mean_rec_err_uxuy,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[1].plot(plot_mode_vec,p95_rec_err_uxuy,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[1].plot(plot_mode_vec,max_rec_err_uxuy,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[1].set_xticks(plot_mode_vec)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[1].set_yscale('log')
    axs[1].set_ylim(5E-5,5E-1)
    axs[1].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=8)
    axs[1].set_ylabel("Relative Error",fontsize=8)
    axs[1].set_title('$\overline{u\'_{x}u\'_{y}}$')
    axs[1].grid('on')
    axs[1].text(-0.5,0.5,'(b)',fontsize=10)

    axs[2].plot(plot_mode_vec,mean_rec_err_uyuy,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[2].plot(plot_mode_vec,p95_rec_err_uyuy,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[2].plot(plot_mode_vec,max_rec_err_uyuy,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[2].set_xticks(plot_mode_vec)
    axs[2].set_xticklabels(error_x_tick_labels)
    axs[2].set_yscale('log')
    axs[2].set_ylim(5E-5,5E-1)
    axs[2].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=8)
    axs[2].set_xlabel('Number of Fourier Modes',fontsize=8)
    axs[2].set_ylabel("Relative Error",fontsize=8)
    axs[2].set_title('$\overline{u\'_{y}u\'_{y}}$')
    axs[2].grid('on')
    axs[2].text(-0.5,0.5,'(c)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_truncation_error.pdf')
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_truncation_error.png',dpi=300)
    plot.close(fig)

    # compare the different number of modes by Restress error
    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3.0)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.15)

    error_y_tick_labels = ['1E-4','1E-3','1E-2','1E-1','1']

    mean_plt_uxux,=axs.plot(plot_mode_vec-0.1,mean_rec_err_uxux,linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_uxux,=axs.plot(plot_mode_vec-0.1,max_rec_err_uxux,linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uxuy,=axs.plot(plot_mode_vec,mean_rec_err_uxuy,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uxuy,=axs.plot(plot_mode_vec,max_rec_err_uxuy,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    mean_plt_uyuy,=axs.plot(plot_mode_vec+0.1,mean_rec_err_uyuy,linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt_uyuy,=axs.plot(plot_mode_vec+0.1,max_rec_err_uyuy,linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')
    axs.set_xticks(plot_mode_vec)
    axs.set_xticklabels(error_x_tick_labels,fontsize=8)
    axs.set_yscale('log')
    axs.set_ylim(5E-5,5E-1)
    axs.set_yticks([1E-4,1E-3,1E-2,1E-1,1],labels=error_y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    axs.legend([mean_plt_uxux,max_plt_uxux,mean_plt_uxuy,max_plt_uxuy,mean_plt_uyuy,max_plt_uyuy,],['Mean $\overline{u_xu_x}$','Max $\overline{u_xu_x}$','Mean $\overline{u_xu_y}$','Max $\overline{u_xu_y}$','Mean $\overline{u_yu_y}$','Max $\overline{u_yu_y}$'],fontsize=8)
    axs.grid('on')
    axs.set_xlabel('Number of Fourier Modes',fontsize=8)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_truncation_error_short.pdf')
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_truncation_error_short.png',dpi=300)
    plot.close(fig)    

# instantanteous truncation error of the primative quantities

if True:
    mean_err_ux_p = np.array(mean_err_ux_p)
    mean_err_uy_p = np.array(mean_err_uy_p)
    mean_err_p_p = np.array(mean_err_p_p)

    p95_err_ux_p = np.array(p95_err_ux_p)
    p95_err_uy_p = np.array(p95_err_uy_p)
    p95_err_p_p = np.array(p95_err_p_p)

    max_err_ux_p = np.array(max_err_ux_p)
    max_err_uy_p = np.array(max_err_uy_p)
    max_err_p_p = np.array(max_err_p_p)

    error_x_tick_labels = ['1','2','3','4','5','6','7','8']
    plot_mode_vec = np.array([1,2,3,4,5,6,7,8])
    error_y_ticks = [1E-3,1E-2,1E-1,1E0]
    error_y_tick_labels = ['1E-3','1E-2','1E-1','1']
    # compare the different number of modes by Restress error
    fig,axs = plot.subplots(3,1)
    fig.set_size_inches(3.37,5.5)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.09)

    mean_plt,=axs[0].plot(plot_mode_vec,mean_err_ux_p,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    p95_plt,=axs[0].plot(plot_mode_vec,p95_err_ux_p,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    max_plt,=axs[0].plot(plot_mode_vec,max_err_ux_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[0].set_xticks(plot_mode_vec)
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[0].set_yscale('log')
    axs[0].set_ylim(5E-4,1)
    axs[0].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    axs[0].set_ylabel("Relative Error",fontsize=8)
    axs[0].set_title('$u\'_{x}$')
    axs[0].legend([mean_plt,p95_plt,max_plt],['Mean','95th Percentile','Max'],fontsize=8)
    axs[0].grid('on')
    axs[0].text(-0.5,1,'(a)',fontsize=10)

    axs[1].plot(plot_mode_vec,mean_err_uy_p,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[1].plot(plot_mode_vec,p95_err_uy_p,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[1].plot(plot_mode_vec,max_err_uy_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[1].set_xticks(plot_mode_vec)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[1].set_yscale('log')
    axs[1].set_ylim(5E-4,1)
    axs[1].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    axs[1].set_ylabel("Relative Error",fontsize=8)
    axs[1].set_title('$u\'_{y}$')
    axs[1].grid('on')
    axs[1].text(-0.5,1,'(b)',fontsize=10)

    axs[2].plot(plot_mode_vec,mean_err_p_p,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[2].plot(plot_mode_vec,p95_err_p_p,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[2].plot(plot_mode_vec,max_err_p_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[2].set_xticks(plot_mode_vec)
    axs[2].set_xticklabels(error_x_tick_labels)
    axs[2].set_yscale('log')
    axs[2].set_ylim(5E-4,1)
    axs[2].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    axs[2].set_xlabel('Number of Fourier Modes',fontsize=8)
    axs[2].set_ylabel("Relative Error",fontsize=8)
    axs[2].set_title('$p\'$')
    axs[2].grid('on')
    axs[2].text(-0.5,1,'(c)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_rec_instantaneous_truncation_error.pdf')
    plot.savefig(figures_dir+'logerr_rec_instantaneous_truncation_error.png',dpi=300)
    plot.close(fig)


    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.15)

    error_y_ticks = [1E-3,1E-2,1E-1,1E0]
    error_y_tick_labels = ['1E-3','1E-2','1E-1','1',]

    mean_plt_ux,=axs.plot(plot_mode_vec-0.1,mean_err_ux_p,linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_ux,=axs.plot(plot_mode_vec-0.1,max_err_ux_p,linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uy,=axs.plot(plot_mode_vec,mean_err_uy_p,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uy,=axs.plot(plot_mode_vec,max_err_uy_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    mean_plt_p,=axs.plot(plot_mode_vec+0.1,mean_err_p_p,linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt_p,=axs.plot(plot_mode_vec+0.1,max_err_p_p,linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')
    axs.set_xticks(plot_mode_vec)
    axs.set_xticklabels(error_x_tick_labels,fontsize=8)
    axs.set_yscale('log')
    axs.set_ylim(5E-4,5E0)
    axs.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    axs.legend([mean_plt_ux,max_plt_ux,mean_plt_uy,max_plt_uy,mean_plt_p,max_plt_p],["Mean $u'_x$","Max $u'_x$","Mean $u'_y$","Max $u'_y$","Mean $p'$","Max $p'$",],fontsize=8,ncols=2)
    axs.grid('on')
    axs.set_xlabel('Number of Fourier Modes',fontsize=8)
    plot.savefig(figures_dir+'logerr_rec_instantaneous_truncation_error_short.pdf')
    plot.savefig(figures_dir+'logerr_rec_instantaneous_truncation_error_short.png',dpi=300)
    plot.close(fig)     

print(max_err_ux_p)
exit()


dx = [] # array for supersample spacing
for s in range(len(cases_supersample_factor)):
    dx.append([])
    for c in [0,1,2]:
        if cases_supersample_factor[s]==0:
            dx[s].append(X_grid[1,0]-X_grid[0,0])

        if cases_supersample_factor[s]>0:
            linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

            x_downsample = x[linear_downsample_inds]
            y_downsample = y[linear_downsample_inds]

            x_ds_grid = (np.reshape(x_downsample,(ndy,ndx))).transpose()
            y_ds_grid = (np.reshape(y_downsample,(ndy,ndx))).transpose()
            dx[s].append(x_ds_grid[1,0]-x_ds_grid[0,0])

pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,2]

# load the reconstuctions and compute the reynolds stresses
ux_p_rec = []
uy_p_rec = []
p_p_rec = []


for s in range(len(cases_supersample_factor)):
    ux_p_rec.append([])
    uy_p_rec.append([])
    p_p_rec.append([])
    for c in [0,1,2]:
        recFile = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.h5','r')
        ux_rec = (np.array(recFile['ux']))[inds_dft,:]
        uy_rec = (np.array(recFile['uy']))[inds_dft,:]
        p_rec = (np.array(recFile['p']))[inds_dft,:]
        ux_rec_m = np.mean(ux_rec,axis=1)
        uy_rec_m = np.mean(uy_rec,axis=1)  
        p_rec_m = np.mean(uy_rec,axis=1)
        # subtract the reconstructed mean  
        ux_p_rec[s].append(ux_rec-np.reshape(ux_rec_m,[ux_rec_m.shape[0],1]))
        uy_p_rec[s].append(uy_rec-np.reshape(uy_rec_m,[uy_rec_m.shape[0],1]))
        p_p_rec[s].append(p_rec-np.reshape(p_rec_m,[p_rec_m.shape[0],1]))
        # subtract the true mean
        #ux_p_rec[s].append(ux_rec-np.reshape(ux[inds_dft],[ux_rec_m.shape[0],1]))
        #uy_p_rec[s].append(uy_rec-np.reshape(uy[inds_dft],[uy_rec_m.shape[0],1]))
        #p_p_rec[s].append(p_rec-np.reshape(p[inds_dft],[p_rec_m.shape[0],1]))

for s in range(len(cases_supersample_factor)):
    for c in [2]:
        ux_p_modes,f_modes = dft(ux_p_rec[s][c])
        uy_p_modes,f_modes = dft(uy_p_rec[s][c])
        p_p_modes,f_modes = dft(p_p_rec[s][c])

        ux_p_modes = ux_p_modes[:,0:half_index]/L_dft
        uy_p_modes = uy_p_modes[:,0:half_index]/L_dft
        p_p_modes = p_p_modes[:,0:half_index]/L_dft

        # get which modes were exported
        ind_exported_modes = np.zeros([3,1],dtype=np.int64)
        for i in range(ind_exported_modes.shape[0]):
            ind_exported_modes[i,0]=np.argwhere(f_ux_modes==f_exported_modes[i])


        ux_mode_inds = ind_exported_modes

        shift=1E4
        line_width=0.5
        x_ticks = [0.0, 0.18382353, 2*0.18382353, 3*0.18382353, 4*0.18382353, 5*0.18382353, 6*0.18382353,7*0.18382353,8*0.18382353,1.7]
        x_tick_labels = [ format(w, ".2f") for w in x_ticks]

        # plot the spectra
        fig,axs = plot.subplots(3,1)
        fig.set_size_inches(3.37,6)
        plot.subplots_adjust(left=0.08,top=0.95,right=0.95,bottom=0.08)
        uxux_plot, = axs[0].plot(f_ux_modes,np.abs(ux_modes[0,:]*shift),linewidth=line_width,color='k')
        uxuy_plot, =axs[0].plot(f_ux_modes,np.abs(ux_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        uyuy_plot, =axs[0].plot(f_ux_modes,np.abs(ux_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        uxux_p_plot, = axs[0].plot(f_ux_modes,np.abs(ux_p_modes[0,:]*shift),linewidth=line_width,linestyle=':',color='r')
        uxuy_p_plot, =axs[0].plot(f_ux_modes,np.abs(ux_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        uyuy_p_plot, =axs[0].plot(f_ux_modes,np.abs(ux_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')
        axs[0].text(1.2,np.abs(ux_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
        axs[0].text(1.2,np.abs(ux_modes[1,1])*1E2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
        axs[0].text(1.2,np.abs(ux_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)

        #selected_modes_f = f_ux_modes[ux_mode_inds]
        #selected_modes_a = np.abs(ux_modes[0,ux_mode_inds])*shift
        #for m in range(selected_modes_f.shape[0]):
        #    axs[0].text(selected_modes_f[m]+0.05,0.5*selected_modes_a[m],str(m+1),fontsize=8)

        axs[0].xaxis.set_tick_params(labelbottom=False)
        axs[0].yaxis.set_tick_params(labelleft=False)
        axs[0].yaxis.set_tick_params(left=False)
        axs[0].set_ylabel('$abs(\Phi_x)$',fontsize=8)
        axs[0].set_yscale('log')
        axs[0].set_xlim(0,1.7)
        axs[0].set_ylim(1E-10,1E4)
        axs[0].set_xticks(x_ticks)
        #axs[0].legend([uxux_plot,uxuy_plot,uyuy_plot],['at max($\overline{u\'_xu\'_x}$)','at max($\overline{u\'_xu\'_y}$)','at max($\overline{u\'_yu\'_y}$)'],fontsize=8)
        axs[0].text(-0.1,shift*0.5,'(a)',fontsize=8)

        axs[1].plot(f_ux_modes,np.abs(uy_modes[0,:])*shift,linewidth=line_width,color='k')
        axs[1].plot(f_ux_modes,np.abs(uy_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        axs[1].plot(f_ux_modes,np.abs(uy_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        axs[1].plot(f_ux_modes,np.abs(uy_p_modes[0,:])*shift,linewidth=line_width,linestyle=':',color='r')
        axs[1].plot(f_ux_modes,np.abs(uy_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        axs[1].plot(f_ux_modes,np.abs(uy_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')
        axs[1].xaxis.set_tick_params(labelbottom=False)
        axs[1].yaxis.set_tick_params(labelleft=False)
        axs[1].yaxis.set_tick_params(left=False)
        axs[1].set_ylabel('$abs(\Phi_y)$',fontsize=8)
        axs[1].text(1.2,np.abs(uy_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
        axs[1].text(1.2,np.abs(uy_modes[1,1])*1E2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
        axs[1].text(1.2,np.abs(uy_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)
        axs[1].set_yscale('log')
        axs[1].set_ylim(1E-10,1E4)
        axs[1].set_xlim(0,1.7)
        axs[1].set_xticks(x_ticks)
        axs[1].text(-0.1,shift*0.5,'(b)',fontsize=8)

        axs[2].plot(f_ux_modes,np.abs(p_modes[0,:])*shift,linewidth=line_width,color='k')
        axs[2].plot(f_ux_modes,np.abs(p_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        axs[2].plot(f_ux_modes,np.abs(p_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        axs[2].plot(f_ux_modes,np.abs(p_p_modes[0,:])*shift,linewidth=line_width,linestyle=':',color='r')
        axs[2].plot(f_ux_modes,np.abs(p_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        axs[2].plot(f_ux_modes,np.abs(p_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')
        axs[2].text(1.2,np.abs(p_modes[0,1]*shift*1E2),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
        axs[2].text(1.2,np.abs(p_modes[1,1])*1E2,'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
        axs[2].text(1.2,np.abs(p_modes[2,1]/shift)*1E2,'at $max(\overline{u\'_yu\'_y})$',fontsize=8)
        axs[2].set_yscale('log')
        axs[2].set_xlabel('Frequency (St)',fontsize=8)
        axs[2].set_ylabel('$abs(\Psi)$',fontsize=8)
        axs[2].yaxis.set_tick_params(labelleft=False)
        axs[2].yaxis.set_tick_params(left=False)
        axs[2].set_xlim(0,1.7)
        axs[2].set_ylim(1E-10,1E4)
        axs[2].set_xticks(x_ticks)
        axs[2].set_xticklabels(x_tick_labels,fontsize=8,rotation=45,ha='right')
        axs[2].text(-0.1,shift*0.5,'(c)',fontsize=8)

        plot.savefig(figures_dir+'ref_pinn_S'+str(cases_supersample_factor[s])+'_spectrum.pdf')
        plot.savefig(figures_dir+'ref_pinn_S'+str(cases_supersample_factor[s])+'_spectrum.png',dpi=300)
        plot.close(fig)      

# combined plot showing all s*
fig,axs = plot.subplots(6,3)
fig.set_size_inches(6.69,8.5)
plot.subplots_adjust(left=0.07,top=0.95,right=0.85,bottom=0.08)

letter_indices = ['a','b','c','d','e','f']

pts_per_d_titles = ['40','20','10','5','2.5','1.25']

for s in range(len(cases_supersample_factor)):
    for c in [2]:
        ux_p_modes,f_modes = dft(ux_p_rec[s][c])
        uy_p_modes,f_modes = dft(uy_p_rec[s][c])
        p_p_modes,f_modes = dft(p_p_rec[s][c])

        ux_p_modes = ux_p_modes[:,0:half_index]/L_dft
        uy_p_modes = uy_p_modes[:,0:half_index]/L_dft
        p_p_modes = p_p_modes[:,0:half_index]/L_dft

        # get which modes were exported
        ind_exported_modes = np.zeros([3,1],dtype=np.int64)
        for i in range(ind_exported_modes.shape[0]):
            ind_exported_modes[i,0]=np.argwhere(f_ux_modes==f_exported_modes[i])


        ux_mode_inds = ind_exported_modes

        shift=1E4
        line_width=0.5
        x_ticks = [0.0, 0.18382353, 2*0.18382353, 3*0.18382353, 4*0.18382353]
        x_tick_labels = [ format(w, ".2f") for w in x_ticks]

        # plot the spectra
        uxux_plot, = axs[s,0].plot(f_ux_modes,np.abs(ux_modes[0,:]*shift),linewidth=line_width,color='k')
        uxuy_plot, =axs[s,0].plot(f_ux_modes,np.abs(ux_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        uyuy_plot, =axs[s,0].plot(f_ux_modes,np.abs(ux_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        uxux_p_plot, = axs[s,0].plot(f_ux_modes,np.abs(ux_p_modes[0,:]*shift),linewidth=line_width,linestyle=':',color='r')
        uxuy_p_plot, =axs[s,0].plot(f_ux_modes,np.abs(ux_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        uyuy_p_plot, =axs[s,0].plot(f_ux_modes,np.abs(ux_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')

        #selected_modes_f = f_ux_modes[ux_mode_inds]
        #selected_modes_a = np.abs(ux_modes[0,ux_mode_inds])*shift
        #for m in range(selected_modes_f.shape[0]):
        #    axs[s,0].text(selected_modes_f[m]+0.05,0.5*selected_modes_a[m],str(m+1),fontsize=8)

        axs[s,0].xaxis.set_tick_params(labelbottom=False)
        axs[s,0].yaxis.set_tick_params(labelleft=False)
        axs[s,0].yaxis.set_tick_params(left=False)

        axs[s,0].set_yscale('log')
        axs[s,0].set_xlim(0,0.8)
        axs[s,0].set_ylim(1E-10,1E4)
        axs[s,0].set_xticks(x_ticks)
        #axs[s,0].legend([uxux_plot,uxuy_plot,uyuy_plot],['at max($\overline{u\'_xu\'_x}$)','at max($\overline{u\'_xu\'_y}$)','at max($\overline{u\'_yu\'_y}$)'],fontsize=8)
        axs[s,0].text(-0.12,shift,'('+letter_indices[s]+'a)',fontsize=8)
        #
        #axs[s,0].title.set_size(7)

        axs[s,1].plot(f_ux_modes,np.abs(uy_modes[0,:])*shift,linewidth=line_width,color='k')
        axs[s,1].plot(f_ux_modes,np.abs(uy_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        axs[s,1].plot(f_ux_modes,np.abs(uy_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        axs[s,1].plot(f_ux_modes,np.abs(uy_p_modes[0,:])*shift,linewidth=line_width,linestyle=':',color='r')
        axs[s,1].plot(f_ux_modes,np.abs(uy_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        axs[s,1].plot(f_ux_modes,np.abs(uy_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')
        axs[s,1].xaxis.set_tick_params(labelbottom=False)
        axs[s,1].yaxis.set_tick_params(labelleft=False)
        axs[s,1].yaxis.set_tick_params(left=False)
        axs[s,1].set_yscale('log')
        axs[s,1].set_ylim(1E-10,1E4)
        axs[s,1].set_xlim(0,0.8)
        axs[s,1].set_xticks(x_ticks)
        axs[s,1].text(-0.12,shift,'('+letter_indices[s]+'b)',fontsize=8)
        

        axs[s,2].plot(f_ux_modes,np.abs(p_modes[0,:])*shift,linewidth=line_width,color='k')
        axs[s,2].plot(f_ux_modes,np.abs(p_modes[1,:]),linewidth=line_width,linestyle='-',color='k')
        axs[s,2].plot(f_ux_modes,np.abs(p_modes[2,:])/shift,linewidth=line_width,linestyle='-',color='k')
        axs[s,2].plot(f_ux_modes,np.abs(p_p_modes[0,:])*shift,linewidth=line_width,linestyle=':',color='r')
        axs[s,2].plot(f_ux_modes,np.abs(p_p_modes[1,:]),linewidth=line_width,linestyle=':',color='r')
        axs[s,2].plot(f_ux_modes,np.abs(p_p_modes[2,:])/shift,linewidth=line_width,linestyle=':',color='r')

        axs[s,2].set_yscale('log')
        axs[s,2].yaxis.set_tick_params(labelleft=False)
        axs[s,2].xaxis.set_tick_params(labelbottom=False)
        axs[s,2].yaxis.set_tick_params(left=False)
        axs[s,2].set_xlim(0,0.8)
        axs[s,2].set_ylim(1E-10,1E4)
        axs[s,2].set_xticks(x_ticks)
        
        axs[s,2].text(-0.12,shift,'('+letter_indices[s]+'c)',fontsize=8)

axs[5,0].set_xlabel('Frequency (St)',fontsize=8)
axs[5,1].set_xlabel('Frequency (St)',fontsize=8)
axs[5,2].set_xlabel('Frequency (St)',fontsize=8)
axs[5,0].xaxis.set_tick_params(labelbottom=True)
axs[5,1].xaxis.set_tick_params(labelbottom=True)
axs[5,2].xaxis.set_tick_params(labelbottom=True)
axs[5,0].set_xticklabels(x_tick_labels,fontsize=8)
axs[5,1].set_xticklabels(x_tick_labels,fontsize=8)
axs[5,2].set_xticklabels(x_tick_labels,fontsize=8)

axs[0,0].title.set_text('$abs(\Phi_x)$')
axs[0,0].title.set_size(8)
axs[0,1].title.set_text('$abs(\Phi_y)$')
axs[0,1].title.set_size(8)
axs[0,2].title.set_text('$abs(\Psi)$')
axs[0,2].title.set_size(8)

for s in range(len(cases_supersample_factor)):
    axs[s,0].set_ylabel('$\Delta x/D='+pts_per_d_titles[s]+'$',fontsize=8)
    axs[s,2].text(0.82,np.abs(p_modes[0,1]*shift),'at $max(\overline{u\'_xu\'_x})$',fontsize=8)
    axs[s,2].text(0.82,np.abs(p_modes[1,1]),'at $max(\overline{u\'_xu\'_y})$',fontsize=8)
    axs[s,2].text(0.82,np.abs(p_modes[2,1]/shift),'at $max(\overline{u\'_yu\'_y})$',fontsize=8)

plot.savefig(figures_dir+'ref_pinn_allS_spectrum.pdf')
plot.savefig(figures_dir+'ref_pinn_allS_spectrum.png',dpi=300)
plot.close(fig)     

exit()
mean_err_uxux = []
mean_err_uxuy = []
mean_err_uyuy = []

p95_err_uxux = []
p95_err_uxuy = []
p95_err_uyuy = []

max_err_uxux = []
max_err_uxuy = []
max_err_uyuy = []

for s in range(len(cases_supersample_factor)):

    mean_err_uxux.append([])
    mean_err_uxuy.append([])
    mean_err_uyuy.append([])

    p95_err_uxux.append([])
    p95_err_uxuy.append([])
    p95_err_uyuy.append([])

    max_err_uxux.append([])
    max_err_uxuy.append([])
    max_err_uyuy.append([])
    for c in rec_mode_vec:
        err_uxux = (np.abs(uxux_ref-uxux_rec[s][c])/MAX_uxux_ref).ravel()
        err_uxuy = (np.abs(uxuy_ref-uxuy_rec[s][c])/MAX_uxuy_ref).ravel()
        err_uyuy = (np.abs(uyuy_ref-uyuy_rec[s][c])/MAX_uyuy_ref).ravel()

        mean_err_uxux[s].append(np.nanmean(err_uxux))
        mean_err_uxuy[s].append(np.nanmean(err_uxuy))
        mean_err_uyuy[s].append(np.nanmean(err_uyuy))

        p95_err_uxux[s].append(np.nanpercentile(err_uxux,95))
        p95_err_uxuy[s].append(np.nanpercentile(err_uxuy,95))
        p95_err_uyuy[s].append(np.nanpercentile(err_uyuy,95))

        max_err_uxux[s].append(np.nanmax(err_uxux))
        max_err_uxuy[s].append(np.nanmax(err_uxuy))
        max_err_uyuy[s].append(np.nanmax(err_uyuy))



