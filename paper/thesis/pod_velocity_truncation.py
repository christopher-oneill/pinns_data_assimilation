

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
from pinns_data_assimilation.lib.dft import dft

# read the data

base_dir = HOMEDIR+'data/mazi_fixed_grid/'
time_data_dir = 'F:/projects/fixed_cylinder/grid/data/'
figures_dir = 'F:/projects/paper_figures/t010_f2/thesis/'

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
POD_file = h5py.File(base_dir+'POD_data16.mat','r')

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

ux_ref = np.reshape(ux_ref,[X_grid.shape[0],X_grid.shape[1],ux_ref.shape[1]])
uy_ref = np.reshape(uy_ref,[X_grid.shape[0],X_grid.shape[1],uy_ref.shape[1]])

L_dft=4082


ux_ref = ux_ref[:,:,0:L_dft]
uy_ref = uy_ref[:,:,0:L_dft]

cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

ux_ref[cylinder_mask,:] = np.NaN
uy_ref[cylinder_mask,:] = np.NaN

# maximum values for normalization
MAX_ux_ref = np.nanmax(np.abs(ux_ref))
MAX_uy_ref = np.nanmax(np.abs(uy_ref))

# compute the reference reynolds stresses

uxux_ref = np.mean(np.power(ux_ref,2.0),2)
uxuy_ref = np.mean(ux_ref*uy_ref,2)
uyuy_ref = np.mean(np.power(uy_ref,2.0),2)

MAX_uxux_ref = np.nanmax(np.abs(uxux_ref))
MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref))
MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref))


print(uxux_ref.shape)
print(uxuy_ref.shape)
print(uyuy_ref.shape)

Phi = np.array(POD_file['Phi'])
Ak = np.array(POD_file['Ak'])
#Ak = Ak[0:L_dft,:] # truncate to the same length as the fourier example
t = np.arange(0,Ak.shape[0])/fs

phi_xr = np.reshape(Phi[0:x.shape[0],:],[X_grid.shape[0],X_grid.shape[1],Phi.shape[1]])
phi_yr = np.reshape(Phi[x.shape[0]:2*x.shape[0],:],[X_grid.shape[0],X_grid.shape[1],Phi.shape[1]])

n_modes = Phi.shape[1]

mean_err_ux = np.zeros([n_modes,1])
mean_err_uy = np.zeros([n_modes,1])
mean_err_uxux = np.zeros([n_modes,1])
mean_err_uxuy = np.zeros([n_modes,1])
mean_err_uyuy = np.zeros([n_modes,1])

max_err_ux = np.zeros([n_modes,1])
max_err_uy = np.zeros([n_modes,1])
max_err_uxux = np.zeros([n_modes,1])
max_err_uxuy = np.zeros([n_modes,1])
max_err_uyuy = np.zeros([n_modes,1])

ux_temp = np.zeros(ux_ref.shape)
uy_temp = np.zeros(uy_ref.shape)

reconstruct_modes = np.arange(0,n_modes)

for i in reconstruct_modes:
    ux_temp = ux_temp + np.reshape(phi_xr[:,:,i],[phi_xr.shape[0],phi_xr.shape[1],1])*np.reshape(Ak[:,i],[1,1,L_dft])
    uy_temp = uy_temp + np.reshape(phi_yr[:,:,i],[phi_yr.shape[0],phi_yr.shape[1],1])*np.reshape(Ak[:,i],[1,1,L_dft])

    uxux_temp = np.mean(np.power(ux_temp,2.0),2)
    uxuy_temp = np.mean(ux_temp*uy_temp,2)
    uyuy_temp = np.mean(np.power(uy_temp,2.0),2)
    
    err_ux = ux_temp - ux_ref
    err_uy = uy_temp - uy_ref
    
    mean_err_ux[i] = np.nanmean(np.abs(err_ux))/MAX_ux_ref
    mean_err_uy[i] = np.nanmean(np.abs(err_uy))/MAX_uy_ref
    max_err_ux[i] =np.nanmax(np.abs(err_ux))/MAX_ux_ref
    max_err_uy[i] =np.nanmax(np.abs(err_uy))/MAX_uy_ref

    err_uxux = uxux_temp - uxux_ref
    err_uxuy = uxuy_temp - uxuy_ref
    err_uyuy = uyuy_temp - uyuy_ref

    mean_err_uxux[i] = np.nanmean(np.abs(err_uxux))/MAX_uxux_ref
    mean_err_uxuy[i] = np.nanmean(np.abs(err_uxuy))/MAX_uxuy_ref
    mean_err_uyuy[i] = np.nanmean(np.abs(err_uyuy))/MAX_uyuy_ref
    max_err_uxux[i] =np.nanmax(np.abs(err_uxux))/MAX_uxux_ref
    max_err_uxuy[i] =np.nanmax(np.abs(err_uxuy))/MAX_uxuy_ref
    max_err_uyuy[i] =np.nanmax(np.abs(err_uyuy))/MAX_uyuy_ref



# plot the truncation
if True:
    plot_mode_pairs = 2*np.arange(0,n_modes//2)+1
    x_mode_pairs = np.arange(0,n_modes//2)+1

    fig = plot.figure(figsize=(7.5,3))
    plot.subplots_adjust(left=0.08,top=0.99,right=0.98,bottom=0.13)
    outer = gridspec.GridSpec(1,2,wspace=0.18,hspace=0.12)

    error_y_ticks = [1E-6,1E-4,1E-2,1E0]
    error_y_tick_labels = ['1E-6','1E-4','1E-2','1',]



    ax = plot.Subplot(fig,outer[0])

    mean_plt_ux,=ax.plot(x_mode_pairs-0.1,mean_err_uxux[plot_mode_pairs],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_ux,=ax.plot(x_mode_pairs-0.1,max_err_uxux[plot_mode_pairs],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uy,=ax.plot(x_mode_pairs,mean_err_uxuy[plot_mode_pairs],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uy,=ax.plot(x_mode_pairs,max_err_uxuy[plot_mode_pairs],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    mean_plt_p,=ax.plot(x_mode_pairs+0.1,mean_err_uyuy[plot_mode_pairs],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt_p,=ax.plot(x_mode_pairs+0.1,max_err_uyuy[plot_mode_pairs],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')
    ax.set_xticks(x_mode_pairs)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_yscale('log')
    ax.set_ylim(1E-7,5E0)
    ax.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    ax.set_ylabel("Error ($\eta$)",fontsize=8)
    ax.legend([mean_plt_ux,max_plt_ux,mean_plt_uy,max_plt_uy,mean_plt_p,max_plt_p],["Mean $\overline{u'_xu'_x}$","Max $\overline{u'_xu'_x}$","Mean $\overline{u'_xu'_y}$","Max $\overline{u'_xu'_y}$","Mean $\overline{u'_yu'_y}$","Max $\overline{u'_yu'_y}$",],fontsize=8,ncols=2)
    ax.grid('on')
    ax.set_xlabel('Number of POD Mode Pairs',fontsize=8)
    ax.text(1,2.0,'(a)',fontsize=8,color='k')
    fig.add_subplot(ax)

    error_y_ticks = [1E-4,1E-3,1E-2,1E-1,1E0]
    error_y_tick_labels = ['1E-4','1E-3','1E-2','1E-1','1',]

    ax = plot.Subplot(fig,outer[1])

    mean_plt_ux,=ax.plot(x_mode_pairs-0.05,mean_err_ux[plot_mode_pairs],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_ux,=ax.plot(x_mode_pairs-0.05,max_err_ux[plot_mode_pairs],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uy,=ax.plot(x_mode_pairs+0.05,mean_err_ux[plot_mode_pairs],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uy,=ax.plot(x_mode_pairs+0.05,max_err_ux[plot_mode_pairs],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    ax.set_xticks(x_mode_pairs)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_yscale('log')
    ax.set_ylim(5E-5,5E0)
    ax.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    #ax.set_ylabel("Error ($\eta$)",fontsize=8)
    ax.legend([mean_plt_ux,max_plt_ux,mean_plt_uy,max_plt_uy,],["Mean $u'_x$","Max $u'_x$","Mean $u'_y$","Max $u'_y$",],fontsize=8,ncols=2)
    ax.grid('on')
    ax.set_xlabel('Number of POD Mode Pairs',fontsize=8)
    ax.text(1,2.0,'(b)',fontsize=8,color='k')
    fig.add_subplot(ax)

    plot.savefig(figures_dir+'POD_truncation.pdf')
    plot.savefig(figures_dir+'POD_truncation.png',dpi=600)
    plot.close(fig)       