

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

from pinns_data_assimilation.lib.file_util import extract_matching_integers
from pinns_data_assimilation.lib.file_util import find_highest_numbered_file
from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

# script

figures_dir = 'C:/projects/paper_figures/mean_field/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

cases_list = ['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_pred.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_pred.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_pred.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_pred.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_pred.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_pred.mat']
errs_list = ['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_error.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_error.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_error.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_error.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_error.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_error.mat']
cases_supersample_factor = [0,2,4,8,16,32]



# load the reference data
base_dir = data_dir
mode_number=2

fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

X_grid_plot = X_grid
Y_grid_plot = Y_grid
X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)

X = np.stack((x,y),axis=1)
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

p = np.array(meanPressureFile['meanPressure'])

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
print('Mode Frequency:',str(omega/(2*np.pi)))

class UserScalingParameters(object):
    pass

ScalingParameters = UserScalingParameters()
ScalingParameters.fs = fs
ScalingParameters.MAX_x = 20.0
ScalingParameters.MAX_y = 20.0 # we use the larger of the two spatial scalings
ScalingParameters.mean_MAX_x = 10.0
ScalingParameters.mean_MAX_y = 10.0 # if the scaling is different it should be specified here
ScalingParameters.MAX_ux = np.max(ux.flatten())
ScalingParameters.MAX_uy = np.max(uy.flatten())
ScalingParameters.MIN_x = -2.0
ScalingParameters.MIN_y = -2.0
ScalingParameters.MIN_ux = np.min(ux.flatten())
ScalingParameters.MIN_uy = np.min(uy.flatten())
ScalingParameters.MAX_uxppuxpp = np.max(uxux.flatten())
ScalingParameters.MAX_uxppuypp = np.max(uxuy.flatten())
ScalingParameters.MAX_uyppuypp = np.max(uyuy.flatten())

ScalingParameters.MAX_phi_xr = np.max(phi_xr.flatten())
ScalingParameters.MAX_phi_xi = np.max(phi_xi.flatten())
ScalingParameters.MAX_phi_yr = np.max(phi_yr.flatten())
ScalingParameters.MAX_phi_yi = np.max(phi_yi.flatten())

ScalingParameters.MAX_tau_xx_r = np.max(tau_xx_r.flatten())
ScalingParameters.MAX_tau_xx_i = np.max(tau_xx_i.flatten())
ScalingParameters.MAX_tau_xy_r = np.max(tau_xy_r.flatten())
ScalingParameters.MAX_tau_xy_i = np.max(tau_xy_i.flatten())
ScalingParameters.MAX_tau_yy_r = np.max(tau_yy_r.flatten())
ScalingParameters.MAX_tau_yy_i = np.max(tau_yy_i.flatten())

ScalingParameters.MAX_p= 1 # estimated maximum pressure, we should 
ScalingParameters.MAX_psi= 0.2*np.power((omega_0/omega),2.0) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
ScalingParameters.nu_mol = 0.0066667


cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))
phi_xr_grid = np.reshape(phi_xr,X_grid.shape)
phi_xr_grid[cylinder_mask] = np.NaN
phi_yr_grid = np.reshape(phi_yr,X_grid.shape)
phi_yr_grid[cylinder_mask] = np.NaN
psi_r_grid = np.reshape(psi_r,X_grid.shape)
psi_r_grid[cylinder_mask] = np.NaN

n_cases = len(cases_list)

MAX_plot_phi_xr = np.nanmax(np.abs(phi_xr.flatten()))
MAX_plot_phi_yr = np.nanmax(np.abs(phi_xr.flatten()))
MAX_plot_psi_r = np.nanmax(np.abs(psi_r.flatten()))

# contour plot levels
levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)

MAX_err_ux_all = 0.0
MAX_err_uy_all = 0.0
MAX_err_p_all = 0.0
MIN_err_ux_all = np.NaN
MIN_err_uy_all = np.NaN
MIN_err_p_all = np.NaN

phi_xr_pred = []
phi_yr_pred = []
psi_r_pred = []

mx_grid = []
my_grid = []
mass_grid = []

phi_xr_pred_grid = []
phi_yr_pred_grid = []
psi_r_pred_grid = []

phi_xr_err_grid = []
phi_yr_err_grid = []
psi_r_err_grid = []


for c in range(n_cases):
    predFile = h5py.File(output_dir+cases_list[c],'r')

    phi_xr_pred.append(np.array(predFile['pred'][:,0])*ScalingParameters.MAX_phi_xr)
    phi_yr_pred.append(np.array(predFile['pred'][:,2])*ScalingParameters.MAX_phi_yr)
    psi_r_pred.append(np.array(predFile['pred'][:,10])*ScalingParameters.MAX_psi)

    phi_xr_pred_grid.append(np.reshape(phi_xr_pred[c],X_grid.shape)) #pred_test_grid = np.copy(np.reshape(pred_test,[X_grid_plot.shape[0],X_grid_plot.shape[1],12]))
    phi_xr_pred_grid[c][cylinder_mask] = np.NaN
    phi_yr_pred_grid.append(np.reshape(phi_yr_pred[c],X_grid.shape))
    phi_yr_pred_grid[c][cylinder_mask] = np.NaN
    psi_r_pred_grid.append(np.reshape(psi_r_pred[c],X_grid.shape))
    psi_r_pred_grid[c][cylinder_mask] = np.NaN

    phi_xr_err_grid.append(phi_xr_grid-phi_xr_pred_grid[c])
    phi_yr_err_grid.append(phi_yr_grid-phi_yr_pred_grid[c])
    psi_r_err_grid.append(psi_r_grid-psi_r_pred_grid[c])

    physFile = h5py.File(output_dir+errs_list[c],'r')
    mx_grid.append(np.reshape(np.array(physFile['mxr']),X_grid.shape))
    mx_grid[c][cylinder_mask] = np.NaN
    my_grid.append(np.reshape(np.array(physFile['myr']),X_grid.shape))
    my_grid[c][cylinder_mask] = np.NaN
    mass_grid.append(np.reshape(np.array(physFile['massr']),X_grid.shape))
    mass_grid[c][cylinder_mask] = np.NaN





fig = plot.figure(figsize=(7,7))
plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
inner = []

for c in range(len(cases_list)):
    MAX_mx = np.nanmax(np.abs(mx_grid[c].ravel()))
    MAX_my = np.nanmax(np.abs(my_grid[c].ravel()))
    MAX_mass = np.nanmax(np.abs(mass_grid[c].ravel()))

    levels_mx = np.geomspace(1E-3,1,21)##
    levels_my = np.geomspace(1E-3,1,21)#np.linspace(1E-6,MAX_my,21)
    levels_mass = np.geomspace(1E-3,1,21)#np.linspace(1E-6,MAX_mass,21)

    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

    if cases_supersample_factor[c]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[c],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample = x[linear_downsample_inds]
        y_downsample = y[linear_downsample_inds]
        valid_inds = (np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d).ravel()

        x_downsample = x_downsample[valid_inds]
        y_downsample = y_downsample[valid_inds]

    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{x,DNS}$',fontsize=5)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    fig.add_subplot(ax)
    
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
    

    ax = plot.Subplot(fig,inner[0][3])
    ux_plot=ax.contourf(X_grid,Y_grid,phi_xr_pred_grid[c],levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{x,PINN}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][4])
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[0][6])
    ux_plot = ax.contourf(X_grid,Y_grid,np.abs(phi_xr_err_grid[c]/MAX_plot_phi_xr)+1E-30,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['nipy_spectral'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$|\\frac{u_{x,DNS}-u_{x,PINN}}{max(u_{x,DNS})}|$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][7])
    cbar = plot.colorbar(ux_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.1e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)


    # quadrant 2

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

    ax = plot.Subplot(fig,inner[1][0])
    uy_plot =ax.contourf(X_grid,Y_grid,phi_yr_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{y,DNS}$',fontsize=5)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)


    ax = plot.Subplot(fig,inner[1][3])
    uy_plot =ax.contourf(X_grid,Y_grid,phi_yr_pred_grid[c],levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{y,PINN},$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][4])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[1][6])
    uy_plot = ax.contourf(X_grid,Y_grid,np.abs(phi_yr_err_grid[c]/MAX_plot_phi_yr)+1E-30,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['nipy_spectral'],extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    t=ax.text(8,1.5,'$|\\frac{u_{y,DNS}-u_{y,PINN}}{max(u_{y,DNS})}|$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][7])
    cbar = plot.colorbar(uy_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
 


    # quadrant 3

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

    ax = plot.Subplot(fig,inner[2][0])
    p_plot =ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(9,1.5,'$p_{DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
    
    ax = plot.Subplot(fig,inner[2][3])
    p_plot =ax.contourf(X_grid,Y_grid,psi_r_pred_grid[c],levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(9,1.5,'$p_{PINN}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][4])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[2][6])
    p_plot = ax.contourf(X_grid,Y_grid,np.abs(psi_r_err_grid[c]/MAX_plot_psi_r)+1E-30,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['nipy_spectral'],extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_ylabel('y/D',fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_xlabel('x/D',fontsize=5)
    ax.text(8,1.5,'$|\\frac{p_{DNS}-p_{PINN}}{max(p_{DNS})}|$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][7])
    cbar = plot.colorbar(p_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    # quadrant 4

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

    ax = plot.Subplot(fig,inner[3][0])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$NS_{x}$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(m_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][3])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$NS_{y}$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][4])
    cbar = plot.colorbar(m_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][6])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
    ax.set_aspect('equal')
    ax.set_xlabel('x/D',fontsize=5)
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$C$',fontsize=5,color='w')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][7])
    cbar = plot.colorbar(m_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'mfg_fb_f007_f2_contours_S'+str(cases_supersample_factor[c])+'.pdf')

exit()



