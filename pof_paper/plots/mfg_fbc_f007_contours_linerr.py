

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

cases_list_f = []
cases_list_f.append(['mfg_fbcf007_f0_S0_j001_output/mfg_fbcf007_f0_S0_j001_ep153846_pred.mat','mfg_fbcf007_f0_S2_j001_output/mfg_fbcf007_f0_S2_j001_ep265734_pred.mat','mfg_fbcf007_f0_S4_j001_output/mfg_fbcf007_f0_S4_j001_ep164835_pred.mat','mfg_fbcf007_f0_S8_j001_output/mfg_fbcf007_f0_S8_j001_ep167832_pred.mat','mfg_fbcf007_f0_S16_j001_output/mfg_fbcf007_f0_S16_j001_ep175824_pred.mat','mfg_fbcf007_f0_S32_j001_output/mfg_fbcf007_f0_S32_j001_ep164835_pred.mat'])
cases_list_f.append(['mfg_fbcf007_f1_S0_j001_output/mfg_fbcf007_f1_S0_j001_ep153846_pred.mat','mfg_fbcf007_f1_S2_j001_output/mfg_fbcf007_f1_S2_j001_ep163836_pred.mat','mfg_fbcf007_f1_S4_j001_output/mfg_fbcf007_f1_S4_j001_ep164835_pred.mat','mfg_fbcf007_f1_S8_j001_output/mfg_fbcf007_f1_S8_j001_ep164835_pred.mat','mfg_fbcf007_f1_S16_j001_output/mfg_fbcf007_f1_S16_j001_ep164835_pred.mat','mfg_fbcf007_f1_S32_j001_output/mfg_fbcf007_f1_S32_j001_ep161838_pred.mat'])
cases_list_f.append(['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_pred.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_pred.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_pred.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_pred.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_pred.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_pred.mat'])
phys_list_f = []
phys_list_f.append(['mfg_fbcf007_f0_S0_j001_output/mfg_fbcf007_f0_S0_j001_ep153846_error.mat','mfg_fbcf007_f0_S2_j001_output/mfg_fbcf007_f0_S2_j001_ep265734_error.mat','mfg_fbcf007_f0_S4_j001_output/mfg_fbcf007_f0_S4_j001_ep164835_error.mat','mfg_fbcf007_f0_S8_j001_output/mfg_fbcf007_f0_S8_j001_ep167832_error.mat','mfg_fbcf007_f0_S16_j001_output/mfg_fbcf007_f0_S16_j001_ep175824_error.mat','mfg_fbcf007_f0_S32_j001_output/mfg_fbcf007_f0_S32_j001_ep164835_error.mat'])
phys_list_f.append(['mfg_fbcf007_f1_S0_j001_output/mfg_fbcf007_f1_S0_j001_ep153846_error.mat','mfg_fbcf007_f1_S2_j001_output/mfg_fbcf007_f1_S2_j001_ep163836_error.mat','mfg_fbcf007_f1_S4_j001_output/mfg_fbcf007_f1_S4_j001_ep164835_error.mat','mfg_fbcf007_f1_S8_j001_output/mfg_fbcf007_f1_S8_j001_ep164835_error.mat','mfg_fbcf007_f1_S16_j001_output/mfg_fbcf007_f1_S16_j001_ep164835_error.mat','mfg_fbcf007_f1_S32_j001_output/mfg_fbcf007_f1_S32_j001_ep161838_error.mat'])
phys_list_f.append(['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_error.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_error.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_error.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_error.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_error.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_error.mat'])
cases_supersample_factor = [0,2,4,8,16,32]


# load the reference data

# get the constants for all the modes
class UserScalingParameters(object):
    pass
ScalingParameters = UserScalingParameters()
ScalingParameters.mean = UserScalingParameters()
ScalingParameters.f =[]
ScalingParameters.f.append(UserScalingParameters())
ScalingParameters.f.append(UserScalingParameters())
ScalingParameters.f.append(UserScalingParameters())

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

ScalingParameters.mean.fs = 10.0
ScalingParameters.mean.MAX_x = 10.0
ScalingParameters.mean.MAX_y = 10.0
ScalingParameters.mean.MAX_p = 1.0
ScalingParameters.mean.nu_mol = 0.0066667

phi_xr_ref = []
phi_yr_ref = []
psi_r_ref = []

# load reference data
for mode_number in [0,1,2]:
    fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xr_ref.append(phi_xr)
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yr_ref.append(phi_yr)

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_r_ref.append(psi_r)

    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi

    ScalingParameters.f[mode_number].MAX_x = 20.0
    ScalingParameters.f[mode_number].MAX_y = 20.0 # we use the larger of the two spatial scalings
    ScalingParameters.f[mode_number].MAX_phi_xr = np.max(phi_xr.flatten())
    ScalingParameters.f[mode_number].MAX_phi_yr = np.max(phi_yr.flatten())
    ScalingParameters.f[mode_number].MAX_psi= 0.2*np.power((omega_0/omega),2.0) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
    ScalingParameters.f[mode_number].omega = omega
    ScalingParameters.f[mode_number].f = np.array(fourierModeFile['modeFrequencies'][mode_number])
    ScalingParameters.f[mode_number].nu_mol = 0.0066667

# now load the modes
phi_xr_pred = []
phi_yr_pred = []
psi_r_pred = []
mx_r_pred = []
my_r_pred = []
mass_r_pred = []
for s in range(len(cases_supersample_factor)):
    phi_xr_pred.append([])
    phi_yr_pred.append([])
    psi_r_pred.append([])
    mx_r_pred.append([])
    my_r_pred.append([])
    mass_r_pred.append([])

    for c in [0,1,2]:
        pred_f_file = h5py.File(output_dir+cases_list_f[c][s],'r')
        phys_f_file = h5py.File(output_dir+phys_list_f[c][s],'r')
        #print(output_dir+cases_list_f[c][s])
        phi_xr_pred[s].append(np.array(pred_f_file['pred'][:,0],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_xr)
        phi_yr_pred[s].append(np.array(pred_f_file['pred'][:,2],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_yr)
        psi_r_pred[s].append(np.array(pred_f_file['pred'][:,10],dtype=np.float64)*ScalingParameters.f[c].MAX_psi)
        mx_r_pred[s].append(np.array(phys_f_file['mxr']))
        my_r_pred[s].append(np.array(phys_f_file['myr']))
        mass_r_pred[s].append(np.array(phys_f_file['massr']))





cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))





for s in range(len(cases_supersample_factor)):
    for c in [0,1,2]:
        fig = plot.figure(figsize=(7,7))
        plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
        outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        inner = []
        # grid the data
        phi_xr_grid = np.reshape(phi_xr_ref[c],X_grid.shape)
        phi_yr_grid = np.reshape(phi_yr_ref[c],X_grid.shape)
        psi_r_grid = np.reshape(psi_r_ref[c],X_grid.shape)

        phi_xr_grid[cylinder_mask]=np.NaN
        phi_yr_grid[cylinder_mask]=np.NaN
        psi_r_grid[cylinder_mask]=np.NaN

        phi_xr_pred_grid = np.reshape(phi_xr_pred[s][c],X_grid.shape)
        phi_yr_pred_grid = np.reshape(phi_yr_pred[s][c],X_grid.shape)
        psi_r_pred_grid = np.reshape(psi_r_pred[s][c],X_grid.shape)

        phi_xr_pred_grid[cylinder_mask]=np.NaN
        phi_yr_pred_grid[cylinder_mask]=np.NaN
        psi_r_pred_grid[cylinder_mask]=np.NaN


        mx_grid = np.reshape(mx_r_pred[s][c],X_grid.shape)
        my_grid = np.reshape(my_r_pred[s][c],X_grid.shape)
        mass_grid = np.reshape(mass_r_pred[s][c],X_grid.shape)

        mx_grid[cylinder_mask]=np.NaN
        my_grid[cylinder_mask]=np.NaN
        mass_grid[cylinder_mask]=np.NaN

        # compute levels
        MAX_plot_phi_xr = np.nanmax(np.abs(phi_xr_grid.ravel()))
        MAX_plot_phi_yr = np.nanmax(np.abs(phi_yr_grid.ravel()))
        MAX_plot_psi_r = np.nanmax(np.abs(psi_r_grid.ravel()))

        phi_xr_err_grid = (phi_xr_grid - phi_xr_pred_grid)/MAX_plot_phi_xr
        phi_yr_err_grid = (phi_yr_grid - phi_yr_pred_grid)/MAX_plot_phi_yr
        psi_r_err_grid = (psi_r_grid - psi_r_pred_grid)/MAX_plot_psi_r

        MAX_phi_xr_err = np.nanmax(np.abs(phi_xr_err_grid.ravel()))
        MAX_phi_yr_err = np.nanmax(np.abs(phi_yr_err_grid.ravel()))
        MAX_psi_r_err = np.nanmax(np.abs(psi_r_err_grid.ravel()))

        MAX_mx = np.nanmax(np.abs(mx_grid.ravel()))
        MAX_my = np.nanmax(np.abs(my_grid.ravel()))
        MAX_mass = np.nanmax(np.abs(mass_grid.ravel()))

        levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
        levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
        levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)
        
        levels_phi_xr_err = np.linspace(-MAX_phi_xr_err,MAX_phi_xr_err,21)
        levels_phi_yr_err = np.linspace(-MAX_phi_yr_err,MAX_phi_yr_err,21)
        levels_psi_r_err = np.linspace(-MAX_psi_r_err,MAX_psi_r_err,21)

        levels_mx = np.linspace(-MAX_mx,MAX_mx,21)#np.geomspace(1E-3,1,21)##
        levels_my = np.linspace(-MAX_my,MAX_my,21)
        levels_mass = np.linspace(-MAX_mass,MAX_mass,21)

        ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

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
        ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(8,1.5,'$u_{x,DNS}$',fontsize=5)
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
        fig.add_subplot(ax)
        
        cax=plot.Subplot(fig,inner[0][1])
        cax.set(xmargin=0.5)
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.3f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
        

        ax = plot.Subplot(fig,inner[0][3])
        ux_plot=ax.contourf(X_grid,Y_grid,phi_xr_pred_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
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
        ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_err_grid,levels=levels_phi_xr_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(8,1.5,'$|\\frac{u_{x,DNS}-u_{x,PINN}}{max(u_{x,DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][7])
        cbar = plot.colorbar(ux_plot,cax,ticks=[-MAX_phi_xr_err,-MAX_phi_xr_err/2.0,0.0,MAX_phi_xr_err/2.0,MAX_phi_xr_err],format=tkr.FormatStrFormatter('%.2e'))
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
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][1])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)


        ax = plot.Subplot(fig,inner[1][3])
        uy_plot =ax.contourf(X_grid,Y_grid,phi_yr_pred_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
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
        uy_plot = ax.contourf(X_grid,Y_grid,phi_yr_err_grid,levels=levels_phi_yr_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        t=ax.text(8,1.5,'$|\\frac{u_{y,DNS}-u_{y,PINN}}{max(u_{y,DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][7])
        cbar = plot.colorbar(uy_plot,cax,ticks=[-MAX_phi_yr_err,-MAX_phi_yr_err/2.0,0.0,MAX_phi_yr_err/2.0,MAX_phi_yr_err],format=tkr.FormatStrFormatter('%.2e'))
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
        p_plot =ax.contourf(X_grid,Y_grid,psi_r_pred_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
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
        p_plot = ax.contourf(X_grid,Y_grid,psi_r_err_grid,levels=levels_psi_r_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_ylabel('y/D',fontsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(8,1.5,'$|\\frac{p_{DNS}-p_{PINN}}{max(p_{DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][7])
        cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_psi_r_err,-MAX_psi_r_err/2.0,0.0,MAX_psi_r_err/2.0,MAX_psi_r_err],format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        # quadrant 4

        inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

        ax = plot.Subplot(fig,inner[3][0])
        m_plot =ax.contourf(X_grid,Y_grid,mx_grid,levels=levels_mx,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')

        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.text(9,1.5,'$FANS_{x,r}$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][1])
        cbar = plot.colorbar(m_plot,cax,ticks=[-MAX_mx,-MAX_mx/2.0,0.0,MAX_mx/2.0,MAX_mx],format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[3][3])
        m_plot =ax.contourf(X_grid,Y_grid,my_grid,levels=levels_my,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')

        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.text(9,1.5,'$FANS_{y,r}$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][4])
        cbar = plot.colorbar(m_plot,cax,ticks=[-MAX_my,-MAX_my/2.0,0.0,MAX_my/2.0,MAX_my],format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[3][6])
        m_plot =ax.contourf(X_grid,Y_grid,mass_grid,levels=levels_mass,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_xlabel('x/D',fontsize=5)
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.text(9,1.5,'$C_r$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][7])
        cbar = plot.colorbar(m_plot,cax,ticks=[-MAX_mass,-MAX_mass/2.0,0.0,MAX_mass/2.0,MAX_mass],format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        plot.savefig(figures_dir+'linerr_mfg_fbc_f007_f'+str(c)+'_contours_S'+str(cases_supersample_factor[s])+'.pdf')
        plot.close(fig)

exit()



