

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import sys
sys.path.append('F:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

from pinns_data_assimilation.lib.file_util import extract_matching_integers
from pinns_data_assimilation.lib.file_util import find_highest_numbered_file
from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

# script

figures_dir = 'F:/projects/paper_figures/t010_f2/thesis/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

# load the reference data

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

ScalingParameters.mean.fs = 10.0
ScalingParameters.mean.MAX_x = 10.0
ScalingParameters.mean.MAX_y = 10.0
ScalingParameters.mean.MAX_p = 1.0
ScalingParameters.mean.nu_mol = 0.0066667

cylinder_mask_ref = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

phi_xr_ref = []
phi_xi_ref = []
phi_yr_ref = []
phi_yi_ref = []
psi_r_ref = []
psi_i_ref = []
tau_xxr_ref = []
tau_xyr_ref = []
tau_yyr_ref = []
tau_xxi_ref = []
tau_xyi_ref = []
tau_yyi_ref = []


fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
# load reference data
for mode_number in [0,1,2,3,4,5,]:
    ScalingParameters.f.append(UserScalingParameters())

    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xr[cylinder_mask_ref.ravel()] = np.NaN
    phi_xr = np.reshape(phi_xr,X_grid.shape)   
    phi_xr_ref.append(phi_xr)
    phi_xi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xi[cylinder_mask_ref.ravel()] = np.NaN
    phi_xi = np.reshape(phi_xi,X_grid.shape) 
    phi_xi_ref.append(phi_xi)
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yr[cylinder_mask_ref.ravel()] = np.NaN
    phi_yr = np.reshape(phi_yr,X_grid.shape) 
    phi_yr_ref.append(phi_yr)
    phi_yi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yi[cylinder_mask_ref.ravel()] = np.NaN
    phi_yi = np.reshape(phi_yi,X_grid.shape) 
    phi_yi_ref.append(phi_yi)

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_r[cylinder_mask_ref.ravel()] = np.NaN
    psi_r = np.reshape(psi_r,X_grid.shape) 
    psi_r_ref.append(psi_r)
    psi_i = np.array(np.imag(fourierModeFile['pressureModes'][:,mode_number]))
    psi_i[cylinder_mask_ref.ravel()] = np.NaN
    psi_i = np.reshape(psi_i,X_grid.shape) 
    psi_i_ref.append(psi_i)

    tau_xxr = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xxr[cylinder_mask_ref.ravel()] = np.NaN
    tau_xxr = np.reshape(tau_xxr,X_grid.shape)   
    tau_xxr_ref.append(tau_xxr)
    tau_xyr = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_xyr[cylinder_mask_ref.ravel()] = np.NaN
    tau_xyr = np.reshape(tau_xyr,X_grid.shape)   
    tau_xyr_ref.append(tau_xyr)
    tau_yyr = np.array(np.real(fourierModeFile['stressModes'][:,mode_number,2]))
    tau_yyr[cylinder_mask_ref.ravel()] = np.NaN
    tau_yyr = np.reshape(tau_yyr,X_grid.shape)   
    tau_yyr_ref.append(tau_yyr)

    tau_xxi = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,0]))
    tau_xxi[cylinder_mask_ref.ravel()] = np.NaN
    tau_xxi = np.reshape(tau_xxi,X_grid.shape)   
    tau_xxi_ref.append(tau_xxi)
    tau_xyi = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,1]))
    tau_xyi[cylinder_mask_ref.ravel()] = np.NaN
    tau_xyi = np.reshape(tau_xyi,X_grid.shape)   
    tau_xyi_ref.append(tau_xyi)
    tau_yyi = np.array(np.imag(fourierModeFile['stressModes'][:,mode_number,2]))
    tau_yyi[cylinder_mask_ref.ravel()] = np.NaN
    tau_yyi = np.reshape(tau_yyi,X_grid.shape)   
    tau_yyi_ref.append(tau_yyi)


    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi

    ScalingParameters.f[mode_number].MAX_x = 20.0
    ScalingParameters.f[mode_number].MAX_y = 20.0 # we use the larger of the two spatial scalings
    ScalingParameters.f[mode_number].MAX_phi_xr = np.nanmax(phi_xr.flatten())
    ScalingParameters.f[mode_number].MAX_phi_xi = np.nanmax(phi_xi.flatten())
    ScalingParameters.f[mode_number].MAX_plot_phi_x = np.nanmax(np.abs(phi_xr.flatten()+1j*phi_xi.flatten()))
    ScalingParameters.f[mode_number].MAX_phi_yr = np.nanmax(phi_yr.flatten())
    ScalingParameters.f[mode_number].MAX_phi_yi = np.nanmax(phi_yi.flatten())
    ScalingParameters.f[mode_number].MAX_plot_phi_y = np.nanmax(np.abs(phi_yr.flatten()+1j*phi_yi.flatten()))
    ScalingParameters.f[mode_number].MAX_psi= 0.2*np.power((omega_0/omega),2.0) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
    ScalingParameters.f[mode_number].MAX_plot_psi = np.nanmax(np.abs(psi_r.flatten()+1j*psi_i.flatten()))
    ScalingParameters.f[mode_number].MAX_plot_tau_xx = np.nanmax(np.abs(tau_xxr.flatten()+1j*tau_xxi.flatten()))
    ScalingParameters.f[mode_number].MAX_plot_tau_xy = np.nanmax(np.abs(tau_xyr.flatten()+1j*tau_xyi.flatten()))
    ScalingParameters.f[mode_number].MAX_plot_tau_yy = np.nanmax(np.abs(tau_yyr.flatten()+1j*tau_yyi.flatten()))
    ScalingParameters.f[mode_number].omega = omega
    ScalingParameters.f[mode_number].f = np.array(fourierModeFile['modeFrequencies'][mode_number])
    ScalingParameters.f[mode_number].nu_mol = 0.0066667


def contour_wcbar(fig,mid,plot_tuple):
    field,field_label,subplot_label,scale = plot_tuple

    levels = np.linspace(-scale,scale,21)
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid,wspace=0.0,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[len(inner)-1][0])
    axes.append(ax)
    ux_plot = ax.contourf(X_grid,Y_grid,field,levels=levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.text(7.5,1.2,field_label,fontsize=8)
    ax.text(-1.85,1.45,subplot_label,fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[len(inner)-1][1])
    caxes.append(cax)
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[scale,scale/2,0.0,-scale/2,-scale],format=tkr.FormatStrFormatter('%.1e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

if True:
       # mode 3 summary dual log version
    x_ticks = np.array([-2,0,2,4,6,8,10])

    # define the vector of quantity tuples for plotting
    p_quantities = [(phi_xr_ref[0],'$\hat{u}_{\mathrm{1x}}$','(a)',ScalingParameters.f[0].MAX_plot_phi_x),
                (phi_yr_ref[0],'$\hat{u}_{\mathrm{1y}}$','(b)',ScalingParameters.f[0].MAX_plot_phi_y),
                (psi_r_ref[0],'$\hat{p}_{\mathrm{1}}$','(c)',ScalingParameters.f[0].MAX_plot_psi),
                (tau_xxr_ref[0],'$\widehat{u\'_{\mathrm{1x}}u\'_{\mathrm{1x}}}$','(d)',ScalingParameters.f[0].MAX_plot_tau_xx),
                (tau_xyr_ref[0],'$\widehat{u\'_{\mathrm{1x}}u\'_{\mathrm{1y}}}$','(e)',ScalingParameters.f[0].MAX_plot_tau_xy),
                (tau_yyr_ref[0],'$\widehat{u\'_{\mathrm{1y}}u\'_{\mathrm{1y}}}$','(f)',ScalingParameters.f[0].MAX_plot_tau_yy),
                (phi_xr_ref[1],'$\hat{u}_{\mathrm{2x}}$','(g)',ScalingParameters.f[1].MAX_plot_phi_x),
                (phi_yr_ref[1],'$\hat{u}_{\mathrm{2y}}$','(h)',ScalingParameters.f[1].MAX_plot_phi_y),
                (psi_r_ref[1],'$\hat{p}_{\mathrm{2}}$','(i)',ScalingParameters.f[1].MAX_plot_psi),
                (tau_xxr_ref[1],'$\widehat{u\'_{\mathrm{2x}}u\'_{\mathrm{2x}}}$','(j)',ScalingParameters.f[1].MAX_plot_tau_xx),
                (tau_xyr_ref[1],'$\widehat{u\'_{\mathrm{2x}}u\'_{\mathrm{2y}}}$','(k)',ScalingParameters.f[1].MAX_plot_tau_xy),
                (tau_yyr_ref[1],'$\widehat{u\'_{\mathrm{2y}}u\'_{\mathrm{2y}}}$','(l)',ScalingParameters.f[1].MAX_plot_tau_yy),
                (phi_xr_ref[2],'$\hat{u}_{\mathrm{3y}}$','(m)',ScalingParameters.f[2].MAX_plot_phi_x),
                (phi_yr_ref[2],'$\hat{u}_{\mathrm{3y}}$','(n)',ScalingParameters.f[2].MAX_plot_phi_y),
                (psi_r_ref[2],'$\hat{p}_{\mathrm{3}}$','(o)',ScalingParameters.f[2].MAX_plot_psi),
                (tau_xxr_ref[2],'$\widehat{u\'_{\mathrm{3x}}u\'_{\mathrm{3x}}}$','(p)',ScalingParameters.f[2].MAX_plot_tau_xx),
                (tau_xyr_ref[2],'$\widehat{u\'_{\mathrm{3x}}u\'_{\mathrm{3y}}}$','(q)',ScalingParameters.f[2].MAX_plot_tau_xy),
                (tau_yyr_ref[2],'$\widehat{u\'_{\mathrm{3y}}u\'_{\mathrm{3y}}}$','(r)',ScalingParameters.f[2].MAX_plot_tau_yy),]

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(7.25,9))
    plot.subplots_adjust(left=0.03,top=0.99,right=0.91,bottom=0.05)
    outer = gridspec.GridSpec(3,1,wspace=0.1,hspace=0.1)
    mid = []
    inner = []

    axes = []
    caxes = []

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[0],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))   
    
    contour_wcbar(fig,mid[0][0],p_quantities[0])
    contour_wcbar(fig,mid[0][1],p_quantities[1])
    contour_wcbar(fig,mid[0][2],p_quantities[2])
    contour_wcbar(fig,mid[0][3],p_quantities[3])
    contour_wcbar(fig,mid[0][4],p_quantities[4])
    contour_wcbar(fig,mid[0][5],p_quantities[5])

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[1],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))  
    contour_wcbar(fig,mid[1][0],p_quantities[6])
    contour_wcbar(fig,mid[1][1],p_quantities[7])
    contour_wcbar(fig,mid[1][2],p_quantities[8])
    contour_wcbar(fig,mid[1][3],p_quantities[9])
    contour_wcbar(fig,mid[1][4],p_quantities[10])
    contour_wcbar(fig,mid[1][5],p_quantities[11])

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[2],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))  
    contour_wcbar(fig,mid[2][0],p_quantities[12])
    contour_wcbar(fig,mid[2][1],p_quantities[13])
    contour_wcbar(fig,mid[2][2],p_quantities[14])
    contour_wcbar(fig,mid[2][3],p_quantities[15])
    contour_wcbar(fig,mid[2][4],p_quantities[16])
    contour_wcbar(fig,mid[2][5],p_quantities[17])

    axes[16].set_xlabel('x',fontsize=8)
    axes[16].xaxis.set_tick_params(labelbottom=True)
    axes[17].set_xlabel('x',fontsize=8)
    axes[17].xaxis.set_tick_params(labelbottom=True)

      

    plot.savefig(figures_dir+'fourier_overview1.png',dpi=300)
    plot.close(fig)


    # define the vector of quantity tuples for plotting
    p_quantities = [(phi_xr_ref[3],'$\hat{u}_{\mathrm{4x}}$','(a)',ScalingParameters.f[3].MAX_plot_phi_x),
                (phi_yr_ref[3],'$\hat{u}_{\mathrm{4y}}$','(b)',ScalingParameters.f[3].MAX_plot_phi_y),
                (psi_r_ref[3],'$\hat{p}_{\mathrm{4}}$','(c)',ScalingParameters.f[3].MAX_plot_psi),
                (tau_xxr_ref[3],'$\widehat{u\'_{\mathrm{4x}}u\'_{\mathrm{4x}}}$','(d)',ScalingParameters.f[3].MAX_plot_tau_xx),
                (tau_xyr_ref[3],'$\widehat{u\'_{\mathrm{4x}}u\'_{\mathrm{4y}}}$','(e)',ScalingParameters.f[3].MAX_plot_tau_xy),
                (tau_yyr_ref[3],'$\widehat{u\'_{\mathrm{4y}}u\'_{\mathrm{4y}}}$','(f)',ScalingParameters.f[3].MAX_plot_tau_yy),
                (phi_xr_ref[4],'$\hat{u}_{\mathrm{5x}}$','(g)',ScalingParameters.f[4].MAX_plot_phi_x),
                (phi_yr_ref[4],'$\hat{u}_{\mathrm{5y}}$','(h)',ScalingParameters.f[4].MAX_plot_phi_y),
                (psi_r_ref[4],'$\hat{p}_{\mathrm{5}}$','(i)',ScalingParameters.f[4].MAX_plot_psi),
                (tau_xxr_ref[4],'$\widehat{u\'_{\mathrm{5x}}u\'_{\mathrm{5x}}}$','(j)',ScalingParameters.f[4].MAX_plot_tau_xx),
                (tau_xyr_ref[4],'$\widehat{u\'_{\mathrm{5x}}u\'_{\mathrm{5y}}}$','(k)',ScalingParameters.f[4].MAX_plot_tau_xy),
                (tau_yyr_ref[4],'$\widehat{u\'_{\mathrm{5y}}u\'_{\mathrm{5y}}}$','(l)',ScalingParameters.f[4].MAX_plot_tau_yy),
                (phi_xr_ref[5],'$\hat{u}_{\mathrm{6y}}$','(m)',ScalingParameters.f[5].MAX_plot_phi_x),
                (phi_yr_ref[5],'$\hat{u}_{\mathrm{6y}}$','(n)',ScalingParameters.f[5].MAX_plot_phi_y),
                (psi_r_ref[5],'$\hat{p}_{\mathrm{6}}$','(o)',ScalingParameters.f[5].MAX_plot_psi),
                (tau_xxr_ref[5],'$\widehat{u\'_{\mathrm{6x}}u\'_{\mathrm{6x}}}$','(p)',ScalingParameters.f[5].MAX_plot_tau_xx),
                (tau_xyr_ref[5],'$\widehat{u\'_{\mathrm{6x}}u\'_{\mathrm{6y}}}$','(q)',ScalingParameters.f[5].MAX_plot_tau_xy),
                (tau_yyr_ref[5],'$\widehat{u\'_{\mathrm{6y}}u\'_{\mathrm{6y}}}$','(r)',ScalingParameters.f[5].MAX_plot_tau_yy),]

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(7.25,9))
    plot.subplots_adjust(left=0.03,top=0.99,right=0.91,bottom=0.05)
    outer = gridspec.GridSpec(3,1,wspace=0.1,hspace=0.1)
    mid = []
    inner = []

    axes = []
    caxes = []

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[0],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))   
    
    contour_wcbar(fig,mid[0][0],p_quantities[0])
    contour_wcbar(fig,mid[0][1],p_quantities[1])
    contour_wcbar(fig,mid[0][2],p_quantities[2])
    contour_wcbar(fig,mid[0][3],p_quantities[3])
    contour_wcbar(fig,mid[0][4],p_quantities[4])
    contour_wcbar(fig,mid[0][5],p_quantities[5])

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[1],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))  
    contour_wcbar(fig,mid[1][0],p_quantities[6])
    contour_wcbar(fig,mid[1][1],p_quantities[7])
    contour_wcbar(fig,mid[1][2],p_quantities[8])
    contour_wcbar(fig,mid[1][3],p_quantities[9])
    contour_wcbar(fig,mid[1][4],p_quantities[10])
    contour_wcbar(fig,mid[1][5],p_quantities[11])

    mid.append(gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=outer[2],wspace=0.27,hspace=0.1,height_ratios=[1,1,1],width_ratios=[1,1]))  
    contour_wcbar(fig,mid[2][0],p_quantities[12])
    contour_wcbar(fig,mid[2][1],p_quantities[13])
    contour_wcbar(fig,mid[2][2],p_quantities[14])
    contour_wcbar(fig,mid[2][3],p_quantities[15])
    contour_wcbar(fig,mid[2][4],p_quantities[16])
    contour_wcbar(fig,mid[2][5],p_quantities[17])

    axes[16].set_xlabel('x',fontsize=8)
    axes[16].xaxis.set_tick_params(labelbottom=True)
    axes[17].set_xlabel('x',fontsize=8)
    axes[17].xaxis.set_tick_params(labelbottom=True)

      

    plot.savefig(figures_dir+'fourier_overview2.png',dpi=300)
    plot.close(fig)