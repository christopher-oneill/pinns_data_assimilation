

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

figures_dir = 'F:/projects/paper_figures/modes/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

cases_supersample_factor = [0,2,4,8,16,32]
cases_frequency = [0,1,2,3,4,5]

cases_list_f = []
phys_list_f = []
for ij in range(len(cases_frequency)):
    temp_cases_list = []
    temp_phys_list = []
    for ik in range(len(cases_supersample_factor)):
        file_path,file_number = find_highest_numbered_file(output_dir+'mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_output/mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_ep','[0-9]*','_pred.mat')
        temp_cases_list.append('mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_output/mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_ep'+str(file_number)+'_pred.mat')
        temp_phys_list.append('mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_output/mfg_t001_f005_f'+str(cases_frequency[ij])+'_S'+str(cases_supersample_factor[ik])+'_j001_ep'+str(file_number)+'_error.mat')
    cases_list_f.append(temp_cases_list)
    phys_list_f.append(temp_phys_list)





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

# load reference data
for mode_number in [0,1,2,3,4,5,6,7]:
    ScalingParameters.f.append(UserScalingParameters())
    fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')
    phi_xr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xr[cylinder_mask_ref.ravel()] = np.NaN
    phi_xr_ref.append(phi_xr)
    phi_xi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,0]))
    phi_xi[cylinder_mask_ref.ravel()] = np.NaN
    phi_xi_ref.append(phi_xi)
    phi_yr = np.array(np.real(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yr[cylinder_mask_ref.ravel()] = np.NaN
    phi_yr_ref.append(phi_yr)
    phi_yi = np.array(np.imag(fourierModeFile['velocityModes'][:,mode_number,1]))
    phi_yi[cylinder_mask_ref.ravel()] = np.NaN
    phi_yi_ref.append(phi_yi)

    psi_r = np.array(np.real(fourierModeFile['pressureModes'][:,mode_number]))
    psi_r[cylinder_mask_ref.ravel()] = np.NaN
    psi_r_ref.append(psi_r)
    psi_i = np.array(np.imag(fourierModeFile['pressureModes'][:,mode_number]))
    psi_i[cylinder_mask_ref.ravel()] = np.NaN
    psi_i_ref.append(psi_i)


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

mean_err_phi_x = []
mean_err_phi_y = []
mean_err_psi = []

mean_mx = []
mean_my = []
mean_mass = []

max_err_phi_x = []
max_err_phi_y = []
max_err_psi = []

max_mx = []
max_my = []
max_mass = []

for s in range(len(cases_supersample_factor)):
    phi_xr_pred.append([])
    phi_yr_pred.append([])
    psi_r_pred.append([])
    mx_r_pred.append([])
    my_r_pred.append([])
    mass_r_pred.append([])

    mean_err_phi_x.append([])
    mean_err_phi_y.append([])
    mean_err_psi.append([])

    mean_mx.append([])
    mean_my.append([])
    mean_mass.append([])

    max_err_phi_x.append([])
    max_err_phi_y.append([])
    max_err_psi.append([])

    max_mx.append([])
    max_my.append([])
    max_mass.append([])
 

    for c in [0,1,2,3,4,5]:
        pred_f_file = h5py.File(output_dir+cases_list_f[c][s],'r')
        phys_f_file = h5py.File(output_dir+phys_list_f[c][s],'r')
        #print(output_dir+cases_list_f[c][s])
        phi_xr_pred[s].append(np.array(pred_f_file['pred'][:,0],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_xr)
        phi_xi_pred = np.array(pred_f_file['pred'][:,1],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_xi
        phi_yr_pred[s].append(np.array(pred_f_file['pred'][:,2],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_yr)
        phi_yi_pred = np.array(pred_f_file['pred'][:,3],dtype=np.float64)*ScalingParameters.f[c].MAX_phi_yi
        psi_r_pred[s].append(np.array(pred_f_file['pred'][:,10],dtype=np.float64)*ScalingParameters.f[c].MAX_psi)
        psi_i_pred = np.array(pred_f_file['pred'][:,11],dtype=np.float64)*ScalingParameters.f[c].MAX_psi
        mx_r_pred[s].append(np.array(phys_f_file['mxr']))
        my_r_pred[s].append(np.array(phys_f_file['myr']))
        mass_r_pred[s].append(np.array(phys_f_file['massr']))

        temp_MAX_phi_x = np.nanmax(np.abs(phi_xr_ref[c]+1j*phi_xi_ref[c]))
        temp_MAX_phi_y = np.nanmax(np.abs(phi_yr_ref[c]+1j*phi_yi_ref[c]))
        temp_MAX_psi = np.nanmax(np.abs(psi_r_ref[c]+1j*psi_i_ref[c]))

        temp_err_phi_x = np.abs(phi_xr_pred[s][c]-phi_xr_ref[c]+1j*(phi_xi_pred-phi_xi_ref[c]))/temp_MAX_phi_x
        temp_err_phi_y = np.abs(phi_yr_pred[s][c]-phi_yr_ref[c]+1j*(phi_yi_pred-phi_yi_ref[c]))/temp_MAX_phi_y
        temp_err_psi = np.abs(psi_r_pred[s][c]-psi_r_ref[c]+1j*(psi_i_pred-psi_i_ref[c]))/temp_MAX_psi

        temp_mx = np.abs(mx_r_pred[s][c].ravel()+1j*(np.array(phys_f_file['mxi'])).ravel())
        temp_my = np.abs(my_r_pred[s][c].ravel()+1j*(np.array(phys_f_file['myi'])).ravel())
        temp_mass = np.abs(mass_r_pred[s][c].ravel()+1j*(np.array(phys_f_file['massi'])).ravel())

        mean_err_phi_x[s].append(np.nanmean(temp_err_phi_x))
        mean_err_phi_y[s].append(np.nanmean(temp_err_phi_y))
        mean_err_psi[s].append(np.nanmean(temp_err_psi))

        mean_mx[s].append(np.nanmean(temp_mx))
        mean_my[s].append(np.nanmean(temp_my))
        mean_mass[s].append(np.nanmean(temp_mass))

        max_err_phi_x[s].append(np.nanmax(temp_err_phi_x))
        max_err_phi_y[s].append(np.nanmax(temp_err_phi_y))
        max_err_psi[s].append(np.nanmax(temp_err_psi))

        max_mx[s].append(np.nanmax(temp_mx))
        max_my[s].append(np.nanmax(temp_my))
        max_mass[s].append(np.nanmax(temp_mass))

print('Mean Field Max amplitudes')
print('ux: ',np.nanmax(np.abs(ux.ravel())),'uy: ',np.nanmax(np.abs(uy.ravel())),'p: ',np.nanmax(np.abs(p.ravel())))

print('Fourier Mode Max amplitudes')
for c in range(8):
    print('Mode ',c)
    print('phi_x: ',ScalingParameters.f[c].MAX_plot_phi_x,'phi_y: ',ScalingParameters.f[c].MAX_plot_phi_y,'psi: ',ScalingParameters.f[c].MAX_plot_psi)


dx = [] # array for supersample spacing
for s in range(len(cases_supersample_factor)):
    dx.append([])
    for c in [0,1,2,3,4,5]:
        if cases_supersample_factor[s]==0:
            dx[s].append(X_grid[1,0]-X_grid[0,0])

        if cases_supersample_factor[s]>0:
            linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

            x_downsample = x[linear_downsample_inds]
            y_downsample = y[linear_downsample_inds]

            x_ds_grid = (np.reshape(x_downsample,(ndy,ndx))).transpose()
            y_ds_grid = (np.reshape(y_downsample,(ndy,ndx))).transpose()
            dx[s].append(x_ds_grid[1,0]-x_ds_grid[0,0])


cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))



text_color_threshold = 1E-1

if True:

    for s in range(len(cases_supersample_factor)):
        for c in [0,1,2,3,4,5]:
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

            phi_xr_err_grid = phi_xr_grid - phi_xr_pred_grid
            phi_yr_err_grid = phi_yr_grid - phi_yr_pred_grid
            psi_r_err_grid = psi_r_grid - psi_r_pred_grid

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

            MAX_phi_xr_err = np.nanmax(np.abs(phi_xr_err_grid.ravel()))
            MAX_phi_yr_err = np.nanmax(np.abs(phi_yr_err_grid.ravel()))
            MAX_psi_r_err = np.nanmax(np.abs(psi_r_err_grid.ravel()))

            MAX_mx = np.nanmax(np.abs(mx_grid.ravel()))
            MAX_my = np.nanmax(np.abs(my_grid.ravel()))
            MAX_mass = np.nanmax(np.abs(mass_grid.ravel()))

            levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
            levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
            levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)

            ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

            levels_mx = np.geomspace(1E-3,1,21)

            # dual log version

            from matplotlib.colors import LinearSegmentedColormap

            cdict3 = {'red':   [(0.0,  1.0, 1.0), # white
                                (0.33,  1.0, 1.0), # orange
                                (0.66,  1.0, 1.0), # pink
                                (1.0,  1.0, 1.0)], # red 

                    'green': [(0.0,  0.0, 1.0),
                                (0.33, 170/255.0, 170/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                                (0.66, 0.0, 0.0),
                                (1.0,  0.0, 0.0)],

                    'blue':  [(0.0,  0.0, 1.0),
                                (0.33,  0.0, 0.0),
                                (0.66,  1.0, 1.0),
                                (1.0,  0.0, 0.0)]}


            cmap1 = LinearSegmentedColormap('WhiteYellowPinkRed',cdict3)
            cmap1.set_bad('white',alpha=0.0)

            
            cdict4 = {'red':   [(0.0,  1.0, 1.0),
                                (0.33,  0.0, 0.0),
                                (0.66,  0.0, 0.0),
                                (1.0,  0.0, 0.0)],

                    'green': [(0.0,  0.0, 1.0),
                                (0.33, 1.0, 1.0),
                                (0.66,  1.0, 1.0),
                                (1.0,  0.0, 0.0)],

                    'blue':  [(0.0,  0.0, 1.0),
                                (0.33,  0.0, 0.0),
                                (0.66,  1.0, 1.0),
                                (1.0,  1.0, 1.0)]}

            cmap2 = LinearSegmentedColormap('WhiteGreenCyanBlue',cdict4)
            cmap2.set_bad('white',alpha=0.0)
            
            cdict5 = {'red':   [(0.0,  0.0, 0.0),
                                (0.16,  0.0, 0.0),
                                (0.33,  0.0, 0.0),
                                (0.5,  1.0, 1.0), # white
                                (0.66,  1.0, 1.0), # orange
                                (0.83,  1.0, 1.0), # pink
                                (1.0,  1.0, 1.0)], # red 

                    'green': [(0.0,  0.0, 0.0),
                                (0.16,  1.0, 1.0),
                                (0.33,  1.0, 1.0),
                                (0.5,  1.0, 1.0),
                                (0.66, 170.0/255.0, 170.0/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                                (0.83, 0.0, 0.0),
                                (1.0,  0.0, 0.0)],

                    'blue':  [(0.0,  1.0, 1.0),
                                (0.16,  1.0, 1.0),
                                (0.33,  0.0, 0.0),
                                (0.5,  1.0, 1.0),
                                (0.66,  0.0, 0.0),
                                (0.83,  1.0, 1.0),
                                (1.0,  0.0, 0.0)]}
            # for the colobars only
            cmap3 = LinearSegmentedColormap('BlueCyanGreenWhiteYellowPinkRed',cdict5)
            cmap3.set_bad('white',alpha=0.0)

            dual_log_cbar_norm = matplotlib.colors.CenteredNorm(0.0,1.0)
            dual_log_cbar_ticks = [-1,-0.666,-0.333,0,0.333,0.666,1]
            dual_log_cbar_labels = ['-1','-1e-1','-1e-2','0','1e-2','1e-1','1']

            levels_mx = np.geomspace(1E-3,1,11)

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
            ax.text(7,1.5,'$\Phi_{x,DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(aa)',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
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
            ax.text(7,1.5,'$\Phi_{x,PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(ab)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            e_plot = phi_xr_err_grid/MAX_plot_phi_xr
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\Phi_{x,DNS}-\Phi_{x,PINN}}{max(\Phi_{x,DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)



            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,phi_yr_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\Phi_{y,DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ba)',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
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
            ax.text(7,1.5,'$\Phi_{y,PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(bb)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            e_plot = phi_yr_err_grid/MAX_plot_phi_yr
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            if np.mean(e_plot.ravel())>text_color_threshold:
                t=ax.text(7,1.5,'$|\\frac{\Phi_{y,DNS}-\Phi_{y,PINN}}{max(\Phi_{y,DNS})}|$',fontsize=5,color='w')
                ax.text(-1.75,1.5,'(bc)',fontsize=5,color='w')
            else:
                t=ax.text(7,1.5,'$|\\frac{\Phi_{y,DNS}-\Phi_{y,PINN}}{max(\Phi_{y,DNS})}|$',fontsize=5,color='k')
                ax.text(-1.75,1.5,'(bc)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\Psi_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ca)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
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
            ax.text(7,1.5,'$\Psi_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(cb)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            e_plot = psi_r_err_grid/MAX_plot_psi_r
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\Psi_{DNS}-\Psi_{PINN}}{max(\Psi_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            e_plot = mx_grid
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.text(7,1.5,'$FANS_{x}$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(da)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][1])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[3][3])
            e_plot = my_grid
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.text(7,1.5,'$FANS_{y}$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(db)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][4])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[3][6])
            e_plot = mass_grid
            e_plot_p =e_plot+1E-30
            e_plot_p[e_plot_p<=0]=np.NaN
            e_plot_n = e_plot
            e_plot_n[e_plot_n>0]=np.NaN
            e_plot_n = np.abs(e_plot_n)

            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
            ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
            ax.set_aspect('equal')
            ax.set_xlabel('x/D',fontsize=5)
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.text(7,1.5,'$FAC$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(dc)',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

            plot.savefig(figures_dir+'logerr_mfg_t001_f005_f'+str(c)+'_contours_S'+str(cases_supersample_factor[s])+'.pdf')
            plot.savefig(figures_dir+'logerr_mfg_t001_f005_f'+str(c)+'_contours_S'+str(cases_supersample_factor[s])+'.png',dpi=300)
            plot.close(fig)

if True:
    # grid the data
    c=0
    phi_xr_grid = np.reshape(phi_xr_ref[c],X_grid.shape)
    phi_yr_grid = np.reshape(phi_yr_ref[c],X_grid.shape)
    psi_r_grid = np.reshape(psi_r_ref[c],X_grid.shape)

    phi_xr_grid[cylinder_mask]=np.NaN
    phi_yr_grid[cylinder_mask]=np.NaN
    psi_r_grid[cylinder_mask]=np.NaN

    s=4
    phi_xr_pred_grid1 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid1 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid1 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid1[cylinder_mask]=np.NaN
    phi_yr_pred_grid1[cylinder_mask]=np.NaN
    psi_r_pred_grid1[cylinder_mask]=np.NaN

    phi_xr_err_grid1 = phi_xr_pred_grid1 - phi_xr_grid
    phi_yr_err_grid1 = phi_yr_pred_grid1 - phi_yr_grid
    psi_r_err_grid1 = psi_r_pred_grid1 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    # grid the data
    s=5
    phi_xr_pred_grid2 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid2 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid2 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid2[cylinder_mask]=np.NaN
    phi_yr_pred_grid2[cylinder_mask]=np.NaN
    psi_r_pred_grid2[cylinder_mask]=np.NaN

    phi_xr_err_grid2 = phi_xr_pred_grid2 - phi_xr_grid
    phi_yr_err_grid2 = phi_yr_pred_grid2 - phi_yr_grid
    psi_r_err_grid2 = psi_r_pred_grid2 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>6.5,y_downsample1>1))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    text_corner_mask = (np.multiply(x_downsample2>6.5,y_downsample2>1))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]

    # compute levels
    MAX_plot_phi_xr = ScalingParameters.f[c].MAX_plot_phi_x
    MAX_plot_phi_yr = ScalingParameters.f[c].MAX_plot_phi_y
    MAX_plot_psi_r =  ScalingParameters.f[c].MAX_plot_psi
    
    levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
    levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
    levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)

    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

    levels_mx = np.geomspace(1E-3,1,21)

    # dual log version

    from matplotlib.colors import LinearSegmentedColormap

    cdict3 = {'red':   [(0.0,  1.0, 1.0), # white
                        (0.33,  1.0, 1.0), # orange
                        (0.66,  1.0, 1.0), # pink
                        (1.0,  1.0, 1.0)], # red 

             'green': [(0.0,  0.0, 1.0),
                        (0.33, 170/255.0, 170/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                        (0.66, 0.0, 0.0),
                        (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  0.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.66,  1.0, 1.0),
                        (1.0,  0.0, 0.0)]}


    cmap1 = LinearSegmentedColormap('WhiteYellowPinkRed',cdict3)
    cmap1.set_bad('white',alpha=0.0)

    
    cdict4 = {'red':   [(0.0,  1.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.66,  0.0, 0.0),
                        (1.0,  0.0, 0.0)],

             'green': [(0.0,  0.0, 1.0),
                        (0.33, 1.0, 1.0),
                        (0.66,  1.0, 1.0),
                        (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  0.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.66,  1.0, 1.0),
                        (1.0,  1.0, 1.0)]}

    cmap2 = LinearSegmentedColormap('WhiteGreenCyanBlue',cdict4)
    cmap2.set_bad('white',alpha=0.0)
    
    cdict5 = {'red':   [(0.0,  0.0, 0.0),
                        (0.16,  0.0, 0.0),
                        (0.33,  0.0, 0.0),
                        (0.5,  1.0, 1.0), # white
                        (0.66,  1.0, 1.0), # orange
                        (0.83,  1.0, 1.0), # pink
                        (1.0,  1.0, 1.0)], # red 

             'green': [(0.0,  0.0, 0.0),
                        (0.16,  1.0, 1.0),
                        (0.33,  1.0, 1.0),
                        (0.5,  1.0, 1.0),
                        (0.66, 170.0/255.0, 170.0/255.0), # (0.33, 170.0/255.0, 170.0/255.0),
                        (0.83, 0.0, 0.0),
                        (1.0,  0.0, 0.0)],

            'blue':  [(0.0,  1.0, 1.0),
                        (0.16,  1.0, 1.0),
                        (0.33,  0.0, 0.0),
                        (0.5,  1.0, 1.0),
                        (0.66,  0.0, 0.0),
                        (0.83,  1.0, 1.0),
                        (1.0,  0.0, 0.0)]}
    # for the colobars only
    cmap3 = LinearSegmentedColormap('BlueCyanGreenWhiteYellowPinkRed',cdict5)
    cmap3.set_bad('white',alpha=0.0)

    dual_log_cbar_norm = matplotlib.colors.CenteredNorm(0.0,1.0)
    dual_log_cbar_ticks = [-1,-0.666,-0.333,0,0.333,0.666,1]
    dual_log_cbar_labels = ['-1','-1e-1','-1e-2','0','1e-2','1e-1','1']

    levels_mx = np.geomspace(1E-3,1,11)

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(3.37,6))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.05)
    outer = gridspec.GridSpec(6,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\Phi_{1x,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = phi_xr_err_grid1/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{1x,PINN}-\Phi_{1x,DNS}}{max(|\Phi_{1x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    # check bar
    #cax=plot.Subplot(fig,inner[1][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])

    e_plot = phi_xr_err_grid2/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{1x,PINN}-\Phi_{1x,DNS}}{max(|\Phi_{1x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\psi_{1,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = psi_r_err_grid1/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)


    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(7,1.3,'$\\frac{\psi_{1,PINN}-\psi_{1,DNS}}{max(|\psi_{1,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])

    e_plot = psi_r_err_grid2/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(7,1.3,'$\\frac{\psi_{1,PINN}-\psi_{1,DNS}}{max(|\psi_{1,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f0_contours_condensed.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f0_contours_condensed.png',dpi=300)
    plot.close(fig)














    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(3.37,8))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.04)
    outer = gridspec.GridSpec(9,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\Phi_{1x,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = phi_xr_err_grid1/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{1x,PINN}-\Phi_{1x,DNS}}{max(|\Phi_{1x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    # check bar
    #cax=plot.Subplot(fig,inner[1][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])

    e_plot = phi_xr_err_grid2/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{1x,PINN}-\Phi_{1x,DNS}}{max(|\Phi_{1x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_yr_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\Phi_{1y,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = phi_yr_err_grid1/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{1y,PINN}-\Phi_{1y,DNS}}{max(|\Phi_{1y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    # check bar
    #cax=plot.Subplot(fig,inner[1][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])

    e_plot = phi_yr_err_grid2/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{1y,PINN}-\Phi_{1y,DNS}}{max(|\Phi_{1y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)



    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[6],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[6][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\psi_{1,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])

    e_plot = psi_r_err_grid1/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)


    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.75,1.3,'$\\frac{\psi_{1,PINN}-\psi_{1,DNS}}{max(|\psi_{1,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(h)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[7][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[8],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])

    e_plot = psi_r_err_grid2/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.75,1.3,'$\\frac{\psi_{1,PINN}-\psi_{1,DNS}}{max(|\psi_{1,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[8][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f0_contours_condensed_all.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f0_contours_condensed_all.png',dpi=300)
    plot.close(fig)








    # grid the data
    c=2
    phi_xr_grid = np.reshape(phi_xr_ref[c],X_grid.shape)
    phi_yr_grid = np.reshape(phi_yr_ref[c],X_grid.shape)
    psi_r_grid = np.reshape(psi_r_ref[c],X_grid.shape)

    phi_xr_grid[cylinder_mask]=np.NaN
    phi_yr_grid[cylinder_mask]=np.NaN
    psi_r_grid[cylinder_mask]=np.NaN

    s=4
    phi_xr_pred_grid1 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid1 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid1 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid1[cylinder_mask]=np.NaN
    phi_yr_pred_grid1[cylinder_mask]=np.NaN
    psi_r_pred_grid1[cylinder_mask]=np.NaN

    phi_xr_err_grid1 = phi_xr_pred_grid1 - phi_xr_grid
    phi_yr_err_grid1 = phi_yr_pred_grid1 - phi_yr_grid
    psi_r_err_grid1 = psi_r_pred_grid1 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    # grid the data
    s=5
    phi_xr_pred_grid2 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid2 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid2 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid2[cylinder_mask]=np.NaN
    phi_yr_pred_grid2[cylinder_mask]=np.NaN
    psi_r_pred_grid2[cylinder_mask]=np.NaN

    phi_xr_err_grid2 = phi_xr_pred_grid2 - phi_xr_grid
    phi_yr_err_grid2 = phi_yr_pred_grid2 - phi_yr_grid
    psi_r_err_grid2 = psi_r_pred_grid2 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>6.5,y_downsample1>1))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    text_corner_mask = (np.multiply(x_downsample2>6.5,y_downsample2>1))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]

    # compute levels
    MAX_plot_phi_xr = ScalingParameters.f[2].MAX_plot_phi_x
    MAX_plot_phi_yr = ScalingParameters.f[2].MAX_plot_phi_y
    MAX_plot_psi_r = ScalingParameters.f[2].MAX_plot_psi
    
    levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
    levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
    levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)

    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

    levels_mx = np.geomspace(1E-3,1,21)

    # mode 3 summary dual log version
    levels_mx = np.geomspace(1E-3,1,11)

        
    fig = plot.figure(figsize=(3.37,6.75))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.93,bottom=0.05)
    outer = gridspec.GridSpec(7,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\Phi_{3x,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = phi_xr_err_grid1/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{3x,PINN}-\Phi_{3x,DNS}}{max(|\Phi_{3x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[2][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_pred_grid2,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(8,1.4,'$\Phi_{3x,PINN}$',fontsize=8)
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[2][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[3][0])

    e_plot = phi_xr_err_grid2/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{3x,PINN}-\Phi_{3x,DNS}}{max(|\Phi_{3x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[4][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_yr_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\Phi_{3y,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[4][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[5][0])

    e_plot = phi_yr_err_grid1/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{3y,PINN}-\Phi_{3y,DNS}}{max(|\Phi_{3y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[6],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[6][0])

    e_plot = phi_yr_err_grid2/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{\Phi_{3y,PINN}-\Phi_{3y,DNS}}{max(|\Phi_{3y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.set_xlabel('x/D',fontsize=8,labelpad=0)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[6][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
    
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f2_contours_uxuy_condensed_duallog.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f2_contours_uxuy_condensed_duallog.png',dpi=300)
    plot.close(fig)

    # mode 0 summary
    fig = plot.figure(figsize=(3.37,5))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.93,bottom=0.06)
    outer = gridspec.GridSpec(5,1,wspace=0.1,hspace=0.1)
    inner = []

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\psi_{3,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[1][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_pred_grid1,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\psi_{3,PINN}$',fontsize=8)
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[2][0])

    e_plot = psi_r_err_grid1/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(7,1.3,'$\\frac{\psi_{3,PINN}-\psi_{3,DNS}}{max(|\psi_{3,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_pred_grid2,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\psi_{3,PINN}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = psi_r_err_grid2/MAX_plot_psi_r
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(7,1.3,'$\\frac{\psi_{3,PINN}-\psi_{3,DNS}}{max(|\psi_{3,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f2_contours_p_condensed_duallog.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f2_contours_p_condensed_duallog.png',dpi=300)
    plot.close(fig)

    











    # grid the data
    c=5
    phi_xr_grid = np.reshape(phi_xr_ref[c],X_grid.shape)
    phi_yr_grid = np.reshape(phi_yr_ref[c],X_grid.shape)
    psi_r_grid = np.reshape(psi_r_ref[c],X_grid.shape)

    phi_xr_grid[cylinder_mask]=np.NaN
    phi_yr_grid[cylinder_mask]=np.NaN
    psi_r_grid[cylinder_mask]=np.NaN

    s=2
    phi_xr_pred_grid1 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid1 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid1 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid1[cylinder_mask]=np.NaN
    phi_yr_pred_grid1[cylinder_mask]=np.NaN
    psi_r_pred_grid1[cylinder_mask]=np.NaN

    phi_xr_err_grid1 = phi_xr_pred_grid1 - phi_xr_grid
    phi_yr_err_grid1 = phi_yr_pred_grid1 - phi_yr_grid
    psi_r_err_grid1 = psi_r_pred_grid1 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    # grid the data
    s=3
    phi_xr_pred_grid2 = np.reshape(phi_xr_pred[s][c],X_grid.shape)
    phi_yr_pred_grid2 = np.reshape(phi_yr_pred[s][c],X_grid.shape)
    psi_r_pred_grid2 = np.reshape(psi_r_pred[s][c],X_grid.shape)

    phi_xr_pred_grid2[cylinder_mask]=np.NaN
    phi_yr_pred_grid2[cylinder_mask]=np.NaN
    psi_r_pred_grid2[cylinder_mask]=np.NaN

    phi_xr_err_grid2 = phi_xr_pred_grid2 - phi_xr_grid
    phi_yr_err_grid2 = phi_yr_pred_grid2 - phi_yr_grid
    psi_r_err_grid2 = psi_r_pred_grid2 - psi_r_grid

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>6.5,y_downsample1>1))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    text_corner_mask = (np.multiply(x_downsample2>6.5,y_downsample2>1))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]

    # compute levels
    MAX_plot_phi_xr = ScalingParameters.f[c].MAX_plot_phi_x
    MAX_plot_phi_yr = ScalingParameters.f[c].MAX_plot_phi_y
    MAX_plot_psi_r =  ScalingParameters.f[c].MAX_plot_psi
    
    levels_phi_xr = np.linspace(-MAX_plot_phi_xr,MAX_plot_phi_xr,21)
    levels_phi_yr = np.linspace(-MAX_plot_phi_yr,MAX_plot_phi_yr,21)
    levels_psi_r = np.linspace(-MAX_plot_psi_r,MAX_plot_psi_r,21)

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(3.37,8))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.86,bottom=0.04)
    outer = gridspec.GridSpec(9,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_xr_grid,levels=levels_phi_xr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\Phi_{6x,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_xr,MAX_plot_phi_xr/2,0.0,-MAX_plot_phi_xr/2,-MAX_plot_phi_xr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = phi_xr_err_grid1/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{6x,PINN}-\Phi_{6x,DNS}}{max(|\Phi_{6x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 10$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    # check bar
    #cax=plot.Subplot(fig,inner[1][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])

    e_plot = phi_xr_err_grid2/MAX_plot_phi_xr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{6x,PINN}-\Phi_{6x,DNS}}{max(|\Phi_{6x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,phi_yr_grid,levels=levels_phi_yr,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\Phi_{6y,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_phi_yr,MAX_plot_phi_yr/2,0.0,-MAX_plot_phi_yr/2,-MAX_plot_phi_yr],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = phi_yr_err_grid1/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{6y,PINN}-\Phi_{6y,DNS}}{max(|\Phi_{6y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 10$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    # check bar
    #cax=plot.Subplot(fig,inner[1][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])

    e_plot = phi_yr_err_grid2/MAX_plot_phi_yr
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.25,1.3,'$\\frac{\Phi_{6y,PINN}-\Phi_{6y,DNS}}{max(|\Phi_{6y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)



    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[6],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[6][0])
    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_grid,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\psi_{6,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])

    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_pred_grid1,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample1,y_downsample1,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.75,1.3,'$\psi_{6,PINN}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(h)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 10$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[7][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[8],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])

    ux_plot = ax.contourf(X_grid,Y_grid,psi_r_pred_grid2,levels=levels_psi_r,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.75,1.3,'$\psi_{6,PINN}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[8][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_psi_r,MAX_plot_psi_r/2,0.0,-MAX_plot_psi_r/2,-MAX_plot_psi_r],format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)


    
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f5_contours_condensed_all.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f5_contours_condensed_all.png',dpi=300)
    plot.close(fig)




# error percent plot
pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,2]



mean_err_phi_x = np.array(mean_err_phi_x)
mean_err_phi_y = np.array(mean_err_phi_y)
mean_err_psi = np.array(mean_err_psi)
mean_mx = np.array(mean_mx)
mean_my = np.array(mean_my)
mean_mass = np.array(mean_mass)

max_err_phi_x = np.array(max_err_phi_x)
max_err_phi_y = np.array(max_err_phi_y)
max_err_psi = np.array(max_err_psi)
max_mx = np.array(max_mx)
max_my = np.array(max_my)
max_mass = np.array(max_mass)

# compute the combined quantities for the additional plot
for c in [0,1,2,3,4,5]:
    print('Mode ',c)
    print('Phi_x')
    print('Mean: ',mean_err_phi_x[:,c])
    print('Max: ',max_err_phi_x[:,c])
    print('Phi_y')
    print('Mean: ',mean_err_phi_y[:,c])
    print('Max: ',max_err_phi_y[:,c])
    print('Psi')
    print('Mean: ',mean_err_psi[:,c])
    print('Max: ',max_err_psi[:,c])
    print('Mx')
    print('Mean: ',mean_mx[:,c])
    print('Max: ',max_mx[:,c])
    print('My')
    print('Mean: ',mean_my[:,c])
    print('Max: ',max_my[:,c])
    print('Mass')
    print('Mean: ',mean_mass[:,c])
    print('Max: ',max_mass[:,c])

error_x_tick_labels = ['40','20','10','5','2.5','1.25']
error_y_ticks = [1E-4,1E-3,1E-2,1E-1,1,10]
error_y_tick_labels = ['1E-4','1E-3','0.01','0.1','1','10']

for c in [0,1,2,3,4,5]:
    # reduced error plot
    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3.0)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.15)

    mean_plt,=axs.plot(pts_per_d*0.9,mean_err_phi_x[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt2,=axs.plot(pts_per_d*1.0,mean_err_phi_y[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    mean_plt3,=axs.plot(pts_per_d*1.1,mean_err_psi[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt,=axs.plot(pts_per_d*0.9,max_err_phi_x[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    max_plt2,=axs.plot(pts_per_d*1.0,max_err_phi_y[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    max_plt3,=axs.plot(pts_per_d*1.1,max_err_psi[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xticks(pts_per_d)
    axs.set_xticklabels(error_x_tick_labels,fontsize=8)
    axs.set_ylim(5E-5,5E1)
    axs.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    axs.legend([mean_plt,mean_plt2,mean_plt3,max_plt,max_plt2,max_plt3],['Mean $\Phi_{'+str(c+1)+'x}$','Mean $\Phi_{'+str(c+1)+'y}$','Mean $\psi_{'+str(c+1)+'}$','Max $\Phi_{'+str(c+1)+'x}$','Max $\Phi_{'+str(c+1)+'y}$','Max $\psi_{'+str(c+1)+'}$'],fontsize=8,ncol=2)
    axs.grid('on')
    axs.set_xlabel('$D/\Delta x$',fontsize=8)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f'+str(c)+'_error_condensed.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_f'+str(c)+'_error_condensed.png',dpi=300)
    plot.close(fig)



subplot_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
if True:
    fig = plot.figure(figsize=(3.37,8))
    plot.subplots_adjust(left=0.2,top=0.99,right=0.97,bottom=0.05)
    outer = gridspec.GridSpec(6,1,wspace=0.1,hspace=0.1,height_ratios=[0.25,0.15,0.15,0.15,0.15,0.15])
    inner = []

    for c in [0]:
        inner.append(plot.Subplot(fig,outer[c]))

        mean_plt,=inner[c].plot(pts_per_d*0.9,mean_err_phi_x[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
        mean_plt2,=inner[c].plot(pts_per_d*1.0,mean_err_phi_y[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
        mean_plt3,=inner[c].plot(pts_per_d*1.1,mean_err_psi[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
        max_plt,=inner[c].plot(pts_per_d*0.9,max_err_phi_x[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
        max_plt2,=inner[c].plot(pts_per_d*1.0,max_err_phi_y[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
        max_plt3,=inner[c].plot(pts_per_d*1.1,max_err_psi[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

        inner[c].set_xscale('log')
        inner[c].set_yscale('log')
        inner[c].set_xticks(pts_per_d)
        inner[c].set_xticklabels(error_x_tick_labels,fontsize=8)
        inner[c].xaxis.set_tick_params(labelbottom=False)
        inner[c].set_ylim(1E-4,1E2)
        inner[c].set_xlim(9E-1,55)
        inner[c].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
        inner[c].set_ylabel("Relative Error",fontsize=8)
        inner[c].legend([mean_plt,mean_plt2,mean_plt3,max_plt,max_plt2,max_plt3],['Mean $\Phi_{x}$','Mean $\Phi_{y}$','Mean $\psi$','Max $\Phi_{x}$','Max $\Phi_{y}$','Max $\psi$'],fontsize=8,ncol=2)
        inner[c].text(10,0.2,'Mode 1',fontsize=8)
        inner[c].text(1,20,subplot_labels[c],fontsize=8)
        inner[c].grid('on')

        fig.add_subplot(inner[c])

    for c in [1,2,3,4]:
        inner.append(plot.Subplot(fig,outer[c]))

        mean_plt,=inner[c].plot(pts_per_d*0.9,mean_err_phi_x[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
        mean_plt2,=inner[c].plot(pts_per_d*1.0,mean_err_phi_y[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
        mean_plt3,=inner[c].plot(pts_per_d*1.1,mean_err_psi[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
        max_plt,=inner[c].plot(pts_per_d*0.9,max_err_phi_x[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
        max_plt2,=inner[c].plot(pts_per_d*1.0,max_err_phi_y[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
        max_plt3,=inner[c].plot(pts_per_d*1.1,max_err_psi[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

        inner[c].set_xscale('log')
        inner[c].set_yscale('log')
        inner[c].set_xticks(pts_per_d)
        inner[c].set_xticklabels(error_x_tick_labels,fontsize=8)
        inner[c].xaxis.set_tick_params(labelbottom=False)
        inner[c].set_ylim(1E-4,1E1)
        inner[c].set_xlim(9E-1,55)
        inner[c].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
        inner[c].set_ylabel("Relative Error",fontsize=8)
        inner[c].text(10,2,'Mode '+str(c+1),fontsize=8)
        inner[c].text(1,2,subplot_labels[c],fontsize=8)
        inner[c].grid('on')

        fig.add_subplot(inner[c])

    for c in [5]:
        inner.append(plot.Subplot(fig,outer[c]))

        mean_plt,=inner[c].plot(pts_per_d*0.9,mean_err_phi_x[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
        mean_plt2,=inner[c].plot(pts_per_d*1.0,mean_err_phi_y[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
        mean_plt3,=inner[c].plot(pts_per_d*1.1,mean_err_psi[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
        max_plt,=inner[c].plot(pts_per_d*0.9,max_err_phi_x[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
        max_plt2,=inner[c].plot(pts_per_d*1.0,max_err_phi_y[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
        max_plt3,=inner[c].plot(pts_per_d*1.1,max_err_psi[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

        inner[c].set_xscale('log')
        inner[c].set_yscale('log')
        inner[c].set_xticks(pts_per_d)
        inner[c].set_xticklabels(error_x_tick_labels,fontsize=8)
        inner[c].set_ylim(1E-4,1E1)
        inner[c].set_xlim(9E-1,55)
        inner[c].set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
        inner[c].set_ylabel("Relative Error",fontsize=8)
        inner[c].text(10,2,'Mode '+str(c+1),fontsize=8)
        inner[c].text(1,2,subplot_labels[c],fontsize=8)
        inner[c].grid('on')
        inner[c].set_xlabel('$D/\Delta x$',fontsize=8)

        fig.add_subplot(inner[c])


    plot.savefig(figures_dir+'logerr_mfg_t001_f005_error_allc.pdf')
    plot.savefig(figures_dir+'logerr_mfg_t001_f005_error_allc.png',dpi=300)
    plot.close(fig)