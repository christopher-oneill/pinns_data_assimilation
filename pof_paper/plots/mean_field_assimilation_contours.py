

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_galerkin_viv.lib.downsample import compute_downsample_inds

from pinns_galerkin_viv.lib.file_util import extract_matching_integers
from pinns_galerkin_viv.lib.file_util import find_highest_numbered_file
from pinns_galerkin_viv.lib.file_util import create_directory_if_not_exists

# script

figures_dir = 'C:/projects/paper_figures/mean_field/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

cases_list = ['mfg_vdnn_mean002_S1_L10N100_output/mfg_vdnn_mean002_S1_L10N100_ep12000_pred.mat','mfg_vdnn_mean008_S16_L10N100_output/mfg_vdnn_mean008_S16_L10N100_ep171000_pred.mat','mfg_vdnn_mean008_S32_L10N100_output/mfg_vdnn_mean008_S32_L10N100_ep186000_pred.mat']
cases_supersample_factor = [1,16,32]



# load the reference data
meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
configFile = h5py.File(data_dir+'configuration.mat','r')
meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
                
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]

MAX_ux = np.max(np.abs(ux))
MAX_uy = np.max(np.abs(uy))
MAX_p= 1 # estimated maximum pressure

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])[0]
MAX_x = max(x.flatten())
MAX_y = max(y.flatten())

cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))
ux_grid = np.reshape(ux,X_grid.shape)
ux_grid[cylinder_mask] = np.NaN
uy_grid = np.reshape(uy,X_grid.shape)
uy_grid[cylinder_mask] = np.NaN
p_grid = np.reshape(p,X_grid.shape)
p_grid[cylinder_mask] = np.NaN

n_cases = len(cases_list)
gridspec = {'width_ratios': [1, 1, 1, 0.15]}
fig,axs = plot.subplots(6,4,gridspec_kw=gridspec)
fig.set_size_inches(6.69,5)
fig.tight_layout()


# contour plot levels
levels_ux = np.linspace(-1.2*MAX_ux,1.2*MAX_ux,21)
levels_uy = np.linspace(-1.2*MAX_uy,1.2*MAX_uy,21)
levels_p = np.linspace(-MAX_ux,MAX_p,21)

MAX_err_ux = 0.0
MAX_err_uy = 0.0
MAX_err_p = 0.0
MIN_err_ux = np.NaN
MIN_err_uy = np.NaN
MIN_err_p = np.NaN

for c in range(n_cases):
    predFile = h5py.File(output_dir+cases_list[c],'r')

    ux_pred = np.array(predFile['pred'][:,0])*MAX_ux
    uy_pred = np.array(predFile['pred'][:,1])*MAX_uy
    p_pred = np.array(predFile['pred'][:,5])*MAX_p 

    MAX_err_ux = np.nanmax([np.nanmax(np.abs((ux-ux_pred)/MAX_ux)),MAX_err_ux])
    MAX_err_uy = np.nanmax([np.nanmax(np.abs((uy-uy_pred)/MAX_uy)),MAX_err_uy])
    MAX_err_p = np.nanmax([np.nanmax(np.abs((p-p_pred)/MAX_p)),MAX_err_p])
    MIN_err_ux = np.nanmin([np.nanmin(np.abs((ux-ux_pred)/MAX_ux)),MIN_err_ux])
    MIN_err_uy = np.nanmin([np.nanmin(np.abs((uy-uy_pred)/MAX_uy)),MIN_err_uy])
    MIN_err_p = np.nanmin([np.nanmin(np.abs((p-p_pred)/MAX_p)),MIN_err_p])

levels_err_ux = np.power(10,np.linspace(-6,0,7))
levels_err_uy = np.power(10,np.linspace(-6,0,7))
levels_err_p = np.power(10,np.linspace(-6,0,7))

for c in range(n_cases):
    predFile = h5py.File(output_dir+cases_list[c],'r')

    ux_pred = np.array(predFile['pred'][:,0])*MAX_ux
    uy_pred = np.array(predFile['pred'][:,1])*MAX_uy
    p_pred = np.array(predFile['pred'][:,5])*MAX_p 

    ux_pred_grid = np.reshape(ux_pred,X_grid.shape)
    ux_pred_grid[cylinder_mask] = np.NaN
    uy_pred_grid = np.reshape(uy_pred,X_grid.shape)
    uy_pred_grid[cylinder_mask] = np.NaN
    p_pred_grid = np.reshape(p_pred,X_grid.shape)
    p_pred_grid[cylinder_mask] = np.NaN

    ux_err = np.abs((ux_grid-ux_pred_grid)/MAX_ux)
    uy_err = np.abs((uy_grid-uy_pred_grid)/MAX_uy)
    p_err = np.abs((p_grid-p_pred_grid)/MAX_p)
  
    # compute points if needed
    if cases_supersample_factor[c]>1:
        n_x = X_grid.shape[0]
        n_y = X_grid.shape[1]
        linear_downsample_inds, n_d_x, n_d_y = compute_downsample_inds(cases_supersample_factor[c],n_x,n_y)

        x_downsample = x[linear_downsample_inds]
        y_downsample = y[linear_downsample_inds]
        valid_inds = np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d
        x_downsample = x_downsample[valid_inds]
        y_downsample = y_downsample[valid_inds]

    plot.subplots_adjust(wspace=0.1, hspace=0.0)
    ux_plot = axs[0,c].contourf(X_grid,Y_grid,ux_pred_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'])
    axs[0,c].set_aspect('equal')

    axs[0,c].set_title('S*='+str(cases_supersample_factor[c]),fontsize=6)

    if cases_supersample_factor[c]>1:
        dots = axs[0,c].plot(x_downsample,y_downsample,markersize=1,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')

    ux_err_plot =axs[1,c].contourf(X_grid,Y_grid,ux_err,levels=levels_err_ux,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(1E-6,1))
    axs[1,c].set_aspect('equal')


    uy_plot =axs[2,c].contourf(X_grid,Y_grid,uy_pred_grid,levels=levels_uy,cmap= matplotlib.colormaps['bwr'])
    axs[2,c].set_aspect('equal')
    if cases_supersample_factor[c]>1:
        dots = axs[2,c].plot(x_downsample,y_downsample,markersize=1,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')

    uy_err_plot =axs[3,c].contourf(X_grid,Y_grid,uy_err,levels=levels_err_uy,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(1E-6,1))
    axs[3,c].set_aspect('equal')


    p_plot =axs[4,c].contourf(X_grid,Y_grid,p_pred_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'])
    axs[4,c].set_aspect('equal')

    p_err_plot =axs[5,c].contourf(X_grid,Y_grid,p_err,levels=levels_err_p,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(1E-6,1))
    axs[5,c].set_aspect('equal')


    if c==0:
        axs[0,c].set_ylabel('y/D',fontsize=4)
        axs[1,c].set_ylabel('y/D',fontsize=4)
        axs[2,c].set_ylabel('y/D',fontsize=4)
        axs[3,c].set_ylabel('y/D',fontsize=4)
        axs[4,c].set_ylabel('y/D',fontsize=4)
        axs[5,c].set_ylabel('y/D',fontsize=4)

    axs[5,c].set_xlabel('x/D',fontsize=4)

    axs[0,c].xaxis.set_tick_params(labelbottom=False)
    axs[1,c].xaxis.set_tick_params(labelbottom=False)
    axs[2,c].xaxis.set_tick_params(labelbottom=False)
    axs[3,c].xaxis.set_tick_params(labelbottom=False)
    axs[4,c].xaxis.set_tick_params(labelbottom=False)


    if c>0:
        axs[0,c].yaxis.set_tick_params(labelleft=False)
        axs[1,c].yaxis.set_tick_params(labelleft=False)
        axs[2,c].yaxis.set_tick_params(labelleft=False)
        axs[3,c].yaxis.set_tick_params(labelleft=False)
        axs[4,c].yaxis.set_tick_params(labelleft=False)
        axs[5,c].yaxis.set_tick_params(labelleft=False)
    
    color_bar_fraction = 0.000016
    colorbar_shrink=0.001
    color_bar_pad = 0.00


    
    colorbar_size_vec = [1.05,0,0.02,1.0]
    if c==n_cases-1:
        cbar1 = plot.colorbar(ux_plot,ticks=[1.4,0.7,0.0,-0.7,-1.4],aspect=40,cax=axs[0][n_cases])
        cbar1.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[0,n_cases-1], colorbar_size_vec) 
        axs[0,n_cases].set_axes_locator(ip)
        axs[0,n_cases].set_ylabel('$u_x$',fontsize=5)
        cbar2=plot.colorbar(ux_err_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[1][n_cases])
        cbar2.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[1,n_cases-1], colorbar_size_vec) 
        axs[1,n_cases].set_axes_locator(ip)
        axs[1,n_cases].set_ylabel('$E(u_x)$',fontsize=5)
        cbar3=plot.colorbar(uy_plot,ticks=[0.7,0.35,0.0,-0.35,-0.7],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[2][n_cases])
        cbar3.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[2,n_cases-1], colorbar_size_vec) 
        axs[2,n_cases].set_axes_locator(ip)
        axs[2,n_cases].set_ylabel('$u_y$',fontsize=5)
        cbar4=plot.colorbar(uy_err_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[3][n_cases])
        cbar4.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[3,n_cases-1], colorbar_size_vec) 
        axs[3,n_cases].set_axes_locator(ip)
        axs[3,n_cases].set_ylabel('$E(u_y)$',fontsize=5)
        cbar5=plot.colorbar(p_plot,ticks=[1,0.5,0.0,-0.5,-1],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[4][n_cases])
        cbar5.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[4,n_cases-1], colorbar_size_vec) 
        axs[4,n_cases].set_axes_locator(ip)
        axs[4,n_cases].set_ylabel('$p$',fontsize=5)
        cbar6=plot.colorbar(p_err_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[5][n_cases])
        cbar6.outline.set_linewidth(0.1)
        ip = InsetPosition(axs[5,n_cases-1], colorbar_size_vec) 
        axs[5,n_cases].set_axes_locator(ip)
        axs[5,n_cases].set_ylabel('$E(p)$',fontsize=5)

        for s in range(6):
            axs[s,c+1].xaxis.set_tick_params(width=0.1,labelsize=4)
            axs[s,c+1].yaxis.set_tick_params(width=0.1,labelsize=4)


    for s in range(6):
        axs[s,c].xaxis.set_tick_params(width=0.1,labelsize=4)
        axs[s,c].yaxis.set_tick_params(width=0.1,labelsize=4)
        for axis in ['top','bottom','left','right']:
            axs[s,c].spines[axis].set_linewidth(0.1)
       
for a in range(6):
    for b in range(n_cases):
        axs[a,b].text(-1.9,1.5,'('+chr(a+97)+','+chr(b+97)+')',fontsize=5)      

plot.savefig(figures_dir+'meanFieldAssimilation_contour.pdf')
