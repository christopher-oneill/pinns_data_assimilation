

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
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

cases_list = ['mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_pred.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_pred.mat']
cases_supersample_factor = [16,32]



# load the reference data
meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
configFile = h5py.File(data_dir+'configuration.mat','r')
meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
                
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = (np.array(meanPressureFile['meanPressure']).transpose())[:,0]

# values for scaling the NN outputs
MAX_ux = np.max(ux)
MAX_uy = np.max(uy)
MAX_p= 1 # estimated maximum pressure

# values for scaling the plots
MAX_plot_ux = np.max(np.abs(ux))
MAX_plot_uy = np.max(np.abs(uy))
MAX_plot_p = np.max(np.abs(p))

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
gridspec = {'width_ratios': [1, 0.15, 1, 1, 0.15]}
fig,axs = plot.subplots(3,5,gridspec_kw=gridspec)
fig.set_size_inches(15,5)
fig.tight_layout()


# contour plot levels
levels_ux = np.linspace(-MAX_plot_ux,MAX_plot_ux,21)
levels_uy = np.linspace(-MAX_plot_uy,MAX_plot_uy,21)
levels_p = np.linspace(-MAX_plot_p,MAX_plot_p,21)

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

levels_err_ux = np.power(10,np.linspace(-4,0,21))
levels_err_uy = np.power(10,np.linspace(-4,0,21))
levels_err_p = np.power(10,np.linspace(-4,0,21))

# mean fields

plot.subplots_adjust(wspace=0.1, hspace=0.1)
ux_plot = axs[0,0].contourf(X_grid,Y_grid,ux_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'])
axs[0,0].set_aspect('equal')
axs[0,0].set_title('Training Data',fontsize=6)
axs[0,0].set_ylabel('y/D',fontsize=4)
axs[0,0].xaxis.set_tick_params(labelbottom=False)
axs[0,0].text(9,1.5,'$u_x$',fontsize=10)

uy_plot =axs[1,0].contourf(X_grid,Y_grid,uy_grid,levels=levels_uy,cmap= matplotlib.colormaps['bwr'])
axs[1,0].set_aspect('equal')
axs[1,0].set_ylabel('y/D',fontsize=4)
axs[1,0].xaxis.set_tick_params(labelbottom=False)
axs[1,0].text(9,1.5,'$u_y$',fontsize=10)

p_plot =axs[2,0].contourf(X_grid,Y_grid,p_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'])
axs[2,0].set_aspect('equal')
axs[2,0].set_ylabel('y/D',fontsize=4)
axs[2,0].text(9,1.5,'$p$',fontsize=10)



        

colorbar_size_vec = [1.05,0,0.02,1.0]
color_bar_fraction = 0.000016
colorbar_shrink=0.001
color_bar_pad = 0.00

cbar1 = plot.colorbar(ux_plot,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],aspect=40,pad=color_bar_pad,cax=axs[0][1])
cbar1.outline.set_linewidth(0.1)
ip = InsetPosition(axs[0,0], colorbar_size_vec) 
axs[0,1].set_axes_locator(ip)


cbar3=plot.colorbar(uy_plot,ticks=[MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[1][1])
cbar3.outline.set_linewidth(0.1)
ip = InsetPosition(axs[1,0], colorbar_size_vec) 
axs[1,1].set_axes_locator(ip)

cbar5=plot.colorbar(p_plot,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[2][1])
cbar5.outline.set_linewidth(0.1)
ip = InsetPosition(axs[2,0], colorbar_size_vec) 
axs[2,1].set_axes_locator(ip)





#error plots
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

    ux_err = np.abs((ux_grid-ux_pred_grid))
    uy_err =  np.abs((uy_grid-uy_pred_grid))
    p_err =  np.abs((p_grid-p_pred_grid))
  
    # compute points if needed
    if cases_supersample_factor[c]>1:
        n_x = X_grid.shape[0]
        n_y = X_grid.shape[1]
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[c],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample = x[linear_downsample_inds]
        y_downsample = y[linear_downsample_inds]
        valid_inds = np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d
        x_downsample = x_downsample[valid_inds]
        y_downsample = y_downsample[valid_inds]

    MAX_plot_ux_err = np.nanmax(ux_err)
    MAX_plot_uy_err = np.nanmax(uy_err)
    MAX_plot_p_err = np.nanmax(p_err)



    ux_err_plot =axs[0,c+2].contourf(X_grid,Y_grid,ux_err,levels=levels_err_ux,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(),extend='both')
    axs[0,c+2].set_aspect('equal')
    axs[0,c+2].set_title('S*='+str(cases_supersample_factor[c]),fontsize=6)
    if cases_supersample_factor[c]>1:
        dots = axs[0,c+2].plot(x_downsample,y_downsample,markersize=1,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')

    uy_err_plot =axs[1,c+2].contourf(X_grid,Y_grid,uy_err,levels=levels_err_uy,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(),extend='both') #levels_err_uy,norm=matplotlib.colors.LogNorm(1E-6,1)
    axs[1,c+2].set_aspect('equal')
    if cases_supersample_factor[c]>1:
        dots = axs[1,c+2].plot(x_downsample,y_downsample,markersize=1,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')

    p_err_plot =axs[2,c+2].contourf(X_grid,Y_grid,p_err,levels=levels_err_p,cmap= matplotlib.colormaps['inferno'],norm=matplotlib.colors.LogNorm(),extend='both')
    axs[2,c+2].set_aspect('equal')

    axs[0,c+2].xaxis.set_tick_params(labelbottom=False)
    axs[1,c+2].xaxis.set_tick_params(labelbottom=False)

    axs[0,c+2].yaxis.set_tick_params(labelleft=False)
    axs[1,c+2].yaxis.set_tick_params(labelleft=False)
    axs[2,c+2].yaxis.set_tick_params(labelleft=False)

if True:
    cbar = plot.colorbar(ux_err_plot,ticks=[1E0,1E-1,1E-2,1E-3,1E-4],fraction=color_bar_fraction,aspect=40,pad=color_bar_pad,cax=axs[0][4]) # ticks=[MAX_plot_ux_err,MAX_plot_ux_err/2,0.0,],
    cbar.outline.set_linewidth(0.1)
    ip = InsetPosition(axs[0,3], colorbar_size_vec) 
    axs[0,4].set_axes_locator(ip)

    cbar=plot.colorbar(uy_err_plot,ticks=[1E0,1E-1,1E-2,1E-3,1E-4],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[1][4]) # ticks=[MAX_plot_uy_err,MAX_plot_uy_err/2,0.0,],
    cbar.outline.set_linewidth(0.1)
    ip = InsetPosition(axs[1,3], colorbar_size_vec) 
    axs[1,4].set_axes_locator(ip)

    cbar=plot.colorbar(p_err_plot,ticks=[1E0,1E-1,1E-2,1E-3,1E-4],fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[2][4]) # ticks=[MAX_plot_p_err,MAX_plot_p_err/2,0.0,],
    cbar.outline.set_linewidth(0.1)
    ip = InsetPosition(axs[2,3], colorbar_size_vec) 
    axs[2,4].set_axes_locator(ip)


plot.savefig(figures_dir+'mfg_fbc003_contours_new.pdf')
exit()

for c in range(n_cases):

    if c==0:

        axs[2,c].set_ylabel('y/D',fontsize=4)
        axs[3,c].set_ylabel('y/D',fontsize=4)
        axs[4,c].set_ylabel('y/D',fontsize=4)
        axs[5,c].set_ylabel('y/D',fontsize=4)

    axs[5,c].set_xlabel('x/D',fontsize=4)    
    for s in range(3):
        axs[s,c+1].xaxis.set_tick_params(width=0.1,labelsize=4)
        axs[s,c+1].yaxis.set_tick_params(width=0.1,labelsize=4)


    for s in range(3):
        axs[s,c].xaxis.set_tick_params(width=0.1,labelsize=4)
        axs[s,c].yaxis.set_tick_params(width=0.1,labelsize=4)
        for axis in ['top','bottom','left','right']:
            axs[s,c].spines[axis].set_linewidth(0.1)
       
for a in range(6):
    for b in range(n_cases):
        axs[a,b].text(-1.9,1.5,'('+chr(a+97)+','+chr(b+97)+')',fontsize=5)      









