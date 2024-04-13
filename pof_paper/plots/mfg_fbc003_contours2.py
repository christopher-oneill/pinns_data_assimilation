

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

cases_list = ['mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_pred.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_pred.mat']
errs_list = ['mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_error.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_error.mat']
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
MAX_p= 1.0 # estimated maximum pressure

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





# contour plot levels
levels_ux = np.linspace(-MAX_plot_ux,MAX_plot_ux,21)
levels_uy = np.linspace(-MAX_plot_uy,MAX_plot_uy,21)
levels_p = np.linspace(-MAX_plot_p,MAX_plot_p,21)

MAX_err_ux_all = 0.0
MAX_err_uy_all = 0.0
MAX_err_p_all = 0.0
MIN_err_ux_all = np.NaN
MIN_err_uy_all = np.NaN
MIN_err_p_all = np.NaN

ux_pred = []
uy_pred = []
p_pred = []

mx_grid = []
my_grid = []
mass_grid = []

ux_pred_grid = []
uy_pred_grid = []
p_pred_grid = []

ux_err_grid = []
uy_err_grid = []
p_err_grid = []

MAX_err_ux = []
MAX_err_uy = []
MAX_err_p = []

for c in range(n_cases):
    predFile = h5py.File(output_dir+cases_list[c],'r')
    
    ux_pred.append(np.array(predFile['pred'][:,0])*MAX_ux)
    uy_pred.append(np.array(predFile['pred'][:,1])*MAX_uy)
    p_pred.append(np.array(predFile['pred'][:,5])*MAX_p)

    ux_pred_grid.append(np.reshape(ux_pred[c],X_grid.shape))
    ux_pred_grid[c][cylinder_mask] = np.NaN
    uy_pred_grid.append(np.reshape(uy_pred[c],X_grid.shape))
    uy_pred_grid[c][cylinder_mask] = np.NaN
    p_pred_grid.append(np.reshape(p_pred[c],X_grid.shape))
    p_pred_grid[c][cylinder_mask] = np.NaN

    ux_err_grid.append(ux_grid-ux_pred_grid[c])
    uy_err_grid.append(uy_grid-uy_pred_grid[c])
    p_err_grid.append(p_grid-p_pred_grid[c])

    MAX_err_ux.append(np.nanmax(np.abs(ux-ux_pred[c])))
    MAX_err_uy.append(np.nanmax(np.abs(uy-uy_pred[c])))
    MAX_err_p.append(np.nanmax(np.abs(p-p_pred[c])))

    MAX_err_ux_all = np.nanmax([np.nanmax(np.abs((ux-ux_pred[c]))),MAX_err_ux_all])
    MAX_err_uy_all = np.nanmax([np.nanmax(np.abs((uy-uy_pred[c]))),MAX_err_uy_all])
    MAX_err_p_all = np.nanmax([np.nanmax(np.abs((p-p_pred[c]))),MAX_err_p_all])
    MIN_err_ux_all = np.nanmin([np.nanmin(np.abs((ux-ux_pred[c]))),MIN_err_ux_all])
    MIN_err_uy_all = np.nanmin([np.nanmin(np.abs((uy-uy_pred[c]))),MIN_err_uy_all])
    MIN_err_p_all = np.nanmin([np.nanmin(np.abs((p-p_pred[c]))),MIN_err_p_all])

    physFile = h5py.File(output_dir+errs_list[c],'r')
    mx_grid.append(np.reshape(np.array(physFile['mxr']),X_grid.shape))
    mx_grid[c][cylinder_mask] = np.NaN
    my_grid.append(np.reshape(np.array(physFile['myr']),X_grid.shape))
    my_grid[c][cylinder_mask] = np.NaN
    mass_grid.append(np.reshape(np.array(physFile['massr']),X_grid.shape))
    mass_grid[c][cylinder_mask] = np.NaN





fig = plot.figure(figsize=(7,7))
plot.subplots_adjust(left=0.05,top=0.95,right=0.95,bottom=0.05)
outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
inner = []

for c in range(len(cases_list)):
    levels_err_ux = np.linspace(-MAX_err_ux[c],MAX_err_ux[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_uy = np.linspace(-MAX_err_uy[c],MAX_err_uy[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_p = np.linspace(-MAX_err_p[c],MAX_err_p[c],21)#np.power(10,np.linspace(-4,0,21))

    MAX_mx = np.nanmax(np.abs(mx_grid[c].ravel()))
    MAX_my = np.nanmax(np.abs(my_grid[c].ravel()))
    MAX_mass = np.nanmax(np.abs(mass_grid[c].ravel()))

    levels_mx = np.linspace(1E-6,MAX_mx,21)
    levels_my = np.linspace(1E-6,MAX_my,21)
    levels_mass = np.linspace(1E-6,MAX_mass,21)

    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{x,DNS}$',fontsize=5)
    fig.add_subplot(ax)
    
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
    

    ax = plot.Subplot(fig,inner[0][3])
    ux_plot=ax.contourf(X_grid,Y_grid,ux_pred_grid[c],levels=levels_ux,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{x,PINN}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][4])
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[0][6])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_err_grid[c]/MAX_plot_ux,levels=levels_err_ux/MAX_plot_ux,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$\\frac{u_{x,DNS}-u_{x,PINN}}{max(u_{x,DNS})}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][7])
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_err_ux[c],MAX_err_ux[c]/2,0.0,-MAX_err_ux[c]/2,-MAX_err_ux[c]]/MAX_plot_ux,format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)


    # quadrant 2

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

    ax = plot.Subplot(fig,inner[1][0])
    uy_plot =ax.contourf(X_grid,Y_grid,uy_grid,levels=levels_uy,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{y,DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)


    ax = plot.Subplot(fig,inner[1][3])
    uy_plot =ax.contourf(X_grid,Y_grid,uy_pred_grid[c],levels=levels_uy,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.5,'$u_{y,PINN},$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][4])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[1][6])
    uy_plot = ax.contourf(X_grid,Y_grid,uy_err_grid[c]/MAX_plot_uy,levels=levels_err_uy/MAX_plot_uy,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(8,1.5,'$\\frac{u_{y,DNS}-u_{y,PINN}}{max(u_{y,DNS})}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][7])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_err_uy[c],MAX_err_uy[c]/2,0.0,-MAX_err_uy[c]/2,-MAX_err_uy[c]]/MAX_plot_uy,format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
 


    # quadrant 3

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

    ax = plot.Subplot(fig,inner[2][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(9,1.5,'$p_{DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)
    
    ax = plot.Subplot(fig,inner[2][3])
    p_plot =ax.contourf(X_grid,Y_grid,p_pred_grid[c],levels=levels_p,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(9,1.5,'$p_{PINN}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][4])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[2][6])
    p_plot = ax.contourf(X_grid,Y_grid,p_err_grid[c]/MAX_plot_p,levels=levels_err_p/MAX_plot_p,cmap= matplotlib.colormaps['bwr'])
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_ylabel('y/D',fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_xlabel('x/D',fontsize=5)
    ax.text(8,1.5,'$\\frac{p_{DNS}-p_{PINN}}{max(p_{DNS})}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][7])
    cbar = plot.colorbar(p_plot,cax,ticks=np.array([MAX_err_p[c],MAX_err_p[c]/2,0.0,-MAX_err_p[c]/2,-MAX_err_p[c]])/MAX_plot_p,format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    # quadrant 4

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

    ax = plot.Subplot(fig,inner[3][0])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['Reds'],norm=matplotlib.colors.LogNorm())
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$p_{DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(m_plot,cax,ticks=[MAX_mx,MAX_mx/2,1E-6],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][3])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['Reds'],norm=matplotlib.colors.LogNorm())
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$p_{DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][4])
    cbar = plot.colorbar(m_plot,cax,ticks=[MAX_mx,MAX_mx/2,1E-6],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][6])
    m_plot =ax.contourf(X_grid,Y_grid,np.abs(mx_grid[c])+1E-30,levels=levels_mx,cmap= matplotlib.colormaps['Reds'],norm=matplotlib.colors.LogNorm())
    ax.set_aspect('equal')
    ax.set_xlabel('x/D',fontsize=5)
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(9,1.5,'$p_{DNS}$',fontsize=5)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][7])
    cbar = plot.colorbar(m_plot,cax,ticks=[MAX_mx,MAX_mx/2,1E-6],format=tkr.FormatStrFormatter('%.3f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'mfg_fbc003_contours2_c'+str(c)+'.pdf')

exit()



