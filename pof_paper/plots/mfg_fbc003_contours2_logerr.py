

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

cases_list = ['mfg_fbc003_001_S0/mfg_fbc003_001_S0_ep72927_pred.mat','mfg_fbc003_001_S2/mfg_fbc003_001_S2_ep74925_pred.mat','mfg_fbc003_001_S4/mfg_fbc003_001_S4_ep86913_pred.mat','mfg_fbc003_001_S8/mfg_fbc003_001_S8_ep101898_pred.mat','mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_pred.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_pred.mat']
errs_list = ['mfg_fbc003_001_S0/mfg_fbc003_001_S0_ep72927_error.mat','mfg_fbc003_001_S2/mfg_fbc003_001_S2_ep74925_error.mat','mfg_fbc003_001_S4/mfg_fbc003_001_S4_ep86913_error.mat','mfg_fbc003_001_S8/mfg_fbc003_001_S8_ep101898_error.mat','mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_error.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_error.mat']
cases_supersample_factor = [0,2,4,8,16,32]



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

mean_err_ux = []
mean_err_uy = []
mean_err_p = []

p95_err_ux = []
p95_err_uy = []
p95_err_p = []

mean_err_mx = []
mean_err_my = []
mean_err_mass = []
MAX_err_mx = []
MAX_err_my = []
MAX_err_mass = []

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

    ux_err_grid.append(ux_pred_grid[c]-ux_grid)
    uy_err_grid.append(uy_pred_grid[c]-uy_grid)
    p_err_grid.append(p_pred_grid[c]-p_grid)

    temp_err_ux = np.abs(ux_pred[c]-ux)
    temp_err_uy = np.abs(uy_pred[c]-uy)
    temp_err_p = np.abs(p_pred[c]-p)

    mean_err_ux.append(np.nanmean(temp_err_ux))
    mean_err_uy.append(np.nanmean(temp_err_uy))
    mean_err_p.append(np.nanmean(temp_err_p))

    p95_err_ux.append(np.nanpercentile(temp_err_ux,95))
    p95_err_uy.append(np.nanpercentile(temp_err_uy,95))
    p95_err_p.append(np.nanpercentile(temp_err_p,95))

    MAX_err_ux.append(np.nanmax(temp_err_ux))
    MAX_err_uy.append(np.nanmax(temp_err_uy))
    MAX_err_p.append(np.nanmax(temp_err_p))



    MAX_err_ux_all = np.nanmax([np.nanmax(temp_err_ux),MAX_err_ux_all])
    MAX_err_uy_all = np.nanmax([np.nanmax(temp_err_uy),MAX_err_uy_all])
    MAX_err_p_all = np.nanmax([np.nanmax(temp_err_p),MAX_err_p_all])
    MIN_err_ux_all = np.nanmin([np.nanmin(temp_err_ux),MIN_err_ux_all])
    MIN_err_uy_all = np.nanmin([np.nanmin(temp_err_uy),MIN_err_uy_all])
    MIN_err_p_all = np.nanmin([np.nanmin(temp_err_p),MIN_err_p_all])

    physFile = h5py.File(output_dir+errs_list[c],'r')
    mx_grid.append(np.reshape(np.array(physFile['mxr']),X_grid.shape))
    mx_grid[c][cylinder_mask] = np.NaN
    my_grid.append(np.reshape(np.array(physFile['myr']),X_grid.shape))
    my_grid[c][cylinder_mask] = np.NaN
    mass_grid.append(np.reshape(np.array(physFile['massr']),X_grid.shape))
    mass_grid[c][cylinder_mask] = np.NaN

    mean_err_mx.append(np.nanmean(np.abs(mx_grid[c])))
    mean_err_my.append(np.nanmean(np.abs(my_grid[c])))
    mean_err_mass.append(np.nanmean(np.abs(mass_grid[c])))
    MAX_err_mx.append(np.nanmax(np.abs(mx_grid[c])))
    MAX_err_my.append(np.nanmax(np.abs(my_grid[c])))
    MAX_err_mass.append(np.nanmax(np.abs(mass_grid[c])))

dx = [] # array for supersample spacing

print('Mean err ux:')
print(np.array(mean_err_ux))
print('Max err ux:')
print(np.array(MAX_err_ux))
print('Mean err uy:')
print(np.array(mean_err_uy))
print('Max err uy:')
print(np.array(MAX_err_uy))
print('Mean err p:')
print(np.array(mean_err_p))
print('Max err p:')
print(np.array(MAX_err_p))


print('Mean err mx:')
print(np.array(mean_err_mx))
print('Max err mx:')
print(np.array(MAX_err_mx))
print('Mean err my:')
print(np.array(mean_err_my))
print('Max err my:')
print(np.array(MAX_err_my))
print('Mean err mass:')
print(np.array(mean_err_mass))
print('Max err mass:')
print(np.array(MAX_err_mass))





text_color_tolerance = 1E-1

x_downsample_list = []
y_downsample_list = []

for c in range(len(cases_list)):
    if cases_supersample_factor[c]==0:
        dx.append(X_grid[1,0]-X_grid[0,0])

        x_downsample_list.append(X_grid.ravel())
        y_downsample_list.append(Y_grid.ravel())

    if cases_supersample_factor[c]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[c],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample = x[linear_downsample_inds]
        y_downsample = y[linear_downsample_inds]

        x_ds_grid = (np.reshape(x_downsample,(ndy,ndx))).transpose()
        y_ds_grid = (np.reshape(y_downsample,(ndy,ndx))).transpose()
        dx.append(x_ds_grid[1,0]-x_ds_grid[0,0])

        valid_inds = np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d
        x_downsample = x_downsample[valid_inds]
        y_downsample = y_downsample[valid_inds]
        
        text_corner_mask = (np.multiply(x_downsample<-0.9,y_downsample>1.5))<1
        x_downsample = x_downsample[text_corner_mask]
        y_downsample = y_downsample[text_corner_mask]
        text_corner_mask = (np.multiply(x_downsample>6.4,y_downsample>1))<1
        x_downsample = x_downsample[text_corner_mask]
        y_downsample = y_downsample[text_corner_mask]
        x_downsample_list.append(x_downsample)
        y_downsample_list.append(y_downsample)

for c in range(len(cases_list)):
    fig = plot.figure(figsize=(7,7))
    plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
    outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
    inner = []

    levels_err_ux = np.linspace(-MAX_err_ux[c],MAX_err_ux[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_uy = np.linspace(-MAX_err_uy[c],MAX_err_uy[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_p = np.linspace(-MAX_err_p[c],MAX_err_p[c],21)#np.power(10,np.linspace(-4,0,21))

    MAX_mx = np.nanmax(np.abs(mx_grid[c].ravel()))
    MAX_my = np.nanmax(np.abs(my_grid[c].ravel()))
    MAX_mass = np.nanmax(np.abs(mass_grid[c].ravel()))

    levels_mx = np.geomspace(1E-3,1,21)##
    levels_my = np.geomspace(1E-3,1,21)#np.linspace(1E-6,MAX_my,21)
    levels_mass = np.geomspace(1E-3,1,21)#np.linspace(1E-6,MAX_mass,21)

    levels_mx = np.geomspace(1E-3,1,11)##


    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

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
    dual_log_cbar_labels = ['-1','-1E-1','-1E-2','0','1E-2','1E-1','1']

    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.88,0.03,0.09]))

    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{x,DNS}$',fontsize=8)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(aa)',fontsize=8)
    fig.add_subplot(ax)
    
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    

    ax = plot.Subplot(fig,inner[0][3])
    ux_plot=ax.contourf(X_grid,Y_grid,ux_pred_grid[c],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{x,PINN}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(ab)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][4])
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[0][6])
    e_plot = ux_err_grid[c]/MAX_plot_ux
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if np.mean(e_plot.ravel())>text_color_tolerance:
        ax.text(6.5,1.2,'$|\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(\overline{u}_{x,DNS})}|$',fontsize=8,color='w')
        ax.text(-1.75,1.4,'(ac)',fontsize=8,color='w')
    else:
        ax.text(6.5,1.2,'$|\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(\overline{u}_{x,DNS})}|$',fontsize=8,color='k')
        ax.text(-1.75,1.4,'(ac)',fontsize=8,color='k')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][7])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    # quadrant 2

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.88,0.03,0.09]))

    ax = plot.Subplot(fig,inner[1][0])
    uy_plot =ax.contourf(X_grid,Y_grid,uy_grid,levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{y,DNS}$',fontsize=8)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(ba)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)


    ax = plot.Subplot(fig,inner[1][3])
    uy_plot =ax.contourf(X_grid,Y_grid,uy_pred_grid[c],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{y,PINN}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(bb)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][4])
    cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[1][6])
    e_plot = uy_err_grid[c]/MAX_plot_uy
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    t=ax.text(6.5,1.2,'$|\\frac{\overline{u}_{y,PINN}-\overline{u}_{y,DNS}}{max(\overline{u}_{y,DNS})}|$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(bc)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][7])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
 


    # quadrant 3

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.88,0.03,0.09]))

    ax = plot.Subplot(fig,inner[2][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{DNS}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(ca)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    
    ax = plot.Subplot(fig,inner[2][3])
    p_plot =ax.contourf(X_grid,Y_grid,p_pred_grid[c],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{PINN}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(cb)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][4])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[2][6])
    e_plot = p_err_grid[c]/MAX_plot_p
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlabel('x/D',fontsize=8)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if np.mean(e_plot.ravel())>text_color_tolerance:
        ax.text(7,1.2,'$|\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(\overline{p}_{DNS})}|$',fontsize=8,color='w')
        ax.text(-1.75,1.4,'(cc)',fontsize=8,color='w')
    else:
        ax.text(7,1.2,'$|\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(\overline{p}_{DNS})}|$',fontsize=8,color='k')
        ax.text(-1.75,1.4,'(cc)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][7])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    # quadrant 4

    inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.88,0.03,0.09]))

    ax = plot.Subplot(fig,inner[3][0])
    e_plot = mx_grid[c]
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    if np.mean(e_plot.ravel())>text_color_tolerance:
        ax.text(8,1.4,'$RANS_{x}$',fontsize=8,color='w')
        ax.text(-1.75,1.4,'(da)',fontsize=8,color='w')
    else:
        ax.text(8,1.4,'$RANS_{x}$',fontsize=8,color='k')
        ax.text(-1.75,1.4,'(da)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][3])
    e_plot = my_grid[c]
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')

    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    if np.mean(e_plot.ravel())>text_color_tolerance:
        ax.text(8,1.4,'$RANS_{y}$',fontsize=8,color='w')
        ax.text(-1.75,1.4,'(db)',fontsize=8,color='w')
    else:
        ax.text(8,1.4,'$RANS_{y}$',fontsize=8,color='k')
        ax.text(-1.75,1.4,'(db)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][4])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    ax = plot.Subplot(fig,inner[3][6])
    e_plot = mass_grid[c]
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_xlabel('x/D',fontsize=8)
    ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelleft=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    if np.mean(e_plot.ravel())>text_color_tolerance:
        ax.text(9,1.4,'$C$',fontsize=8,color='w')
        ax.text(-1.75,1.4,'(dc)',fontsize=8,color='w')
    else:
        ax.text(9,1.4,'$C$',fontsize=8,color='k')
        ax.text(-1.75,1.4,'(dc)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][7])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'logerr_mfg_fbc003_S'+str(cases_supersample_factor[c])+'.pdf')
    plot.savefig(figures_dir+'logerr_mfg_fbc003_S'+str(cases_supersample_factor[c])+'.png',dpi=300)

    plot.close(fig)





if True:
    # custom combined plot for the paper
    fig = plot.figure(figsize=(3.37,7))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.93,bottom=0.05)
    outer = gridspec.GridSpec(7,1,wspace=0.1,hspace=0.1)

    levels_err_ux = np.linspace(-MAX_err_ux[c],MAX_err_ux[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_uy = np.linspace(-MAX_err_uy[c],MAX_err_uy[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_p = np.linspace(-MAX_err_p[c],MAX_err_p[c],21)#np.power(10,np.linspace(-4,0,21))

    MAX_mx = np.nanmax(np.abs(mx_grid[c].ravel()))
    MAX_my = np.nanmax(np.abs(my_grid[c].ravel()))
    MAX_mass = np.nanmax(np.abs(mass_grid[c].ravel()))

    levels_mx = np.geomspace(1E-3,1,21)##

    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

    # fix the dots for the D/delta x label

    c=4
    text_corner_mask = (np.multiply(x_downsample_list[c]>6,y_downsample_list[c]<-1.5))<1
    x_downsample_list[c] = x_downsample_list[c][text_corner_mask]
    y_downsample_list[c] = y_downsample_list[c][text_corner_mask]
    c=5
    text_corner_mask = (np.multiply(x_downsample_list[c]>6,y_downsample_list[c]<-1.5))<1
    x_downsample_list[c] = x_downsample_list[c][text_corner_mask]
    y_downsample_list[c] = y_downsample_list[c][text_corner_mask]

    inner = []
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{x,DNS}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(a)',fontsize=8)
    fig.add_subplot(ax)
    
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    c=4
    ax = plot.Subplot(fig,inner[1][0])
    e_plot = np.abs(ux_err_grid[c]/MAX_plot_ux)+1E-30
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)

    ax.text(6.5,1.2,'$|\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(\overline{u}_{x,DNS})}|$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(b)',fontsize=8,color='k')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(ux_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    c=5
    ax = plot.Subplot(fig,inner[2][0])
    e_plot = np.abs(ux_err_grid[c]/MAX_plot_ux)+1E-30
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(6.5,1.2,'$|\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(\overline{u}_{x,DNS})}|$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(c)',fontsize=8,color='k')
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(ux_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{DNS}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(d)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    
    c=4
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    e_plot = np.abs(p_err_grid[c]/MAX_plot_p)+1E-30
    p_plot = ax.contourf(X_grid,Y_grid,e_plot,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(7,1.2,'$|\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(\overline{p}_{DNS})}|$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(e)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(p_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    c=5
    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_pred_grid[c],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{PINN}$',fontsize=8)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(f)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[6],wspace=0.05,hspace=0.1,width_ratios=[0.94,0.03,0.03]))
    
    ax = plot.Subplot(fig,inner[6][0])
    e_plot = np.abs(p_err_grid[c]/MAX_plot_p)+1E-30
    p_plot = ax.contourf(X_grid,Y_grid,e_plot,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(7,1.2,'$|\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(\overline{p}_{DNS})}|$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(g)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[6][1])
    cbar = plot.colorbar(p_plot,cax,ticks=ticks_mx,format=tkr.FormatStrFormatter('%.0e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)



    plot.savefig(figures_dir+'logerr_mfg_fbc003_condensed.pdf')
    plot.savefig(figures_dir+'logerr_mfg_fbc003_condensed.png',dpi=300)

    plot.close(fig)

    
    
    
    
    
    
    
    # custom combined plot for the paper
    fig = plot.figure(figsize=(3.37,7))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.05)
    outer = gridspec.GridSpec(7,1,wspace=0.1,hspace=0.1)

    levels_err_ux = np.linspace(-MAX_err_ux[c],MAX_err_ux[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_uy = np.linspace(-MAX_err_uy[c],MAX_err_uy[c],21)#np.power(10,np.linspace(-4,0,21))
    levels_err_p = np.linspace(-MAX_err_p[c],MAX_err_p[c],21)#np.power(10,np.linspace(-4,0,21))

    levels_mx = np.geomspace(1E-3,1,11)##


    ticks_mx = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

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
    dual_log_cbar_labels = ['-1','-1E-1','-1E-2','0','1E-2','1E-1','1']

    # fix the dots for the D/delta x label

    c=4
    text_corner_mask = (np.multiply(x_downsample_list[c]>6,y_downsample_list[c]<-1.5))<1
    x_downsample_list[c] = x_downsample_list[c][text_corner_mask]
    y_downsample_list[c] = y_downsample_list[c][text_corner_mask]
    c=5
    text_corner_mask = (np.multiply(x_downsample_list[c]>6,y_downsample_list[c]<-1.5))<1
    x_downsample_list[c] = x_downsample_list[c][text_corner_mask]
    y_downsample_list[c] = y_downsample_list[c][text_corner_mask]

    inner = []
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_grid,levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u}_{x,DNS}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(a)',fontsize=8)
    fig.add_subplot(ax)
    
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    c=4
    ax = plot.Subplot(fig,inner[1][0])
   
    e_plot = ux_err_grid[c]/MAX_plot_ux
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)


    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)

    ax.text(6.5,1.2,'$\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(|\overline{u}_{x,DNS}|)}$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(b)',fontsize=8,color='k')
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
    c=5
    ax = plot.Subplot(fig,inner[2][0])
    e_plot = ux_err_grid[c]/MAX_plot_ux
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    if cases_supersample_factor[c]>1:
        dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(6.5,1.2,'$\\frac{\overline{u}_{x,PINN}-\overline{u}_{x,DNS}}{max(|\overline{u}_{x,DNS}|)}$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(c)',fontsize=8,color='k')
    fig.add_subplot(ax)

    # these are for checking the color bars are accurate, thus commented 
    #cax=plot.Subplot(fig,inner[2][1])
    #cbar = plot.colorbar(ux_plot,cax,ticks=[1E-3,1E-2,1E-1,1],extend='both')
    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    #fig.add_subplot(cax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)


    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_grid,levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{DNS}$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(d)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    
    c=4
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    e_plot = p_err_grid[c]/MAX_plot_p
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    #if cases_supersample_factor[c]>1:
    #    dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(7,1.2,'$\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(|\overline{p}_{DNS}|)}$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 2.5$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(e)',fontsize=8)
    fig.add_subplot(ax)

    # dual log colorbar
    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    c=5
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    p_plot =ax.contourf(X_grid,Y_grid,p_pred_grid[c],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$\overline{p}_{PINN}$',fontsize=8)
    #if cases_supersample_factor[c]>1:
    #    dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(f)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(p_plot,cax,ticks=[MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[6],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    
    ax = plot.Subplot(fig,inner[6][0])
    e_plot = p_err_grid[c]/MAX_plot_p
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)
    p_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    p_plot2 = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
    #if cases_supersample_factor[c]>1:
    #    dots = ax.plot(x_downsample_list[c],y_downsample_list[c],markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(7,1.2,'$\\frac{\overline{p}_{PINN}-\overline{p}_{DNS}}{max(|\overline{p}_{DNS}|)}$',fontsize=8,color='k')
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.75,1.4,'(g)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[6][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)



    plot.savefig(figures_dir+'logerr_mfg_fbc003_condensed_dual.pdf')
    plot.savefig(figures_dir+'logerr_mfg_fbc003_condensed_dual.png',dpi=300)

    plot.close(fig)


# error percent plot
pts_per_d = 1.0/np.array(dx)

fig,axs = plot.subplots(3,1)
fig.set_size_inches(3.37,5.5)
plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.09)

mean_err_ux = np.array(mean_err_ux)
mean_err_uy = np.array(mean_err_uy)
mean_err_p = np.array(mean_err_p)

p95_err_ux = np.array(p95_err_ux)
p95_err_uy = np.array(p95_err_uy)
p95_err_p = np.array(p95_err_p)

MAX_err_ux = np.array(MAX_err_ux)
MAX_err_uy = np.array(MAX_err_uy)
MAX_err_p = np.array(MAX_err_p)

error_x_tick_labels = ['40','20','10','5','2.5','1.25']
error_y_ticks = [1E-3,1E-2,1E-1,1]
error_y_tick_labels = ['1E-3','1E-2','1E-1','1']

supersample_factors = np.array(cases_supersample_factor)
mean_plt,=axs[0].plot(pts_per_d,mean_err_ux,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
p95_plt,=axs[0].plot(pts_per_d,p95_err_ux,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
max_plt,=axs[0].plot(pts_per_d,MAX_err_ux,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
axs[0].set_xscale('log')
axs[0].set_xticks(pts_per_d)
axs[0].xaxis.set_tick_params(labelbottom=False)
axs[0].set_yscale('log')
axs[0].set_ylim(2E-4,1E0)
axs[0].set_yticks(error_y_ticks)
axs[0].set_yticklabels(error_y_tick_labels,fontsize=8)
axs[0].set_ylabel("Relative Error",fontsize=8)
axs[0].set_title('$\overline{u}_x$',fontsize=8)
axs[0].legend([mean_plt,p95_plt,max_plt],['Mean','95th Percentile','Max'],fontsize=8)
axs[0].grid('on')
axs[0].text(0.45,1.0,'(a)',fontsize=8)

axs[1].plot(pts_per_d,mean_err_uy,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
axs[1].plot(pts_per_d,p95_err_uy,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
axs[1].plot(pts_per_d,MAX_err_uy,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
axs[1].set_xscale('log')
axs[1].set_xticks(pts_per_d)
axs[1].xaxis.set_tick_params(labelbottom=False)
axs[1].set_yscale('log')
axs[1].set_ylim(2E-4,1E0)
axs[1].set_yticks(error_y_ticks)
axs[1].set_yticklabels(error_y_tick_labels,fontsize=8)
axs[1].set_ylabel("Relative Error",fontsize=8)
axs[1].set_title('$\overline{u}_y$',fontsize=8)
axs[1].grid('on')
axs[1].text(0.45,1.0,'(b)',fontsize=8)

axs[2].plot(pts_per_d,mean_err_p,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
axs[2].plot(pts_per_d,p95_err_p,linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
axs[2].plot(pts_per_d,MAX_err_p,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
axs[2].set_xscale('log')
axs[2].set_xticks(pts_per_d)
axs[2].set_xticklabels(error_x_tick_labels,fontsize=8)
axs[2].set_yscale('log')
axs[2].set_ylim(2E-4,1E0)
axs[2].set_yticks(error_y_ticks)
axs[2].set_yticklabels(error_y_tick_labels,fontsize=8)
axs[2].set_xlabel('$D/\Delta x$',fontsize=8)
axs[2].set_ylabel("Relative Error",fontsize=8)
axs[2].set_title('$\overline{p}$',fontsize=8)
axs[2].grid('on')
axs[2].text(0.45,1.0,'(c)',fontsize=8)

#fig.tight_layout()
plot.savefig(figures_dir+'logerr_mfg_fbc003_error.pdf')
plot.savefig(figures_dir+'logerr_mfg_fbc003_error.png',dpi=300)
plot.close(fig)



fig,axs = plot.subplots(1,1)
fig.set_size_inches(3.37,2.5)
plot.subplots_adjust(left=0.2,top=0.97,right=0.97,bottom=0.15)

error_x_tick_labels = ['40','20','10','5','2.5','1.25']
error_y_ticks = [1E-3,1E-2,1E-1,1]
error_y_tick_labels = ['1E-3','1E-2','1E-1','1']

supersample_factors = np.array(cases_supersample_factor)
mean_plt_x,=axs.plot(0.95*pts_per_d,mean_err_ux,linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
max_plt_x,=axs.plot(0.95*pts_per_d,MAX_err_ux,linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
mean_plt_y,=axs.plot(pts_per_d,mean_err_uy,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
max_plt_y,=axs.plot(pts_per_d,MAX_err_uy,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
mean_plt_p,=axs.plot(1.05*pts_per_d,mean_err_p,linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
max_plt_p,=axs.plot(1.05*pts_per_d,MAX_err_p,linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')
axs.set_xscale('log')
axs.set_xticks(pts_per_d)
axs.set_yscale('log')
axs.set_ylim(2E-4,1E0)
axs.set_yticks(error_y_ticks)
axs.set_yticklabels(error_y_tick_labels,fontsize=8)
axs.set_ylabel("Relative Error",fontsize=8)
axs.legend([mean_plt_x,max_plt_x,mean_plt_y,max_plt_y,mean_plt_p,max_plt_p,],['Mean $\overline{u}_x$','Max $\overline{u}_x$','Mean $\overline{u}_y$','Max $\overline{u}_y$','Mean $\overline{p}$','Max $\overline{p}$'],fontsize=8,ncols=2)
axs.grid('on')
axs.set_xticks(pts_per_d)
axs.set_xticklabels(error_x_tick_labels,fontsize=8)
axs.set_xlabel('$D/\Delta x$',fontsize=8,labelpad=-1)



#fig.tight_layout()
plot.savefig(figures_dir+'logerr_mfg_fbc003_error_condensed.pdf')
plot.savefig(figures_dir+'logerr_mfg_fbc003_error_condensed.png',dpi=300)
plot.close(fig)



