

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('F:/projects/pinns_local/code/')

figures_dir = 'F:/projects/paper_figures/t010_f2/thesis/overview/'
rec_dir = 'F:/projects/paper_figures/t010_f2/data/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

from pinns_data_assimilation.lib.file_util import find_highest_numbered_file
from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

create_directory_if_not_exists(figures_dir)

# load the reference data
base_dir = data_dir

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')


POD_dataFile = h5py.File(base_dir+'POD_data.mat','r')




x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

X_grid_plot = X_grid
Y_grid_plot = Y_grid
X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)

# load mean quantities
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

# values for scaling the NN outputs
MAX_ux = np.max(ux)
MAX_uy = np.max(uy)
MAX_p= 1.0 # estimated maximum pressure

# values for scaling the plots
MAX_plot_ux = np.max(np.abs(ux))
MAX_plot_uy = np.max(np.abs(uy))
MAX_plot_p = np.max(np.abs(p))

# load stresses
uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

# values for scaling the plots
MAX_plot_uxux = np.max(np.abs(uxux))
MAX_plot_uxuy = np.max(np.abs(uxuy))
MAX_plot_uyuy = np.max(np.abs(uyuy))


# compute the unsteady quantities
ux_p = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uy_p = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()
p_p = np.array(fluctuatingPressureFile['fluctuatingPressure'][:,:]).transpose()

uxt = ux_p+np.reshape(ux,[ux.shape[0],1])
uyt = uy_p+np.reshape(uy,[uy.shape[0],1])
pt = p_p+np.reshape(p,[uy.shape[0],1])

ux_p = uxt
uy_p = uyt
p_p = pt

# reshape all quantities to grids for plotting
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

ux = np.reshape(ux,[X_grid.shape[0],X_grid.shape[1]])
ux[cylinder_mask] = np.NaN
uy = np.reshape(uy,[X_grid.shape[0],X_grid.shape[1]])
uy[cylinder_mask] = np.NaN
p = np.reshape(p,[X_grid.shape[0],X_grid.shape[1]])
p[cylinder_mask] = np.NaN

uxux = np.reshape(uxux,[X_grid.shape[0],X_grid.shape[1]])
uxux[cylinder_mask] = np.NaN
uxuy = np.reshape(uxuy,[X_grid.shape[0],X_grid.shape[1]])
uxuy[cylinder_mask] = np.NaN
uyuy = np.reshape(uyuy,[X_grid.shape[0],X_grid.shape[1]])
uyuy[cylinder_mask] = np.NaN

uxt = np.reshape(uxt,[X_grid.shape[0],X_grid.shape[1],uxt.shape[1]])
uyt = np.reshape(uyt,[X_grid.shape[0],X_grid.shape[1],uyt.shape[1]])
pt = np.reshape(pt,[X_grid.shape[0],X_grid.shape[1],pt.shape[1]])


# contour plot levels
levels_ux = np.linspace(-MAX_plot_ux,MAX_plot_ux,21)
ux_ticks = [MAX_plot_ux,MAX_plot_ux/2,0.0,-MAX_plot_ux/2,-MAX_plot_ux]
levels_uy = np.linspace(-MAX_plot_uy,MAX_plot_uy,21)
uy_ticks = [MAX_plot_uy,MAX_plot_uy/2,0.0,-MAX_plot_uy/2,-MAX_plot_uy]
levels_p = np.linspace(-MAX_plot_p,MAX_plot_p,21)
p_ticks = [MAX_plot_p,MAX_plot_p/2,0.0,-MAX_plot_p/2,-MAX_plot_p]
levels_uxux = np.linspace(-MAX_plot_uxux,MAX_plot_uxux,21)
uxux_ticks = [MAX_plot_uxux,MAX_plot_uxux/2,0.0,-MAX_plot_uxux/2,-MAX_plot_uxux]
levels_uxuy = np.linspace(-MAX_plot_uxuy,MAX_plot_uxuy,21)
uxuy_ticks = [MAX_plot_uxuy,MAX_plot_uxuy/2,0.0,-MAX_plot_uxuy/2,-MAX_plot_uxuy]
levels_uyuy = np.linspace(-MAX_plot_uyuy,MAX_plot_uyuy,21)
uyuy_ticks = [MAX_plot_uyuy,MAX_plot_uyuy/2,0.0,-MAX_plot_uyuy/2,-MAX_plot_uyuy]

x_ticks = [-2,0,2,4,6,8,10]
y_ticks = [-2,0,2]

# plot the time averaged quantities
if True:
    fig = plot.figure(figsize=(7.5,3.6))
    plot.subplots_adjust(left=0.06,top=0.97,right=0.93,bottom=0.1)
    outer = gridspec.GridSpec(3,2,wspace=0.23,hspace=0.1)

    inner = []
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux,levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u_x}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=ux_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uy,levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u_y}$',fontsize=8)
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[1][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=uy_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # p
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])
    ux_plot = ax.contourf(X_grid,Y_grid,p,levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(7.5,1.4,'$\overline{p}$',fontsize=8)
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[2][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=p_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxux
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uxux,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u\'_xu\'_x}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=uxux_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uxuy,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7.5,1.4,'$\overline{u\'_xu\'_y}$',fontsize=8)
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[4][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=uxuy_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uyuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uyuy,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(7.5,1.4,'$\overline{u\'_yu\'_y}$',fontsize=8)
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[5][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=uyuy_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'mean_quanties.png',dpi=600)
    plot.close(fig)




# compute the phase average based on the POD phase angle
n_x = int(np.array(POD_dataFile['n_x']))
phi_x_temp = np.array(POD_dataFile['Phi_ext'][:,0:n_x]).transpose()
phi_y_temp = np.array(POD_dataFile['Phi_ext'][:,n_x:2*n_x]).transpose()
n_trunc = int(np.array(POD_dataFile['Phi']).shape[0])
Ak_temp = np.array(POD_dataFile['Ak']).transpose()

phase = np.arctan2(Ak_temp[:,1],Ak_temp[:,0])

n_bins = 12
phase_bin_edges = np.linspace(np.pi,-np.pi,n_bins+1)
phase_bin_centers = 0.5*(phase_bin_edges[0:-1]+phase_bin_edges[1:])

phase_indices = np.digitize(phase,phase_bin_edges[1:]) # the implementation here only needs the right side of the bins

ux_p_pa = np.zeros((ux_p.shape[0],n_bins)) # nx by n_bins
uy_p_pa = np.zeros((ux_p.shape[0],n_bins)) # nx by n_bins
p_p_pa = np.zeros((ux_p.shape[0],n_bins)) # nx by n_bins



for j in range(n_bins):
    ux_p_pa[:,j] = np.mean(ux_p[:,phase_indices==j],axis=1)
    uy_p_pa[:,j] = np.mean(uy_p[:,phase_indices==j],axis=1)
    p_p_pa[:,j] = np.mean(p_p[:,phase_indices==j],axis=1)

ux_p_pa = np.reshape(ux_p_pa,[X_grid.shape[0],X_grid.shape[1],n_bins])
uy_p_pa = np.reshape(uy_p_pa,[X_grid.shape[0],X_grid.shape[1],n_bins])
p_p_pa = np.reshape(p_p_pa,[X_grid.shape[0],X_grid.shape[1],n_bins])

from pinns_data_assimilation.lib.vortex import vorticity
from pinns_data_assimilation.lib.vortex import Qcriterion

vort_pa = vorticity(ux_p_pa,uy_p_pa,X_grid,Y_grid)

vort_levels = np.linspace(-2,2,21)
vort_ticks = [-2,-1,0,1,2]

if True:
    fig = plot.figure(figsize=(7.5,3.6))
    plot.subplots_adjust(left=0.06,top=0.97,right=0.93,bottom=0.1)
    outer = gridspec.GridSpec(3,2,wspace=0.23,hspace=0.1)

    inner = []
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,0],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=1/12',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)

    ax.text(5.5,0.5,'v1',fontsize=8,color='white')
    ax.text(3.25,-0.5,'v2',fontsize=8)
    ax.text(1.5,0.25,'v3',fontsize=8,color='white')

    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,1],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=2/12',fontsize=8)
    ax.text(-1.85,1.45,'(b)',fontsize=8)

    ax.text(6,0.5,'v1',fontsize=8,color='white')
    ax.text(3.5,-0.5,'v2',fontsize=8)
    ax.text(1.5,0.25,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[1][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # p
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,2],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=3/12',fontsize=8)
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.25,0.5,'v1',fontsize=8,color='white')
    ax.text(4,-0.5,'v2',fontsize=8)
    ax.text(2,0.25,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[2][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxux
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,3],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=4/12',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    ax.text(6.75,0.5,'v1',fontsize=8,color='white')
    ax.text(4.25,-0.5,'v2',fontsize=8)
    ax.text(2,0.5,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,4],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=5/12',fontsize=8)
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(7,0.5,'v1',fontsize=8,color='white')
    ax.text(4.75,-0.5,'v2',fontsize=8)
    ax.text(2,0.5,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[4][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uyuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,5],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=6/12',fontsize=8)
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(7.5,0.5,'v1',fontsize=8,color='white')
    ax.text(5,-0.5,'v2',fontsize=8)
    ax.text(2.5,0.5,'v3',fontsize=8,color='white')
    ax.text(1.25,-0.25,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[5][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'vorticity_phase_average6.png',dpi=600)
    plot.close(fig)

if True:
    fig = plot.figure(figsize=(7.5,6.6))
    plot.subplots_adjust(left=0.06,top=0.97,right=0.93,bottom=0.1)
    outer = gridspec.GridSpec(6,2,wspace=0.23,hspace=0.1)

    inner = []
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,0],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=1/12',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)

    ax.text(5.5,0.5,'v1',fontsize=8,color='white')
    ax.text(3.25,-0.5,'v2',fontsize=8)
    ax.text(1.5,0.25,'v3',fontsize=8,color='white')

    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,1],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=2/12',fontsize=8)
    ax.text(-1.85,1.45,'(b)',fontsize=8)

    ax.text(6,0.5,'v1',fontsize=8,color='white')
    ax.text(3.5,-0.5,'v2',fontsize=8)
    ax.text(1.5,0.25,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[1][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # p
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,2],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    #ax.set_xlabel('x/D',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=3/12',fontsize=8)
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.25,0.5,'v1',fontsize=8,color='white')
    ax.text(4,-0.5,'v2',fontsize=8)
    ax.text(2,0.25,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[2][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxux
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[6],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,3],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=4/12',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    ax.text(6.75,0.5,'v1',fontsize=8,color='white')
    ax.text(4.25,-0.5,'v2',fontsize=8)
    ax.text(2,0.5,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[8],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,4],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=5/12',fontsize=8)
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(7,0.5,'v1',fontsize=8,color='white')
    ax.text(4.75,-0.5,'v2',fontsize=8)
    ax.text(2,0.5,'v3',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[4][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uyuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[10],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,5],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=6/12',fontsize=8)
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(7.5,0.5,'v1',fontsize=8,color='white')
    ax.text(5,-0.5,'v2',fontsize=8)
    ax.text(2.5,0.5,'v3',fontsize=8,color='white')
    ax.text(1.25,-0.25,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[5][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[6][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,6],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=7/12',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)

    ax.text(7.75,0.5,'v1',fontsize=8,color='white')
    ax.text(5.5,-0.5,'v2',fontsize=8)
    ax.text(3.25,0.5,'v3',fontsize=8,color='white')
    ax.text(1.5,-0.25,'v4',fontsize=8)

    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,7],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=8/12',fontsize=8)
    ax.text(-1.85,1.45,'(h)',fontsize=8)

    ax.text(8.25,0.5,'v1',fontsize=8,color='white')
    ax.text(6,-0.5,'v2',fontsize=8)
    ax.text(3.5,0.5,'v3',fontsize=8,color='white')
    ax.text(1.5,-0.25,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[7][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # p
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,8],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=9/12',fontsize=8)
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(8.75,0.5,'v1',fontsize=8,color='white')
    ax.text(6.25,-0.5,'v2',fontsize=8)
    ax.text(4,0.5,'v3',fontsize=8,color='white')
    ax.text(2,-0.25,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[8][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxux
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[9][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,9],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=10/12',fontsize=8)
    ax.text(-1.85,1.45,'(j)',fontsize=8)
    ax.text(9.25,0.5,'v1',fontsize=8,color='white')
    ax.text(6.75,-0.5,'v2',fontsize=8)
    ax.text(4.25,0.5,'v3',fontsize=8,color='white')
    ax.text(2,-0.5,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[9][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uxuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[9],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[10][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,10],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(6.5,1.4,'Phase=11/12',fontsize=8)
    ax.text(-1.85,1.45,'(k)',fontsize=8)
    ax.text(7,-0.5,'v2',fontsize=8)
    ax.text(4.75,0.5,'v3',fontsize=8,color='white')
    ax.text(2.25,-0.5,'v4',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[10][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    # uyuy
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[11],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[11][0])
    ux_plot = ax.contourf(X_grid,Y_grid,vort_pa[:,:,11],levels=vort_levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x',fontsize=8,labelpad=2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True,labelsize=8)
    ax.text(6.5,1.4,'Phase=12/12',fontsize=8)
    ax.text(-1.85,1.45,'(l)',fontsize=8)
    ax.text(7.5,-0.5,'v2',fontsize=8)
    ax.text(5,0.5,'v3',fontsize=8,color='white')
    ax.text(2.75,-0.5,'v4',fontsize=8)
    ax.text(1.25,0.25,'v5',fontsize=8,color='white')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[11][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=vort_ticks,format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'vorticity_phase_average12.png',dpi=600)
    plot.close(fig)


