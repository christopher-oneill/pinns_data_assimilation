

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import sys
sys.path.append('C:/projects/pinns_local/code/')

# script

figures_dir = 'C:/projects/paper_figures/mean_field/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

# load the reference data
meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
configFile = h5py.File(data_dir+'configuration.mat','r')
meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
vorticityFile  = h5py.File(data_dir+'vorticity.mat','r')
                
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]
vorticity = np.array(vorticityFile['vorticity'][2000,:]).transpose()



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
uxux_grid = np.reshape(uxux,X_grid.shape)
uxux_grid[cylinder_mask] = np.NaN
uxuy_grid = np.reshape(uxuy,X_grid.shape)
uxuy_grid[cylinder_mask] = np.NaN
uyuy_grid = np.reshape(uyuy,X_grid.shape)
uyuy_grid[cylinder_mask] = np.NaN
p_grid = np.reshape(p,X_grid.shape)
p_grid[cylinder_mask] = np.NaN
vorticity_grid = np.reshape(vorticity,X_grid.shape)
vorticity_grid[cylinder_mask] = np.NaN

gridspec = {'width_ratios': [1, 0.15]}
fig,axs = plot.subplots(7,2,gridspec_kw=gridspec)
fig.set_size_inches(3.3,7)
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


plot.subplots_adjust(wspace=0.1, hspace=0.0)
ux_plot = axs[0,0].contourf(X_grid,Y_grid,ux_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[0,0].set_aspect('equal')
circle1 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[0,0].add_patch(circle1)

uy_plot =axs[1,0].contourf(X_grid,Y_grid,uy_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[1,0].set_aspect('equal')
circle2 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[1,0].add_patch(circle2)

uxux_plot =axs[2,0].contourf(X_grid,Y_grid,uxux_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[2,0].set_aspect('equal')
circle3 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[2,0].add_patch(circle3)

uxuy_plot =axs[3,0].contourf(X_grid,Y_grid,uxuy_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[3,0].set_aspect('equal')
circle4 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[3,0].add_patch(circle4)

uyuy_plot =axs[4,0].contourf(X_grid,Y_grid,uyuy_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[4,0].set_aspect('equal')
circle5 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[4,0].add_patch(circle5)

p_plot =axs[5,0].contourf(X_grid,Y_grid,p_grid,levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[5,0].set_aspect('equal')
circle6 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[5,0].add_patch(circle6)

omega_plot =axs[6,0].contourf(X_grid,Y_grid,np.clip(vorticity_grid,-2,2),levels=21,cmap= matplotlib.colormaps['bwr'],norm=matplotlib.colors.CenteredNorm())
axs[6,0].set_aspect('equal')
circle7 = plot.Circle((0, 0), 0.5, color='k',linewidth=0.1,fill=False)
axs[6,0].add_patch(circle7)

axs[0,0].set_ylabel('y/D',fontsize=4)
axs[1,0].set_ylabel('y/D',fontsize=4)
axs[2,0].set_ylabel('y/D',fontsize=4)
axs[3,0].set_ylabel('y/D',fontsize=4)
axs[4,0].set_ylabel('y/D',fontsize=4)
axs[5,0].set_ylabel('y/D',fontsize=4)
axs[6,0].set_ylabel('y/D',fontsize=4)

axs[6,0].set_xlabel('x/D',fontsize=4)

axs[0,0].xaxis.set_tick_params(labelbottom=False)
axs[1,0].xaxis.set_tick_params(labelbottom=False)
axs[2,0].xaxis.set_tick_params(labelbottom=False)
axs[3,0].xaxis.set_tick_params(labelbottom=False)
axs[4,0].xaxis.set_tick_params(labelbottom=False)
axs[5,0].xaxis.set_tick_params(labelbottom=False)


color_bar_fraction = 0.000016
colorbar_shrink=0.001
color_bar_pad = 0.00


    
colorbar_size_vec = [1.05,0,0.02,1.0]

cbar1 = plot.colorbar(ux_plot,fraction=color_bar_fraction,pad=color_bar_pad,cax=axs[0][1])
cbar1.outline.set_linewidth(0.1)
ip = InsetPosition(axs[0,0], colorbar_size_vec) 
axs[0,1].set_axes_locator(ip)
axs[0,1].set_ylabel('$\overline{u_x}$',fontsize=5)

cbar2=plot.colorbar(uy_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[1][1]) # ticks=[0.7,0.35,0.0,-0.35,-0.7]
cbar2.outline.set_linewidth(0.1)
ip = InsetPosition(axs[1,0], colorbar_size_vec) 
axs[1,1].set_axes_locator(ip)
axs[1,1].set_ylabel('$\overline{u_y}$',fontsize=5)

cbar3=plot.colorbar(uxux_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[2][1])
cbar3.outline.set_linewidth(0.1)
ip = InsetPosition(axs[2,0], colorbar_size_vec) 
axs[2,1].set_axes_locator(ip)
axs[2,1].set_ylabel('$\overline{u_x\'u_x\'}$',fontsize=5)

cbar4=plot.colorbar(uxuy_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[3][1])
cbar4.outline.set_linewidth(0.1)
ip = InsetPosition(axs[3,0], colorbar_size_vec) 
axs[3,1].set_axes_locator(ip)
axs[3,1].set_ylabel('$\overline{u_x\'u_y\'}$',fontsize=5)

cbar5=plot.colorbar(uyuy_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[4][1])
cbar5.outline.set_linewidth(0.1)
ip = InsetPosition(axs[4,0], colorbar_size_vec) 
axs[4,1].set_axes_locator(ip)
axs[4,1].set_ylabel('$\overline{u_y\'u_y\'}$',fontsize=5)

cbar6=plot.colorbar(p_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[5][1])
cbar6.outline.set_linewidth(0.1)
ip = InsetPosition(axs[5,0], colorbar_size_vec) 
axs[5,1].set_axes_locator(ip)
axs[5,1].set_ylabel('$\overline{p}$',fontsize=5)

cbar7=plot.colorbar(omega_plot,fraction=color_bar_fraction, pad=color_bar_pad,cax=axs[6][1],ticks=[-2,-1,0,1,2])
cbar7.ax.set_yticklabels(['<-2','-1','0','1','>2'])
cbar7.outline.set_linewidth(0.1)
ip = InsetPosition(axs[6,0], colorbar_size_vec) 
axs[6,1].set_axes_locator(ip)
axs[6,1].set_ylabel('$\omega_z, {tU_\infty}/{D}=?$',fontsize=5)

for s in range(7):
    axs[s,1].xaxis.set_tick_params(width=0.1,labelsize=4)
    axs[s,1].yaxis.set_tick_params(width=0.1,labelsize=4)

for s in range(7):
    axs[s,0].xaxis.set_tick_params(width=0.1,labelsize=4)
    axs[s,0].yaxis.set_tick_params(width=0.1,labelsize=4)
    for axis in ['top','bottom','left','right']:
        axs[s,0].spines[axis].set_linewidth(0.1)
       
for a in range(7):
    axs[a,0].text(-1.9,1.5,'('+chr(a+97)+')',fontsize=5)      


plot.savefig(figures_dir+'meanField_contour.pdf')

