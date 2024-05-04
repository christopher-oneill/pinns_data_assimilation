
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

figures_dir = 'C:/projects/paper_figures/reconstruction/'
rec_dir = 'C:/projects/paper_figures/data/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

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


ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure'][0,:]).transpose()

ux_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][0,:,:]).transpose()
uy_ref = np.array(fluctuatingVelocityFile['fluctuatingVelocity'][1,:,:]).transpose()
p_ref = np.array(fluctuatingPressureFile['fluctuatingPressure'][:,:]).transpose()

L_dft=4082

ux_ref = np.reshape(ux_ref,[X_grid.shape[0],X_grid.shape[1],ux_ref.shape[1]])
uy_ref = np.reshape(uy_ref,[X_grid.shape[0],X_grid.shape[1],uy_ref.shape[1]])
p_ref = np.reshape(p_ref,[X_grid.shape[0],X_grid.shape[1],p_ref.shape[1]])


fs=10.0
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

# crop the fluctuating fields to the first 4082 so they are the same as the fourier data
ux_ref = ux_ref[:,:,0:L_dft]
uy_ref = uy_ref[:,:,0:L_dft]
p_ref = p_ref[:,:,0:L_dft]

# compute reference reynolds stresses 
uxux_ref = np.mean(np.multiply(ux_ref,ux_ref),axis=2)
uxuy_ref = np.mean(np.multiply(ux_ref,uy_ref),axis=2)
uyuy_ref = np.mean(np.multiply(uy_ref,uy_ref),axis=2)

uxux_ref[cylinder_mask]=np.NaN
uxuy_ref[cylinder_mask]=np.NaN
uyuy_ref[cylinder_mask]=np.NaN

MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))

# load the reference fourier reconstructions
uxux_rec_ref = []
uxuy_rec_ref = []
uyuy_rec_ref = []

rec_mode_vec = [0,1,2]
cases_supersample_factor = [0,2,4,8,16,32]

mean_rec_err_uxux = []
mean_rec_err_uxuy = []
mean_rec_err_uyuy = []

p95_rec_err_uxux = []
p95_rec_err_uxuy = []
p95_rec_err_uyuy = []

max_rec_err_uxux = []
max_rec_err_uxuy = []
max_rec_err_uyuy = []

for c in [0,1,2,3,4,5]:
    recFile = h5py.File(rec_dir+'rec_fourier_c'+str(c)+'.h5','r')
    ux_rec_ref = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    uy_rec_ref = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    ux_rec_ref_m = np.reshape(np.mean(ux_rec_ref,axis=2),[ux_rec_ref.shape[0],ux_rec_ref.shape[1],1])
    uy_rec_ref_m = np.reshape(np.mean(uy_rec_ref,axis=2),[uy_rec_ref.shape[0],uy_rec_ref.shape[1],1])
    uxux_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,ux_rec_ref-ux_rec_ref_m),axis=2))
    uxuy_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uyuy_rec_ref.append(np.mean(np.multiply(uy_rec_ref-uy_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uxux_rec_ref[c][cylinder_mask]=np.NaN
    uxuy_rec_ref[c][cylinder_mask]=np.NaN
    uyuy_rec_ref[c][cylinder_mask]=np.NaN

    mean_rec_err_uxux.append(np.nanmean((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel()))
    mean_rec_err_uxuy.append(np.nanmean((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel()))
    mean_rec_err_uyuy.append(np.nanmean((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel()))

    p95_rec_err_uxux.append(np.nanpercentile((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel(),95))
    p95_rec_err_uxuy.append(np.nanpercentile((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel(),95))
    p95_rec_err_uyuy.append(np.nanpercentile((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel(),95))

    max_rec_err_uxux.append(np.nanmax((np.abs(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref).ravel()))
    max_rec_err_uxuy.append(np.nanmax((np.abs(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref).ravel()))
    max_rec_err_uyuy.append(np.nanmax((np.abs(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref).ravel()))


# generate the contour profiles
profile_locations = np.linspace(-1.5,9.5,22) # spacing of 0.5
X_line_locations = X_grid_plot[:,0]
X_distance_matrix = np.power(np.power(np.reshape(X_line_locations,[X_line_locations.size,1])-np.reshape(profile_locations,[1,profile_locations.size]),2.0),0.5)
# find index of closest data line
line_inds = np.argmin(X_distance_matrix,axis=0)
profile_locations = X_grid_plot[line_inds,:]
point_locations = Y_grid_plot[0,:]

profile_x_offset = np.array([0,0,0])
profile_x_scale = np.array([0.5,0.5,0.5])


if False:
    s=0 # needed so no points appear in the reference comparison plotss
    for c in [0,1,2,3,4,5]:

        if True:

            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
            MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
            MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
            
            levels_uxux = 1.1*np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
            levels_uxuy = 1.1*np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
            levels_uyuy = 1.1*np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)
            
            levels_err = np.geomspace(1E-3,1,21)
            ticks_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

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
            ux_plot = ax.contourf(X_grid,Y_grid,uxux_ref,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(aa)',fontsize=5,color='k')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec_ref[c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(ab)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs((uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{FMD}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.1e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_ref,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ba)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec_ref[c],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(bb)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{FMD}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(bc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_ref,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ca)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec_ref[c],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(cb)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{FMD}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxux_ref[line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxux_rec_ref[c][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.text(-1.75,1.5,'(da)',fontsize=5,color='k')
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{x}}_{DNS}$','$\overline{u\'_{x}u\'_{x}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)
        
            ax = plot.Subplot(fig,inner[3][3])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxuy_ref[line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxuy_rec_ref[c][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(-1.75,1.5,'(db)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{y}}_{DNS}$','$\overline{u\'_{x}u\'_{y}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            ax = plot.Subplot(fig,inner[3][6])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uyuy_ref[line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uyuy_rec_ref[c][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(-1.75,1.5,'(dc)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{y}u\'_{y}}_{DNS}$','$\overline{u\'_{y}u\'_{y}}_{FMD}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            # for now empty ...

            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_recref_c'+str(c)+'.pdf')
            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_recref_c'+str(c)+'.png',dpi=300)
            plot.close('all')

if False:



    # compare the different number of modes by Restress error
    fig,axs = plot.subplots(3,1)
    fig.set_size_inches(3.37,5.5)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.09)

    mean_plt,=axs[0].plot(pts_per_d,mean_err_uxux[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    p95_plt,=axs[0].plot(pts_per_d,p95_err_uxux[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    max_plt,=axs[0].plot(pts_per_d,max_err_uxux[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[0].set_xscale('log')
    axs[0].set_xticks(pts_per_d)
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[0].set_yscale('log')
    axs[0].set_ylim(5E-5,1E0)
    axs[0].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    axs[0].set_ylabel("Relative Error",fontsize=7)
    axs[0].set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs[0].legend([mean_plt,p95_plt,max_plt],['Mean','95th Percentile','Max'],fontsize=5)
    axs[0].grid('on')
    axs[0].text(0.45,10.0,'(a)',fontsize=10)

    axs[1].plot(pts_per_d,mean_err_uxuy[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[1].plot(pts_per_d,p95_err_uxuy[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[1].plot(pts_per_d,max_err_uxuy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[1].set_xscale('log')
    axs[1].set_xticks(pts_per_d)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[1].set_yscale('log')
    axs[1].set_ylim(5E-5,1E0)
    axs[1].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    #axs[1].set_yticklabels([0.1,0.5,1.0])
    axs[1].set_ylabel("Relative Error",fontsize=7)
    axs[1].set_title('$\overline{u\'_{x}u\'_{y}}$')
    axs[1].grid('on')
    axs[1].text(0.45,10.0,'(b)',fontsize=10)

    axs[2].plot(pts_per_d,mean_err_uyuy[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[2].plot(pts_per_d,p95_err_uyuy[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[2].plot(pts_per_d,max_err_uyuy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[2].set_xscale('log')
    axs[2].set_xticks(pts_per_d)
    axs[2].set_xticklabels(error_x_tick_labels)
    axs[2].set_yscale('log')
    axs[2].set_ylim(5E-5,1E0)
    axs[2].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    #axs[2].set_yticklabels([0.1,0.5,1.0,2.0])
    axs[2].set_xlabel('Pts/D',fontsize=7)
    axs[2].set_ylabel("Relative Error",fontsize=7)
    axs[2].set_title('$\overline{u\'_{y}u\'_{y}}$')
    axs[2].grid('on')
    axs[2].text(0.45,10.0,'(c)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_f'+str(c)+'_error.pdf')
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_f'+str(c)+'_error.png',dpi=300)
    plot.close(fig)          


dx = [] # array for supersample spacing
for s in range(len(cases_supersample_factor)):
    dx.append([])
    for c in [0,1,2]:
        if cases_supersample_factor[s]==0:
            dx[s].append(X_grid[1,0]-X_grid[0,0])

        if cases_supersample_factor[s]>0:
            linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

            x_downsample = x[linear_downsample_inds]
            y_downsample = y[linear_downsample_inds]

            x_ds_grid = (np.reshape(x_downsample,(ndy,ndx))).transpose()
            y_ds_grid = (np.reshape(y_downsample,(ndy,ndx))).transpose()
            dx[s].append(x_ds_grid[1,0]-x_ds_grid[0,0])


# load the reconstuctions and compute the reynolds stresses
uxux_rec = []
uxuy_rec = []
uyuy_rec = []
k_rec = []

for s in range(len(cases_supersample_factor)):
    uxux_rec.append([])
    uxuy_rec.append([])
    uyuy_rec.append([])
    k_rec.append([])
    for c in rec_mode_vec:
        recFile = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.h5','r')
        ux_rec = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        uy_rec = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        ux_rec_m = np.reshape(np.mean(ux_rec,axis=2),[ux_rec.shape[0],ux_rec.shape[1],1])
        uy_rec_m = np.reshape(np.mean(uy_rec,axis=2),[uy_rec.shape[0],uy_rec.shape[1],1])
        uxux_rec[s].append(np.mean(np.multiply(ux_rec-ux_rec_m,ux_rec-ux_rec_m),axis=2))
        uxuy_rec[s].append(np.mean(np.multiply(ux_rec-ux_rec_m,uy_rec-uy_rec_m),axis=2))
        uyuy_rec[s].append(np.mean(np.multiply(uy_rec-uy_rec_m,uy_rec-uy_rec_m),axis=2))
        uxux_rec[s][c][cylinder_mask]=np.NaN
        uxuy_rec[s][c][cylinder_mask]=np.NaN
        uyuy_rec[s][c][cylinder_mask]=np.NaN
        k_rec[s].append(0.5*np.mean(np.power(ux_rec,2.0)+np.power(uy_rec,2.0),axis=2))
        k_rec[s][c][cylinder_mask]=np.NaN

if False:
    for c in rec_mode_vec:

        for s in range(len(cases_supersample_factor)):


            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
            MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
            MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
            
            levels_uxux = 1.1*np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
            levels_uxuy = 1.1*np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
            levels_uyuy = 1.1*np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)
            
            levels_err = np.geomspace(1E-3,1,21)
            ticks_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])

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
            ux_plot = ax.contourf(X_grid,Y_grid,uxux_ref,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(aa)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec[s][c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(ab)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs((uxux_ref-uxux_rec[s][c])/MAX_uxux_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)

            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{PINN}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.1e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_ref,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ba)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec[s][c],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(bb)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_ref-uxuy_rec[s][c])/MAX_uxuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{PINN}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(bc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_ref,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{DNS}$',fontsize=5)
            ax.text(-1.75,1.5,'(ca)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec[s][c],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(cb)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_ref-uyuy_rec[s][c])/MAX_uyuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{PINN}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxux_ref[line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxux_rec[s][c][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.text(-1.75,1.5,'(da)',fontsize=5,color='k')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{x}}_{DNS}$','$\overline{u\'_{x}u\'_{x}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)
        
            ax = plot.Subplot(fig,inner[3][3])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxuy_ref[line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxuy_rec[s][c][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.text(-1.75,1.5,'(db)',fontsize=5,color='k')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{y}}_{DNS}$','$\overline{u\'_{x}u\'_{y}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            ax = plot.Subplot(fig,inner[3][6])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uyuy_ref[line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uyuy_rec[s][c][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.text(-1.75,1.5,'(dc)',fontsize=5,color='k')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{y}u\'_{y}}_{DNS}$','$\overline{u\'_{y}u\'_{y}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.pdf')
            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.png',dpi=300)
            plot.close('all')

            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            # quadrant 1

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            # (1,(1,1))
            ax = plot.Subplot(fig,inner[0][0])
            ux_plot = ax.contourf(X_grid,Y_grid,uxux_rec_ref[c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(aa)',fontsize=5,color='k')
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec[s][c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(ab)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs((uxux_rec_ref[c]-uxux_rec[s][c])/MAX_uxux_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}-\overline{u\'_{x}u\'_{x}}_{PINN}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.1e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec_ref[c],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(ba)',fontsize=5,color='k')
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec[s][c],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(bb)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_rec_ref[c]-uxuy_rec[s][c])/MAX_uxuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{FMD}-\overline{u\'_{x}u\'_{y}}_{PINN}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(bc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec_ref[c],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{FMD}$',fontsize=5)
            ax.text(-1.75,1.5,'(ca)',fontsize=5,color='k')
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec[s][c],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}_{PINN}$',fontsize=5)
            ax.text(-1.75,1.5,'(cb)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_rec_ref[c]-uyuy_rec[s][c])/MAX_uyuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            circle = plot.Circle((0,0),0.5,color='k',fill=False)
            ax.add_patch(circle)
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{FMD}-\overline{u\'_{y}u\'_{y}}_{PINN}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxux_rec_ref[c][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxux_rec[s][c][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.text(-1.75,1.5,'(da)',fontsize=5,color='k')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{x}}_{FMD}$','$\overline{u\'_{x}u\'_{x}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)
        
            ax = plot.Subplot(fig,inner[3][3])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxuy_rec_ref[c][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxuy_rec[s][c][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.text(-1.75,1.5,'(db)',fontsize=5,color='k')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{x}u\'_{y}}_{FMD}$','$\overline{u\'_{x}u\'_{y}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            ax = plot.Subplot(fig,inner[3][6])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uyuy_rec_ref[c][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uyuy_rec[s][c][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            ax.set_ylim(-2,2)
            ax.set_xlim(-2,10)
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(-1.75,1.5,'(dc)',fontsize=5,color='k')
            circle = plot.Circle((0,0),0.5,color='w',fill=True,zorder=2)
            ax.add_patch(circle)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=3)
            ax.add_patch(circle)
            ax.legend([line0,line1,line2],['0.0','$\overline{u\'_{y}u\'_{y}}_{FMD}$','$\overline{u\'_{y}u\'_{y}}_{PINN}$'],bbox_to_anchor=(1.0,0.75),fontsize=5,framealpha=0.0)
            fig.add_subplot(ax)

            # for now empty ...

            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.pdf')
            plot.savefig(figures_dir+'logerr_reynoldsStress_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.png',dpi=300)
            plot.close('all')


mean_err_uxux = []
mean_err_uxuy = []
mean_err_uyuy = []

p95_err_uxux = []
p95_err_uxuy = []
p95_err_uyuy = []

max_err_uxux = []
max_err_uxuy = []
max_err_uyuy = []

for s in range(len(cases_supersample_factor)):

    mean_err_uxux.append([])
    mean_err_uxuy.append([])
    mean_err_uyuy.append([])

    p95_err_uxux.append([])
    p95_err_uxuy.append([])
    p95_err_uyuy.append([])

    max_err_uxux.append([])
    max_err_uxuy.append([])
    max_err_uyuy.append([])
    for c in rec_mode_vec:
        err_uxux = (np.abs(uxux_ref-uxux_rec[s][c])/MAX_uxux_ref).ravel()
        err_uxuy = (np.abs(uxuy_ref-uxuy_rec[s][c])/MAX_uxuy_ref).ravel()
        err_uyuy = (np.abs(uyuy_ref-uyuy_rec[s][c])/MAX_uyuy_ref).ravel()

        mean_err_uxux[s].append(np.nanmean(err_uxux))
        mean_err_uxuy[s].append(np.nanmean(err_uxuy))
        mean_err_uyuy[s].append(np.nanmean(err_uyuy))

        p95_err_uxux[s].append(np.nanpercentile(err_uxux,95))
        p95_err_uxuy[s].append(np.nanpercentile(err_uxuy,95))
        p95_err_uyuy[s].append(np.nanpercentile(err_uyuy,95))

        max_err_uxux[s].append(np.nanmax(err_uxux))
        max_err_uxuy[s].append(np.nanmax(err_uxuy))
        max_err_uyuy[s].append(np.nanmax(err_uyuy))



# error percent plot
pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,2]


mean_err_uxux = np.array(mean_err_uxux)
mean_err_uxuy = np.array(mean_err_uxuy)
mean_err_uyuy = np.array(mean_err_uyuy)

p95_err_uxux = np.array(p95_err_uxux)
p95_err_uxuy = np.array(p95_err_uxuy)
p95_err_uyuy = np.array(p95_err_uyuy)

max_err_uxux = np.array(max_err_uxux)
max_err_uxuy = np.array(max_err_uxuy)
max_err_uyuy = np.array(max_err_uyuy)

error_x_tick_labels = ['40','20','10','5','2.5','1.25']
error_y_tick_labels = ['1E-4','1E-3','1E-2','1E-1']

for c in [0,1,2]:
    fig,axs = plot.subplots(3,1)
    fig.set_size_inches(3.37,5.5)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.09)

    mean_plt,=axs[0].plot(pts_per_d,mean_err_uxux[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    p95_plt,=axs[0].plot(pts_per_d,p95_err_uxux[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    max_plt,=axs[0].plot(pts_per_d,max_err_uxux[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[0].set_xscale('log')
    axs[0].set_xticks(pts_per_d)
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[0].set_yscale('log')
    axs[0].set_ylim(5E-5,1E0)
    axs[0].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    axs[0].set_ylabel("Relative Error",fontsize=7)
    axs[0].set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs[0].legend([mean_plt,p95_plt,max_plt],['Mean','95th Percentile','Max'],fontsize=5)
    axs[0].grid('on')
    axs[0].text(0.45,10.0,'(a)',fontsize=10)

    axs[1].plot(pts_per_d,mean_err_uxuy[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[1].plot(pts_per_d,p95_err_uxuy[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[1].plot(pts_per_d,max_err_uxuy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[1].set_xscale('log')
    axs[1].set_xticks(pts_per_d)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[1].set_yscale('log')
    axs[1].set_ylim(5E-5,1E0)
    axs[1].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    #axs[1].set_yticklabels([0.1,0.5,1.0])
    axs[1].set_ylabel("Relative Error",fontsize=7)
    axs[1].set_title('$\overline{u\'_{x}u\'_{y}}$')
    axs[1].grid('on')
    axs[1].text(0.45,10.0,'(b)',fontsize=10)

    axs[2].plot(pts_per_d,mean_err_uyuy[:,c],linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    axs[2].plot(pts_per_d,p95_err_uyuy[:,c],linewidth=0,marker='^',color='orange',markersize=3,markerfacecolor='orange')
    axs[2].plot(pts_per_d,max_err_uyuy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs[2].set_xscale('log')
    axs[2].set_xticks(pts_per_d)
    axs[2].set_xticklabels(error_x_tick_labels)
    axs[2].set_yscale('log')
    axs[2].set_ylim(5E-5,1E0)
    axs[2].set_yticks([1E-4,1E-3,1E-2,1E-1],labels=error_y_tick_labels,fontsize=7)
    #axs[2].set_yticklabels([0.1,0.5,1.0,2.0])
    axs[2].set_xlabel('Pts/D',fontsize=7)
    axs[2].set_ylabel("Relative Error",fontsize=7)
    axs[2].set_title('$\overline{u\'_{y}u\'_{y}}$')
    axs[2].grid('on')
    axs[2].text(0.45,10.0,'(c)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_f'+str(c)+'_error.pdf')
    plot.savefig(figures_dir+'logerr_rec_reynoldsStress_f'+str(c)+'_error.png',dpi=300)
    plot.close(fig)
