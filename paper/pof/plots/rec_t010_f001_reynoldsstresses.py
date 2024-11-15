
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('F:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center

figures_dir = 'F:/projects/paper_figures/t010/rec_reynoldsStress/'
rec_dir = 'F:/projects/paper_figures/t010/data/'
data_dir = 'F:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'F:/projects/pinns_narval/sync/output/'

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

rec_mode_vec = [0,1,2,3,4,5]
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


if True:
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




for ic in range(len(rec_mode_vec)):
    recFile = h5py.File(rec_dir+'rec_fourier_c'+str(rec_mode_vec[ic])+'.h5','r')
    ux_rec_ref = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    uy_rec_ref = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    ux_rec_ref_m = np.reshape(np.mean(ux_rec_ref,axis=2),[ux_rec_ref.shape[0],ux_rec_ref.shape[1],1])
    uy_rec_ref_m = np.reshape(np.mean(uy_rec_ref,axis=2),[uy_rec_ref.shape[0],uy_rec_ref.shape[1],1])
    uxux_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,ux_rec_ref-ux_rec_ref_m),axis=2))
    uxuy_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uyuy_rec_ref.append(np.mean(np.multiply(uy_rec_ref-uy_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uxux_rec_ref[ic][cylinder_mask]=np.NaN
    uxuy_rec_ref[ic][cylinder_mask]=np.NaN
    uyuy_rec_ref[ic][cylinder_mask]=np.NaN

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

dx = [] # array for supersample spacing
for s in range(len(cases_supersample_factor)):
    dx.append([])
    for ic in range(len(rec_mode_vec)):
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
for s in range(len(cases_supersample_factor)):
    uxux_rec.append([])
    uxuy_rec.append([])
    uyuy_rec.append([])
    for ic in range(len(rec_mode_vec)):
        recFile = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[ic])+'.h5','r')
        ux_rec = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        uy_rec = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
        ux_rec_m = np.reshape(np.mean(ux_rec,axis=2),[ux_rec.shape[0],ux_rec.shape[1],1])
        uy_rec_m = np.reshape(np.mean(uy_rec,axis=2),[uy_rec.shape[0],uy_rec.shape[1],1])
        uxux_rec[s].append(np.mean(np.multiply(ux_rec-ux_rec_m,ux_rec-ux_rec_m),axis=2))
        uxuy_rec[s].append(np.mean(np.multiply(ux_rec-ux_rec_m,uy_rec-uy_rec_m),axis=2))
        uyuy_rec[s].append(np.mean(np.multiply(uy_rec-uy_rec_m,uy_rec-uy_rec_m),axis=2))
        uxux_rec[s][ic][cylinder_mask]=np.NaN
        uxuy_rec[s][ic][cylinder_mask]=np.NaN
        uyuy_rec[s][ic][cylinder_mask]=np.NaN

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

if True:
    for ic in range(len(rec_mode_vec)):

        create_directory_if_not_exists(figures_dir+'c'+str(rec_mode_vec[ic])+'/')

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
            ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec[s][ic],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
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

            e_plot = (uxux_ref-uxux_rec[s][ic])/MAX_uxux_ref
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
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{PINN}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(ac)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
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
            uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec[s][ic],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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

            e_plot = (uxuy_ref-uxuy_rec[s][ic])/MAX_uxuy_ref
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
            t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{PINN}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(bc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
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
            p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec[s][ic],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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
            e_plot = (uyuy_ref-uyuy_rec[s][ic])/MAX_uyuy_ref
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
            ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{PINN}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
            ax.text(-1.75,1.5,'(cc)',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
            cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            for k in range(profile_locations.shape[0]):
                line0,=ax.plot(np.zeros(point_locations.shape)+profile_locations[k],point_locations,'--k',linewidth=0.5)
                line1,=ax.plot((uxux_ref[line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                line2,=ax.plot((uxux_rec[s][ic][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

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
                line2,=ax.plot((uxuy_rec[s][ic][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
            
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
                line2,=ax.plot((uyuy_rec[s][ic][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
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

            
            plot.savefig(figures_dir+'c'+str(rec_mode_vec[ic])+'/logerr_reynoldsStress_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[ic])+'.pdf')
            plot.savefig(figures_dir+'c'+str(rec_mode_vec[ic])+'/logerr_reynoldsStress_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[ic])+'.png',dpi=300)
            plot.close('all')


            if False:
                fig = plot.figure(figsize=(7,7))
                plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
                outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
                inner = []

                # quadrant 1

                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

                # (1,(1,1))
                ax = plot.Subplot(fig,inner[0][0])
                ux_plot = ax.contourf(X_grid,Y_grid,uxux_rec_ref[ic],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec[s][ic],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                ux_plot = ax.contourf(X_grid,Y_grid,np.abs((uxux_rec_ref[ic]-uxux_rec[s][ic])/MAX_uxux_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
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
                uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec_ref[ic],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                uy_plot =ax.contourf(X_grid,Y_grid,uxuy_rec[s][ic],levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_rec_ref[ic]-uxuy_rec[s][ic])/MAX_uxuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
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
                p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec_ref[ic],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                p_plot =ax.contourf(X_grid,Y_grid,uyuy_rec[s][ic],levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
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
                p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_rec_ref[ic]-uyuy_rec[s][ic])/MAX_uyuy_ref)+1E-30,levels=levels_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
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
                    line1,=ax.plot((uxux_rec_ref[ic][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                    line2,=ax.plot((uxux_rec[s][ic][line_inds[k],:]/MAX_uxux_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)

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
                    line1,=ax.plot((uxuy_rec_ref[ic][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                    line2,=ax.plot((uxuy_rec[s][ic][line_inds[k],:]/MAX_uxuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
                
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
                    line1,=ax.plot((uyuy_rec_ref[ic][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-k',linewidth=0.5)
                    line2,=ax.plot((uyuy_rec[s][ic][line_inds[k],:]/MAX_uyuy_ref+profile_x_offset[0])*profile_x_scale[0]+profile_locations[k],point_locations,'-r',linewidth=0.5)
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
                if rec_mode_vec[ic]==5:
                    plot.savefig(figures_dir+'logerr_reynoldsStress_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[ic])+'.pdf')
                    plot.savefig(figures_dir+'logerr_reynoldsStress_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[ic])+'.png',dpi=300)
                plot.close('all')


if True:
    # get the data

    MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
    MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
    MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
            
    levels_uxux = np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
    levels_uxuy = np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
    levels_uyuy = np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)

    ic=5
    s=4
    uxux_err_grid1 = uxux_rec[s][ic] - uxux_ref
    uxuy_err_grid1 = uxuy_rec[s][ic] - uxuy_ref
    uyuy_err_grid1 = uyuy_rec[s][ic] - uyuy_ref

    uxux_err_grid1[cylinder_mask]=np.NaN
    uxuy_err_grid1[cylinder_mask]=np.NaN
    uyuy_err_grid1[cylinder_mask]=np.NaN

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    # grid the data
    s=5
    uxux_err_grid2 = uxux_rec[s][ic] - uxux_ref
    uxuy_err_grid2 = uxuy_rec[s][ic] - uxuy_ref
    uyuy_err_grid2 = uyuy_rec[s][ic] - uyuy_ref

    uxux_err_grid2[cylinder_mask]=np.NaN
    uxuy_err_grid2[cylinder_mask]=np.NaN
    uyuy_err_grid2[cylinder_mask]=np.NaN

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>5.75,y_downsample1>0.8))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    text_corner_mask = (np.multiply(x_downsample2>5.75,y_downsample2>0.8))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]



    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(3.37,8))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.04)
    outer = gridspec.GridSpec(9,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uxux_ref,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_x}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = uxux_err_grid1/MAX_uxux_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_x}_{\mathrm{PINN}})$',fontsize=8,color='k')
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

    e_plot = uxux_err_grid2/MAX_uxux_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_x}_{\mathrm{PINN}})$',fontsize=8,color='k')
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
    ux_plot = ax.contourf(X_grid,Y_grid,uxuy_ref,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_y}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = uxuy_err_grid1/MAX_uxuy_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
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

    e_plot = uxuy_err_grid2/MAX_uxuy_ref
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
    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
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
    ux_plot = ax.contourf(X_grid,Y_grid,uyuy_ref,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_y}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])

    e_plot = uyuy_err_grid1/MAX_uyuy_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_yu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
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

    e_plot = uyuy_err_grid2/MAX_uyuy_ref
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
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.2,'$\eta(\overline{u\'_yu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[8][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
    
    plot.savefig(figures_dir+'c5/logerr_reynoldsStress_contours_ref_pinn_S16S32_c5.pdf')
    plot.savefig(figures_dir+'c5/logerr_reynoldsStress_contours_ref_pinn_S16S32_c5.png',dpi=300)
    plot.close(fig)



    ic=5
    s=3
    uxux_err_grid1 = uxux_rec[s][ic] - uxux_ref
    uxuy_err_grid1 = uxuy_rec[s][ic] - uxuy_ref
    uyuy_err_grid1 = uyuy_rec[s][ic] - uyuy_ref

    uxux_err_grid1[cylinder_mask]=np.NaN
    uxuy_err_grid1[cylinder_mask]=np.NaN
    uyuy_err_grid1[cylinder_mask]=np.NaN

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    # grid the data
    s=5
    uxux_err_grid2 = uxux_rec[s][ic] - uxux_ref
    uxuy_err_grid2 = uxuy_rec[s][ic] - uxuy_ref
    uyuy_err_grid2 = uyuy_rec[s][ic] - uyuy_ref

    uxux_err_grid2[cylinder_mask]=np.NaN
    uxuy_err_grid2[cylinder_mask]=np.NaN
    uyuy_err_grid2[cylinder_mask]=np.NaN

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>5.75,y_downsample1>0.8))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    text_corner_mask = (np.multiply(x_downsample2>5.75,y_downsample2>0.8))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]



    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(3.37,8))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.04)
    outer = gridspec.GridSpec(9,1,wspace=0.1,hspace=0.1)
    inner = []

    x_ticks = np.array([-2,0,2,4,6,8,10])
    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uxux_ref,levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_x}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = uxux_err_grid1/MAX_uxux_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_x}_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
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

    e_plot = uxux_err_grid2/MAX_uxux_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_x}_{\mathrm{PINN}})$',fontsize=8,color='k')
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
    ux_plot = ax.contourf(X_grid,Y_grid,uxuy_ref,levels=levels_uxuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_y}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])

    e_plot = uxuy_err_grid1/MAX_uxuy_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[4][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])

    e_plot = uxuy_err_grid2/MAX_uxuy_ref
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
    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.2,'$\eta(\overline{u\'_xu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
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
    ux_plot = ax.contourf(X_grid,Y_grid,uyuy_ref,levels=levels_uyuy,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(7,1.3,'$\overline{u\'_xu\'_y}_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])

    e_plot = uyuy_err_grid1/MAX_uyuy_ref
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
    ax.text(6.5,1.2,'$\eta(\overline{u\'_yu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(h)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[7][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[8],wspace=0.02,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])

    e_plot = uyuy_err_grid2/MAX_uyuy_ref
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
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.2,'$\eta(\overline{u\'_yu\'_y}_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[8][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
    
    plot.savefig(figures_dir+'c5/logerr_reynoldsStress_contours_ref_pinn_S8S32_c5.pdf')
    plot.savefig(figures_dir+'c5/logerr_reynoldsStress_contours_ref_pinn_S8S32_c5.png',dpi=300)
    plot.close(fig)

mean_err_uxux = []
mean_err_uxuy = []
mean_err_uyuy = []

p95_err_uxux = []
p95_err_uxuy = []
p95_err_uyuy = []

max_err_uxux = []
max_err_uxuy = []
max_err_uyuy = []

mean_err_uxux_FMD = []
mean_err_uxuy_FMD = []
mean_err_uyuy_FMD = []

max_err_uxux_FMD = []
max_err_uxuy_FMD = []
max_err_uyuy_FMD = []

mean_err_uxux_PINN_FMD = []
mean_err_uxuy_PINN_FMD = []
mean_err_uyuy_PINN_FMD = []

max_err_uxux_PINN_FMD = []
max_err_uxuy_PINN_FMD = []
max_err_uyuy_PINN_FMD = []


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

    mean_err_uxux_FMD.append([])
    mean_err_uxuy_FMD.append([])
    mean_err_uyuy_FMD.append([])

    max_err_uxux_FMD.append([])
    max_err_uxuy_FMD.append([])
    max_err_uyuy_FMD.append([])

    mean_err_uxux_PINN_FMD.append([])
    mean_err_uxuy_PINN_FMD.append([])
    mean_err_uyuy_PINN_FMD.append([])

    max_err_uxux_PINN_FMD.append([])
    max_err_uxuy_PINN_FMD.append([])
    max_err_uyuy_PINN_FMD.append([])


    for ic in range(len(rec_mode_vec)):
        err_uxux = (np.abs(uxux_rec[s][ic]-uxux_ref)/MAX_uxux_ref).ravel()
        err_uxuy = (np.abs(uxuy_rec[s][ic]-uxuy_ref)/MAX_uxuy_ref).ravel()
        err_uyuy = (np.abs(uyuy_rec[s][ic]-uyuy_ref)/MAX_uyuy_ref).ravel()

        mean_err_uxux[s].append(np.nanmean(err_uxux))
        mean_err_uxuy[s].append(np.nanmean(err_uxuy))
        mean_err_uyuy[s].append(np.nanmean(err_uyuy))

        p95_err_uxux[s].append(np.nanpercentile(err_uxux,95))
        p95_err_uxuy[s].append(np.nanpercentile(err_uxuy,95))
        p95_err_uyuy[s].append(np.nanpercentile(err_uyuy,95))

        max_err_uxux[s].append(np.nanmax(err_uxux))
        max_err_uxuy[s].append(np.nanmax(err_uxuy))
        max_err_uyuy[s].append(np.nanmax(err_uyuy))

        err_uxux_FMD = (np.abs(uxux_rec_ref[ic]-uxux_ref)/MAX_uxux_ref).ravel()
        err_uxuy_FMD = (np.abs(uxuy_rec_ref[ic]-uxuy_ref)/MAX_uxuy_ref).ravel()
        err_uyuy_FMD = (np.abs(uyuy_rec_ref[ic]-uyuy_ref)/MAX_uyuy_ref).ravel()

        mean_err_uxux_FMD[s].append(np.nanmean(err_uxux_FMD))
        mean_err_uxuy_FMD[s].append(np.nanmean(err_uxuy_FMD))
        mean_err_uyuy_FMD[s].append(np.nanmean(err_uyuy_FMD))

        max_err_uxux_FMD[s].append(np.nanmax(err_uxux_FMD))
        max_err_uxuy_FMD[s].append(np.nanmax(err_uxuy_FMD))
        max_err_uyuy_FMD[s].append(np.nanmax(err_uyuy_FMD))

        err_uxux_PINN_FMD = (np.abs(uxux_rec[s][ic]-uxux_rec_ref[ic])/MAX_uxux_ref).ravel()
        err_uxuy_PINN_FMD = (np.abs(uxuy_rec[s][ic]-uxuy_rec_ref[ic])/MAX_uxuy_ref).ravel()
        err_uyuy_PINN_FMD = (np.abs(uyuy_rec[s][ic]-uyuy_rec_ref[ic])/MAX_uyuy_ref).ravel()

        mean_err_uxux_PINN_FMD[s].append(np.nanmean(err_uxux_PINN_FMD))
        mean_err_uxuy_PINN_FMD[s].append(np.nanmean(err_uxuy_PINN_FMD))
        mean_err_uyuy_PINN_FMD[s].append(np.nanmean(err_uyuy_PINN_FMD))

        max_err_uxux_PINN_FMD[s].append(np.nanmax(err_uxux_PINN_FMD))
        max_err_uxuy_PINN_FMD[s].append(np.nanmax(err_uxuy_PINN_FMD))
        max_err_uyuy_PINN_FMD[s].append(np.nanmax(err_uyuy_PINN_FMD))






# error percent plot
pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,0]


mean_err_uxux = np.array(mean_err_uxux)
mean_err_uxuy = np.array(mean_err_uxuy)
mean_err_uyuy = np.array(mean_err_uyuy)

p95_err_uxux = np.array(p95_err_uxux)
p95_err_uxuy = np.array(p95_err_uxuy)
p95_err_uyuy = np.array(p95_err_uyuy)

max_err_uxux = np.array(max_err_uxux)
max_err_uxuy = np.array(max_err_uxuy)
max_err_uyuy = np.array(max_err_uyuy)

mean_err_uxux_FMD = np.array(mean_err_uxux_FMD)
mean_err_uxuy_FMD = np.array(mean_err_uxuy_FMD)
mean_err_uyuy_FMD = np.array(mean_err_uyuy_FMD)

max_err_uxux_FMD = np.array(max_err_uxux_FMD)
max_err_uxuy_FMD = np.array(max_err_uxuy_FMD)
max_err_uyuy_FMD = np.array(max_err_uyuy_FMD)

mean_err_uxux_PINN_FMD = np.array(mean_err_uxux_PINN_FMD)
mean_err_uxuy_PINN_FMD = np.array(mean_err_uxuy_PINN_FMD)
mean_err_uyuy_PINN_FMD = np.array(mean_err_uyuy_PINN_FMD)

max_err_uxux_PINN_FMD = np.array(max_err_uxux_PINN_FMD)
max_err_uxuy_PINN_FMD = np.array(max_err_uxuy_PINN_FMD)
max_err_uyuy_PINN_FMD = np.array(max_err_uyuy_PINN_FMD)

err_file = open(figures_dir+'Err_Rec_ReynoldsStresses.txt','w')

err_file.write('Err FMD Reynolds Stresses\n')
for ic in range(len(rec_mode_vec)):
    err_file.write('Mode ')
    err_file.write(str(rec_mode_vec[ic]))
    err_file.write('\nmean uxux: ')
    err_file.write(str(mean_err_uxux_FMD[0,ic]))
    err_file.write('\nmean uxuy: ')
    err_file.write(str(mean_err_uxuy_FMD[0,ic]))
    err_file.write('\nmean uyuy: ')
    err_file.write(str(mean_err_uyuy_FMD[0,ic]))
    err_file.write('\nmax uxux: ')
    err_file.write(str(max_err_uxux_FMD[0,ic]))
    err_file.write('\nmax uxuy: ')
    err_file.write(str(max_err_uxuy_FMD[0,ic]))
    err_file.write('\nmax uyuy: ')
    err_file.write(str(max_err_uyuy_FMD[0,ic]))
    err_file.write('\n')

err_file.write('Err PINN Reynolds Stresses\n')
for ic in range(len(rec_mode_vec)):
    err_file.write('Mode ')
    err_file.write(str(rec_mode_vec[ic]))
    err_file.write('\nmean uxux: ')
    err_file.write(str(mean_err_uxux[:,ic]))
    err_file.write('\nmax uxux: ')
    err_file.write(str(max_err_uxux[:,ic]))
    err_file.write('\nmean uxuy: ')
    err_file.write(str(mean_err_uxuy[:,ic]))
    err_file.write('\nmax uxuy: ')
    err_file.write(str(max_err_uxuy[:,ic]))
    err_file.write('\nmean uyuy: ')
    err_file.write(str(mean_err_uyuy[:,ic]))
    err_file.write('\nmax uyuy: ')
    err_file.write(str(max_err_uyuy[:,ic]))
    err_file.write('\n')

err_file.close()

for ic in range(len(rec_mode_vec)):
    error_x_tick_labels = ['40','20','10','5','2.5','1.25']
    error_y_tick_labels = ['1E-4','1E-3','1E-2','1E-1','1']

    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3.0)
    plot.subplots_adjust(left=0.2,top=0.95,right=0.97,bottom=0.15)

    line_x = np.array([1.0,55.0])

    # lines to compare with FMD
    #mean_plt_uxux,=axs.plot(line_x,[mean_err_uxux_FMD[0,ic],mean_err_uxux_FMD[mean_err_uxux_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='-',marker='',color='blue',markersize=0,markerfacecolor='blue')
    #max_plt_uxux,=axs.plot(line_x,[max_err_uxux_FMD[0,ic],max_err_uxux_FMD[max_err_uxux_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='--',marker='',color='blue',markersize=0,markerfacecolor='blue')
    #mean_plt_uxuy,=axs.plot(line_x,[mean_err_uxuy_FMD[0,ic],mean_err_uxuy_FMD[mean_err_uxuy_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='-',marker='',color='red',markersize=0,markerfacecolor='red')
    #max_plt_uxuy,=axs.plot(line_x,[max_err_uxuy_FMD[0,ic],max_err_uxuy_FMD[max_err_uxuy_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='--',marker='',color='red',markersize=0,markerfacecolor='red')
    #mean_plt_uyuy,=axs.plot(line_x,[mean_err_uyuy_FMD[0,ic],mean_err_uyuy_FMD[mean_err_uyuy_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='-',marker='',color='green',markersize=0,markerfacecolor='green')
    #max_plt_uyuy,=axs.plot(line_x,[max_err_uyuy_FMD[0,ic],max_err_uyuy_FMD[max_err_uyuy_FMD.shape[0]-1,ic]],linewidth=0.75,linestyle='--',marker='',color='green',markersize=0,markerfacecolor='green')

    mean_plt_uxux,=axs.plot(pts_per_d*0.9,mean_err_uxux[:,ic],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_uxux,=axs.plot(pts_per_d*0.9,max_err_uxux[:,ic],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uxuy,=axs.plot(pts_per_d,mean_err_uxuy[:,ic],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uxuy,=axs.plot(pts_per_d,max_err_uxuy[:,ic],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    mean_plt_uyuy,=axs.plot(pts_per_d*1.1,mean_err_uyuy[:,ic],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt_uyuy,=axs.plot(pts_per_d*1.1,max_err_uyuy[:,ic],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

    axs.set_xscale('log')
    axs.set_xticks(pts_per_d)
    axs.set_xticklabels(error_x_tick_labels,fontsize=8)
    axs.set_yscale('log')
    axs.set_ylim(5E-5,1E1)
    axs.set_xlim(1.0,55.0)
    axs.set_yticks([1E-4,1E-3,1E-2,1E-1,1],labels=error_y_tick_labels,fontsize=8)
    axs.set_ylabel("Error ($\eta$)",fontsize=8)
    axs.legend([mean_plt_uxux,max_plt_uxux,mean_plt_uxuy,max_plt_uxuy,mean_plt_uyuy,max_plt_uyuy,],['Mean $\overline{u\'_xu\'_x}$','Max $\overline{u\'_xu\'_x}$','Mean $\overline{u\'_xu\'_y}$','Max $\overline{u\'_xu\'_y}$','Mean $\overline{u\'_yu\'_y}$','Max $\overline{u\'_yu\'_y}$',],fontsize=8,ncols=2)
    axs.grid('on')
    axs.set_xlabel('$D/\Delta x$',fontsize=8)


    
    plot.savefig(figures_dir+'c'+str(rec_mode_vec[ic])+'/logerr_rec_reynoldsStress_f'+str(rec_mode_vec[ic])+'_error.pdf')
    plot.savefig(figures_dir+'c'+str(rec_mode_vec[ic])+'/logerr_rec_reynoldsStress_f'+str(rec_mode_vec[ic])+'_error.png',dpi=300)
    plot.close(fig)
