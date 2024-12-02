
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec

import sys
sys.path.append('F:/projects/pinns_local/code/')
from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center
from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

figures_dir = 'F:/projects/paper_figures/t010/reconstruction/'
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

ux_ref = ux_ref+np.reshape(ux,[ux.shape[0],1])
uy_ref = uy_ref+np.reshape(uy,[uy.shape[0],1])
p_ref = p_ref+np.reshape(p,[uy.shape[0],1])

ux_ref = np.reshape(ux_ref,[X_grid.shape[0],X_grid.shape[1],ux_ref.shape[1]])
uy_ref = np.reshape(uy_ref,[X_grid.shape[0],X_grid.shape[1],uy_ref.shape[1]])
p_ref = np.reshape(p_ref,[X_grid.shape[0],X_grid.shape[1],p_ref.shape[1]])



L_dft=4082
fs=10.0
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<=(0.5*d))

# crop the fluctuating fields to the first 4082 so they are the same as the fourier data
ux_ref = ux_ref[:,:,0:L_dft]
uy_ref = uy_ref[:,:,0:L_dft]
p_ref = p_ref[:,:,0:L_dft]
from pinns_data_assimilation.lib.vortex import vorticity
#vorticity_ref = vorticity(ux_ref,uy_ref,X_grid,Y_grid)

ux_ref[cylinder_mask,:]=np.NaN
uy_ref[cylinder_mask,:]=np.NaN
p_ref[cylinder_mask,:]=np.NaN
#vorticity_ref[cylinder_mask,:]=np.NaN

# load the reference fourier reconstructions

rec_mode_vec = [0,1,2,3,4,5]
cases_supersample_factor = [0,2,4,8,16,32]


ux_FMD = []
uy_FMD = []
p_FMD = []
#vorticity_FMD = []

for c in range(len(rec_mode_vec)):
    recFile = h5py.File(rec_dir+'rec_fourier_c'+str(rec_mode_vec[c])+'.h5','r')
    ux_FMD.append(np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    uy_FMD.append(np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    p_FMD.append(np.reshape(np.array(recFile['p']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    #vorticity_temp = vorticity(ux_FMD[c],uy_FMD[c],X_grid,Y_grid)
    #vorticity_FMD.append(vorticity_temp)
    ux_FMD[c][cylinder_mask,:]=np.NaN
    uy_FMD[c][cylinder_mask,:]=np.NaN
    p_FMD[c][cylinder_mask,:]=np.NaN
    #vorticity_FMD[c][cylinder_mask,:]=np.NaN



snapshots = [1100] # 

MAX_ux_ref = np.nanmax(np.abs(ux_ref.ravel()))
MAX_uy_ref = np.nanmax(np.abs(uy_ref.ravel()))
MAX_p_ref = np.nanmax(np.abs(p_ref.ravel()))
#MAX_vorticity_ref = 2.0#np.nanmax(np.abs(vorticity_ref.ravel()))

levels_ux = 1.1*np.linspace(-MAX_ux_ref,MAX_ux_ref,21)
levels_uy = 1.1*np.linspace(-MAX_uy_ref,MAX_uy_ref,21)
levels_p = 1.1*np.linspace(-MAX_p_ref,MAX_p_ref,21)
#levels_vorticity = np.linspace(-MAX_vorticity_ref,MAX_vorticity_ref,21)

if True:
    s=0 # needed so no points appear in the reference comparison plotss
    for c in range(len(rec_mode_vec)):

        for sn in range(len(snapshots)):

            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []
            snap_ind = snapshots[sn]
            print('snapshot: ',snap_ind)

            ux_err = (ux_ref[:,:,snap_ind]-ux_FMD[c][:,:,snap_ind])/MAX_ux_ref
            uy_err = (uy_ref[:,:,snap_ind]-uy_FMD[c][:,:,snap_ind])/MAX_uy_ref
            p_err = (p_ref[:,:,snap_ind]-p_FMD[c][:,:,snap_ind])/MAX_p_ref
            
            MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
            MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
            MAX_p_err = np.nanmax(np.abs(p_err.ravel()))

            levels_ux_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_ux_err,MAX_ux_err,21)
            levels_uy_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_uy_err,MAX_uy_err,21)
            levels_p_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_p_err,MAX_p_err,21)

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
            ux_plot = ax.contourf(X_grid,Y_grid,ux_ref[:,:,snap_ind],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,DNS}$',fontsize=8)
            ax.text(-1.75,1.4,'(aa)',fontsize=8,color='k')
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,ux_FMD[c][:,:,snap_ind],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,FMD}$',fontsize=8)
            ax.text(-1.75,1.4,'(ab)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$|\\frac{u_{x,DNS}-u_{x,FMD}}{max(|u_{x,DNS}|)}|$',fontsize=8,color='k')
            ax.text(-1.75,1.4,'(ac)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_ref[:,:,snap_ind],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,DNS}$',fontsize=8)
            ax.text(-1.75,1.4,'(ba)',fontsize=8,color='k')
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_FMD[c][:,:,snap_ind],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,FMD}$',fontsize=8)
            ax.text(-1.75,1.4,'(bb)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            t=ax.text(7,1.5,'$|\\frac{u_{y,DNS}-u_{y,FMD}}{max(|u_{y,DNS}|)}|$',fontsize=8,color='k')
            ax.text(-1.75,1.4,'(bc)',fontsize=8,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,p_ref[:,:,snap_ind],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{DNS}$',fontsize=8)
            ax.text(-1.75,1.4,'(ca)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_p_ref,-MAX_p_ref/2,0.0,MAX_p_ref/2.0,MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,p_FMD[c][:,:,snap_ind],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{FMD}$',fontsize=8)
            ax.text(-1.75,1.4,'(cb)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_p_ref,-MAX_p_ref/2,0.0,MAX_p_ref/2,MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_ylabel('y/D',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.set_xlabel('x/D',fontsize=8)
            ax.text(7,1.5,'$|\\frac{p_{DNS}-p_{FMD}}{max(|p_{DNS}|)}|$',fontsize=8,color='k')
            ax.text(-1.75,1.4,'(cc)',fontsize=8,color='k')
            circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
            ax.add_patch(circle)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            # [-MAX_p_err,-MAX_p_err/2,0.0,MAX_p_err/2.0,MAX_p_err]
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=8)
            fig.add_subplot(cax)

            # quadrant 4
            create_directory_if_not_exists(figures_dir+'c'+str(rec_mode_vec[c])+'/')
            plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snap_ind)+'_contours_ref_recref_c'+str(rec_mode_vec[c])+'.pdf')
            plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snap_ind)+'_contours_ref_recref_c'+str(rec_mode_vec[c])+'.png',dpi=300)
            plot.close(fig)


# load the reconstuctions and compute the reynolds stresses
ux_rec = []
uy_rec = []
p_rec = []
#vorticity_rec = []

for s in range(len(cases_supersample_factor)):
    ux_rec.append([])
    uy_rec.append([])
    p_rec.append([])
    #vorticity_rec.append([])

    for c in range(len(rec_mode_vec)):
        recFile = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.h5','r')
        ux_rec[s].append(np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        uy_rec[s].append(np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        p_rec[s].append(np.reshape(np.array(recFile['p']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        #vorticity_temp = vorticity(ux_rec[s][c],uy_rec[s][c],X_grid,Y_grid)
        #vorticity_rec[s].append(vorticity_temp)
        ux_rec[s][c][cylinder_mask,:]=np.NaN
        uy_rec[s][c][cylinder_mask,:]=np.NaN
        p_rec[s][c][cylinder_mask,:]=np.NaN
        #vorticity_rec[s][c][cylinder_mask,:]=np.NaN

        


if True:

    for c in range(len(rec_mode_vec)):

        for s in range(len(cases_supersample_factor)):


            for sn in range(len(snapshots)):
                snap_ind = snapshots[sn]
                print('snapshot: ',snap_ind)

                MAX_ux_FMD = np.nanmax(np.abs(ux_FMD[c].ravel()))
                MAX_uy_FMD = np.nanmax(np.abs(uy_FMD[c].ravel()))
                MAX_p_FMD = np.nanmax(np.abs(p_FMD[c].ravel()))

                
                MAX_ux_rec = np.nanmax(np.abs(ux_rec[s][c].ravel()))
                MAX_uy_rec = np.nanmax(np.abs(uy_rec[s][c].ravel()))
                MAX_p_rec = np.nanmax(np.abs(p_rec[s][c].ravel()))


                levels_ux_rec = np.linspace(-MAX_ux_rec,MAX_ux_rec,21)
                levels_uy_rec = np.linspace(-MAX_uy_rec,MAX_uy_rec,21)
                levels_p_rec = np.linspace(-MAX_p_rec,MAX_p_rec,21)


                ux_err = (ux_ref[:,:,snap_ind]-ux_rec[s][c][:,:,snap_ind])/MAX_ux_ref
                uy_err = (uy_ref[:,:,snap_ind]-uy_rec[s][c][:,:,snap_ind])/MAX_uy_ref
                p_err = (p_ref[:,:,snap_ind]-p_rec[s][c][:,:,snap_ind])/MAX_p_ref

                
                MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
                MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
                MAX_p_err = np.nanmax(np.abs(p_err.ravel()))

                levels_ux_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_ux_err,MAX_ux_err,21)
                levels_uy_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_uy_err,MAX_uy_err,21)
                levels_p_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_p_err,MAX_p_err,21)
                levels_vorticity_err = np.geomspace(1E-3,10,21)#np.linspace(-MAX_vorticity_err,MAX_vorticity_err,21)

                x_ticks = [-2,0,2,4,6,8,10]

                ticks_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])
                ticks_vorticity_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1,3E0,1E1])

                if cases_supersample_factor[s]>0:
                    linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

                    x_downsample = x[linear_downsample_inds]
                    y_downsample = y[linear_downsample_inds]
                    valid_inds = (np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d).ravel()
                    x_downsample = x_downsample[valid_inds]
                    y_downsample = y_downsample[valid_inds]

                fig = plot.figure(figsize=(3.37,8))
                plot.subplots_adjust(left=0.13,top=0.99,right=0.9,bottom=0.06)
                outer = gridspec.GridSpec(3,1,wspace=0.0,hspace=0.05)
                inner = []

                # define the quadrants
                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))
                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))
                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))

                # quadrant 1

                # (1,(1,1))
                ax = plot.Subplot(fig,inner[0][0])
                ux_plot = ax.contourf(X_grid,Y_grid,ux_ref[:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.set_xticks(x_ticks)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(8,1.5,'$u_{x,DNS}$',fontsize=8)
                ax.text(-1.75,1.4,'(aa)',fontsize=8,color='k')
                if cases_supersample_factor[s]>1:
                    dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)
                
                cax=plot.Subplot(fig,inner[0][1])
                cax.set(xmargin=0.5)
                cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)
                

                ax = plot.Subplot(fig,inner[0][3])
                ux_plot=ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(8,1.5,'$u_{x,PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(ab)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[0][4])
                cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[0][6])
                ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(6,1.2,'$|\\frac{u_{x,DNS}-u_{x,PINN}}{max(|u\'_{x,DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(ac)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[0][7])
                # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
                cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)


                # quadrant 2

                

                ax = plot.Subplot(fig,inner[1][0])
                uy_plot =ax.contourf(X_grid,Y_grid,uy_ref[:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelleft=True)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$u_{y,DNS}$',fontsize=8)
                ax.text(-1.75,1.4,'(ba)',fontsize=8,color='k')
                if cases_supersample_factor[s]>1:
                    dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][1])
                cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[1][3])
                uy_plot =ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_xticks(x_ticks)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(8,1.5,'$u_{y,PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(bb)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][4])
                cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[1][6])
                uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
                ax.set_ylabel('y/D',fontsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=True)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                t=ax.text(6,1.2,'$|\\frac{u_{y,DNS}-u_{y,PINN}}{max(|u_{y,DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(bc)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][7])
                # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
                cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)
            


                # quadrant 3

                

                ax = plot.Subplot(fig,inner[2][0])
                p_plot =ax.contourf(X_grid,Y_grid,p_ref[:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$p_{DNS}$',fontsize=8)
                ax.text(-1.75,1.4,'(ca)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][1])
                cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)
                
                ax = plot.Subplot(fig,inner[2][3])
                p_plot =ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$p_{PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(cb)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][4])
                cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[2][6])
                p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,norm=matplotlib.colors.LogNorm(),cmap= matplotlib.colormaps['hot_r'],extend='both')
                ax.set_aspect('equal')
                ax.set_xticks(x_ticks)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_ylabel('y/D',fontsize=8)
                ax.xaxis.set_tick_params(labelsize=8)
                ax.set_xlabel('x/D',fontsize=8)
                ax.text(6,1.2,'$|\\frac{p_{DNS}-p_{PINN}}{max(|p_{DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(cc)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][7])
                # [MAX_p_err,MAX_p_err/2,0.0,-MAX_p_err/2,-MAX_p_err]
                cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snapshots[sn])+'_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.pdf')
                plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snapshots[sn])+'_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.png',dpi=300)
                plot.close(fig)

                # fourier truncated original data versus the pinn data

                ux_err = (ux_FMD[c][:,:,snap_ind]-ux_rec[s][c][:,:,snap_ind])/MAX_ux_ref
                uy_err = (uy_FMD[c][:,:,snap_ind]-uy_rec[s][c][:,:,snap_ind])/MAX_uy_ref
                p_err = (p_FMD[c][:,:,snap_ind]-p_rec[s][c][:,:,snap_ind])/MAX_p_ref

                
                MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
                MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
                MAX_p_err = np.nanmax(np.abs(p_err.ravel()))

                levels_ux_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_ux_err,MAX_ux_err,21)
                levels_uy_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_uy_err,MAX_uy_err,21)
                levels_p_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_p_err,MAX_p_err,21)
                levels_vorticity_err = np.geomspace(1E-3,10,21)#np.linspace(-MAX_vorticity_err,MAX_vorticity_err,21)

                ticks_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1])
                ticks_vorticity_err = np.array([1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1,3E0,1E1])

                fig = plot.figure(figsize=(3.37,8))
                plot.subplots_adjust(left=0.13,top=0.99,right=0.9,bottom=0.06)
                outer = gridspec.GridSpec(3,1,wspace=0.00,hspace=0.05)
                inner = []

                # quadrant 1

                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))
                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))
                inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.00,hspace=0.15,width_ratios=[0.9,0.03,0.07]))

                # (1,(1,1))
                ax = plot.Subplot(fig,inner[0][0])
                ux_plot = ax.contourf(X_grid,Y_grid,ux_FMD[c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$u_{x,FMD}$',fontsize=8)
                ax.text(-1.75,1.4,'(aa)',fontsize=8,color='k')
                if cases_supersample_factor[s]>1:
                    dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)
                
                cax=plot.Subplot(fig,inner[0][1])
                cax.set(xmargin=0.5)
                cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(cax)
                

                ax = plot.Subplot(fig,inner[0][3])
                ux_plot=ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(8,1.5,'$u_{x,PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(ab)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[0][4])
                cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[0][6])
                ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.set_xticks(x_ticks)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.text(6,1.2,'$|\\frac{u_{x,FMD}-u_{x,PINN}}{max(|u_{x,DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(ac)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[0][7])
                # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
                cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)


                # quadrant 2

                ax = plot.Subplot(fig,inner[1][0])
                uy_plot =ax.contourf(X_grid,Y_grid,uy_FMD[c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
                ax.yaxis.set_tick_params(labelleft=True)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_ylabel('y/D',fontsize=8)
                ax.text(8,1.5,'$u_{y,FMD}$',fontsize=8)
                ax.text(-1.75,1.4,'(ba)',fontsize=8,color='k')
                if cases_supersample_factor[s]>1:
                    dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][1])
                cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[1][3])
                uy_plot =ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_xticks(x_ticks)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_ylabel('y/D',fontsize=8)
                ax.text(8,1.5,'$u_{y,PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(bb)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][4])
                cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[1][6])
                uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
                ax.set_aspect('equal')
                #ax.set_yticks(np.array([2.0,0.0,-1.0,-2.0]))
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_xticks(x_ticks)
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                t=ax.text(6,1.2,'$|\\frac{u_{y,FMD}-u_{y,PINN}}{max(|u_{y,DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(bc)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[1][7])
                # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
                cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)
            


                # quadrant 3


                ax = plot.Subplot(fig,inner[2][0])
                p_plot =ax.contourf(X_grid,Y_grid,p_FMD[c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$p_{FMD}$',fontsize=8)
                ax.text(-1.75,1.4,'(ca)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][1])
                cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)
                
                ax = plot.Subplot(fig,inner[2][3])
                p_plot =ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.set_ylabel('y/D',fontsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks(x_ticks)
                ax.text(8,1.5,'$p_{PINN}$',fontsize=8)
                ax.text(-1.75,1.4,'(cb)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][4])
                cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[2][6])
                p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
                ax.set_aspect('equal')
                ax.yaxis.set_tick_params(labelsize=8)
                ax.set_ylabel('y/D',fontsize=8)
                ax.xaxis.set_tick_params(labelsize=8)
                ax.set_xticks(x_ticks)
                ax.set_xlabel('x/D',fontsize=8)
                ax.text(6,1.2,'$|\\frac{p_{FMD}-p_{PINN}}{max(|p_{DNS}|)}|$',fontsize=8,color='k')
                ax.text(-1.75,1.4,'(cc)',fontsize=8,color='k')
                circle = plot.Circle((0,0),0.5,color='k',fill=False,zorder=2)
                ax.add_patch(circle)
                fig.add_subplot(ax)

                cax=plot.Subplot(fig,inner[2][7])
                # [MAX_p_err,MAX_p_err/2,0.0,-MAX_p_err/2,-MAX_p_err]
                cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.0e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snapshots[sn])+'_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.pdf')
                plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_snapshot'+str(snapshots[sn])+'_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.png',dpi=300)
                plot.close(fig)



if False:
    # plot comparing the FMD error with the PINN error
    sn=0

    s=4
    c=2
    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>6.5,y_downsample1>1))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    s=5
    c=2
    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample2>6.5,y_downsample2>1))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]

    s=4
    c=5


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
    
    # combined figure of S16,S32

    # dual log scale error plots
    fig = plot.figure(figsize=(3.37,8.0))
    plot.subplots_adjust(left=0.1,top=0.99,right=0.88,bottom=0.05)
    outer = gridspec.GridSpec(9,1,wspace=0.1,hspace=0.1)
    inner = []

    MAX_plot_ux_ref = np.nanmax(np.abs(ux_ref.ravel()))
    levels_ux_ref = np.linspace(-MAX_plot_ux_ref,MAX_plot_ux_ref,21)
    MAX_plot_uy_ref = np.nanmax(np.abs(uy_ref.ravel()))
    levels_uy_ref = np.linspace(-MAX_plot_uy_ref,MAX_plot_uy_ref,21)
    MAX_plot_p_ref = np.nanmax(np.abs(p_ref.ravel()))
    levels_p_ref = np.linspace(-MAX_plot_p_ref,MAX_plot_p_ref,21)

    x_ticks = np.array([-2,0,2,4,6,8,10])
    
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_ref[:,:,snapshots[sn]],levels=levels_ux_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{x,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux_ref,MAX_plot_ux_ref/2,0.0,-MAX_plot_ux_ref/2,-MAX_plot_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])

    e_plot = (ux_rec[s][c][:,:,snapshots[sn]]-ux_ref[:,:,snapshots[sn]])/MAX_plot_ux_ref
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
    ax.text(6.5,1.3,'$\\frac{u_{x,PINN}-u_{x,DNS}}{max(|u_{x,DNS}|)}$',fontsize=8,color='k')
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

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])
    s=5
    e_plot = (ux_rec[s][c][:,:,snapshots[sn]]-ux_ref[:,:,snapshots[sn]])/MAX_plot_ux_ref
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
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{u_{x,PINN}-u_{x,DNS}}{max(|u_{x,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
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
    cax=plot.Subplot(fig,inner[2][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)



    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uy_ref[:,:,snapshots[sn]],levels=levels_uy_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{y,DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(d)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[3][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_uy_ref,MAX_plot_uy_ref/2,0.0,-MAX_plot_uy_ref/2,-MAX_plot_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    s=4
    e_plot = (uy_rec[s][c][:,:,snapshots[sn]]-uy_ref[:,:,snapshots[sn]])/MAX_plot_uy_ref
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
    ax.text(6.5,1.3,'$\\frac{u_{y,PINN}-u_{y,DNS}}{max(|u_{y,DNS}|)}$',fontsize=8,color='k')
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

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[5],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    s=5
    e_plot = (uy_rec[s][c][:,:,snapshots[sn]]-uy_ref[:,:,snapshots[sn]])/MAX_plot_uy_ref
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
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{u_{y,PINN}-u_{y,DNS}}{max(|u_{y,DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
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
    cax=plot.Subplot(fig,inner[5][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
  
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[6],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[6][0])
    ux_plot = ax.contourf(X_grid,Y_grid,p_ref[:,:,snapshots[sn]],levels=levels_p_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$p_{DNS}$',fontsize=8)
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_p_ref,MAX_plot_p_ref/2,0.0,-MAX_plot_p_ref/2,-MAX_plot_p_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[7],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])
    s=4
    e_plot = (p_rec[s][c][:,:,snapshots[sn]]-p_ref[:,:,snapshots[sn]])/MAX_plot_p_ref
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
    ax.text(6.5,1.3,'$\\frac{p_{PINN}-p_{DNS}}{max(|p_{DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(h)',fontsize=8)
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
    cax=plot.Subplot(fig,inner[7][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[8],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])
    s=5
    e_plot = (p_rec[s][c][:,:,snapshots[sn]]-p_ref[:,:,snapshots[sn]])/MAX_plot_p_ref
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=0)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\\frac{p_{PINN}-p_{DNS}}{max(|p_{DNS}|)}$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[8][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    plot.savefig(figures_dir+'c5/logerr_snapshot'+str(snapshots[sn])+'_contours_ref_pinn_S16S32_c5_condensed.pdf')
    plot.savefig(figures_dir+'c5/logerr_snapshot'+str(snapshots[sn])+'_contours_ref_pinn_S16S32_c5_condensed.png',dpi=300)
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

mean_err_ux = []
mean_err_uy = []
mean_err_p = []
p95_err_ux = []
p95_err_uy = []
p95_err_p = []
max_err_ux = []
max_err_uy = []
max_err_p = []

mean_err_ux_FMD = []
mean_err_uy_FMD = []
mean_err_p_FMD = []
max_err_ux_FMD = []
max_err_uy_FMD = []
max_err_p_FMD = []

mean_err_ux_PINN_FMD = []
mean_err_uy_PINN_FMD = []
mean_err_p_PINN_FMD = []
max_err_ux_PINN_FMD = []
max_err_uy_PINN_FMD = []
max_err_p_PINN_FMD = []

for s in range(len(cases_supersample_factor)):
    mean_err_ux.append([])
    mean_err_uy.append([])
    mean_err_p.append([])
    p95_err_ux.append([])
    p95_err_uy.append([])
    p95_err_p.append([])
    max_err_ux.append([])
    max_err_uy.append([])
    max_err_p.append([])

    mean_err_ux_FMD.append([])
    mean_err_uy_FMD.append([])
    mean_err_p_FMD.append([])
    max_err_ux_FMD.append([])
    max_err_uy_FMD.append([])
    max_err_p_FMD.append([])

    mean_err_ux_PINN_FMD.append([])
    mean_err_uy_PINN_FMD.append([])
    mean_err_p_PINN_FMD.append([])
    max_err_ux_PINN_FMD.append([])
    max_err_uy_PINN_FMD.append([])
    max_err_p_PINN_FMD.append([])

    for c in range(len(rec_mode_vec)):
        err_ux = (np.abs(ux_rec[s][c]-ux_ref)/MAX_ux_ref).ravel()
        err_uy = (np.abs(uy_rec[s][c]-uy_ref)/MAX_uy_ref).ravel()
        err_p = (np.abs(p_rec[s][c]-p_ref)/MAX_p_ref).ravel()

        mean_err_ux[s].append(np.nanmean(err_ux))
        mean_err_uy[s].append(np.nanmean(err_uy))
        mean_err_p[s].append(np.nanmean(err_p))

        p95_err_ux[s].append(np.nanpercentile(err_ux,95))
        p95_err_uy[s].append(np.nanpercentile(err_uy,95))
        p95_err_p[s].append(np.nanpercentile(err_p,95))

        max_err_ux[s].append(np.nanmax(err_ux))
        max_err_uy[s].append(np.nanmax(err_uy))
        max_err_p[s].append(np.nanmax(err_p))

        err_ux_FMD = (np.abs(ux_FMD[c]-ux_ref)/MAX_ux_ref).ravel()
        err_uy_FMD = (np.abs(uy_FMD[c]-uy_ref)/MAX_uy_ref).ravel()
        err_p_FMD = (np.abs(p_FMD[c]-p_ref)/MAX_p_ref).ravel()

        mean_err_ux_FMD[s].append(np.nanmean(err_ux_FMD))
        mean_err_uy_FMD[s].append(np.nanmean(err_uy_FMD))
        mean_err_p_FMD[s].append(np.nanmean(err_p_FMD))
        max_err_ux_FMD[s].append(np.nanmax(err_ux_FMD))
        max_err_uy_FMD[s].append(np.nanmax(err_uy_FMD))
        max_err_p_FMD[s].append(np.nanmax(err_p_FMD))

        err_ux_PINN_FMD = (np.abs(ux_rec[s][c]-ux_FMD[c])/MAX_ux_ref).ravel()
        err_uy_PINN_FMD = (np.abs(uy_rec[s][c]-uy_FMD[c])/MAX_uy_ref).ravel()
        err_p_PINN_FMD = (np.abs(p_rec[s][c]-p_FMD[c])/MAX_p_ref).ravel()

        mean_err_ux_PINN_FMD[s].append(np.nanmean(err_ux_PINN_FMD))
        mean_err_uy_PINN_FMD[s].append(np.nanmean(err_uy_PINN_FMD))
        mean_err_p_PINN_FMD[s].append(np.nanmean(err_p_PINN_FMD))
        max_err_ux_PINN_FMD[s].append(np.nanmax(err_ux_PINN_FMD))
        max_err_uy_PINN_FMD[s].append(np.nanmax(err_uy_PINN_FMD))
        max_err_p_PINN_FMD[s].append(np.nanmax(err_p_PINN_FMD))

# error percent plot
pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,2]

mean_err_ux = np.array(mean_err_ux)
mean_err_uy = np.array(mean_err_uy)
mean_err_p = np.array(mean_err_p)
p95_err_ux = np.array(p95_err_ux)
p95_err_uy = np.array(p95_err_uy)
p95_err_p = np.array(p95_err_p)
max_err_ux = np.array(max_err_ux)
max_err_uy = np.array(max_err_uy)
max_err_p = np.array(max_err_p)

mean_err_ux_FMD = np.array(mean_err_ux_FMD)
mean_err_uy_FMD = np.array(mean_err_uy_FMD)
mean_err_p_FMD = np.array(mean_err_p_FMD)
max_err_ux_FMD = np.array(max_err_ux_FMD)
max_err_uy_FMD = np.array(max_err_uy_FMD)
max_err_p_FMD = np.array(max_err_p_FMD)

mean_err_ux_PINN_FMD = np.array(mean_err_ux_PINN_FMD)
mean_err_uy_PINN_FMD = np.array(mean_err_uy_PINN_FMD)
mean_err_p_PINN_FMD = np.array(mean_err_p_PINN_FMD)
max_err_ux_PINN_FMD = np.array(max_err_ux_PINN_FMD)
max_err_uy_PINN_FMD = np.array(max_err_uy_PINN_FMD)
max_err_p_PINN_FMD = np.array(max_err_p_PINN_FMD)



for c in [5]:

    print('FMD truncation ',c)
    print('error relative to DNS: ')
    print('mean ux: ',mean_err_ux[:,c])
    print('max ux: ',max_err_ux[:,c])
    print('mean uy: ',mean_err_uy[:,c])
    print('max uy: ',max_err_uy[:,c])
    print('mean p: ',mean_err_p[:,c])
    print('max p: ',max_err_p[:,c])
    print('error relative to FMD:')
    print('mean ux: ',mean_err_ux_PINN_FMD[:,c])
    print('max ux: ',max_err_ux_PINN_FMD[:,c])
    print('mean uy: ',mean_err_uy_PINN_FMD[:,c])
    print('max uy: ',max_err_uy_PINN_FMD[:,c])
    print('mean p: ',mean_err_p_PINN_FMD[:,c])
    print('max p: ',max_err_p_PINN_FMD[:,c])
    
if True:
    for c in range(len(rec_mode_vec)):
    
        error_x_tick_labels = ['40','20','10','5','2.5','1.25']
        error_y_tick_labels = ['1E-4','1E-3','0.01','0.1','1']
        error_y_ticks = [1E-4,1E-3,1E-2,1E-1,1]

        fig = plot.figure(figsize=(3.37,3.0))
        plot.subplots_adjust(left=0.2,top=0.99,right=0.97,bottom=0.15)
        outer = gridspec.GridSpec(1,1,wspace=0.1,hspace=0.1,)

        ax = plot.Subplot(fig,outer[0])

        #line_x = np.array([1.0,55.0])
        #mean_plt_uxux,=ax.plot(line_x,[mean_err_ux_FMD[0,c],mean_err_ux_FMD[mean_err_ux_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='-',marker='',color='blue',markersize=3,markerfacecolor='blue')
        #max_plt_uxux,=ax.plot(line_x,[max_err_ux_FMD[0,c],max_err_ux_FMD[max_err_ux_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='--',marker='',color='blue',markersize=3,markerfacecolor='blue')
        #mean_plt_uxuy,=ax.plot(line_x,[mean_err_uy_FMD[0,c],mean_err_uy_FMD[mean_err_uy_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='-',marker='',color='red',markersize=3,markerfacecolor='red')
        #max_plt_uxuy,=ax.plot(line_x,[max_err_uy_FMD[0,c],max_err_uy_FMD[max_err_uy_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='--',marker='',color='red',markersize=3,markerfacecolor='red')
        #mean_plt_uyuy,=ax.plot(line_x,[mean_err_p_FMD[0,c],mean_err_p_FMD[mean_err_p_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='-',marker='',color='green',markersize=3,markerfacecolor='green')
        #max_plt_uyuy,=ax.plot(line_x,[max_err_p_FMD[0,c],max_err_p_FMD[max_err_p_FMD.shape[0]-1,c]],linewidth=0.75,linestyle='--',marker='',color='green',markersize=3,markerfacecolor='green')

        mean_plt_uxux,=ax.plot(pts_per_d*0.9,mean_err_ux[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
        max_plt_uxux,=ax.plot(pts_per_d*0.9,max_err_ux[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
        mean_plt_uxuy,=ax.plot(pts_per_d*1.0,mean_err_uy[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
        max_plt_uxuy,=ax.plot(pts_per_d*1.0,max_err_uy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
        mean_plt_uyuy,=ax.plot(pts_per_d*1.1,mean_err_p[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
        max_plt_uyuy,=ax.plot(pts_per_d*1.1,max_err_p[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

        ax.set_xscale('log')
        ax.set_xticks(pts_per_d)
        ax.set_xticklabels(error_x_tick_labels,fontsize=8)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_yscale('log')
        #ax.text(1.1,5E0,'(a)',fontsize=8)
        ax.set_ylim(1E-4,1E0)
        ax.set_xlim(1.0,55.0)
        ax.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
        ax.set_ylabel("Relative Error",fontsize=8)
        ax.set_xlabel("$D/\Delta x$",fontsize=8)
        ax.legend([mean_plt_uxux,max_plt_uxux,mean_plt_uxuy,max_plt_uxuy,mean_plt_uyuy,max_plt_uyuy,],['Mean $u_x$','Max $u_x$','Mean $u_y$','Max $u_y$','Mean $p$','Max $p$',],fontsize=8,ncols=2)
        ax.grid('on')
        
        fig.add_subplot(ax)

        #fig.tight_layout()
        plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_ref_pinn_instantaneous_c'+str(rec_mode_vec[c])+'_error.pdf')
        plot.savefig(figures_dir+'c'+str(rec_mode_vec[c])+'/logerr_ref_pinn_instantaneous_c'+str(rec_mode_vec[c])+'_error.png',dpi=300)
        plot.close(fig)



if True:
    sn=0
    s=3
    c=2
    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample1 = x[linear_downsample_inds]
        y_downsample1 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample1,2.0)+np.power(y_downsample1,2.0),0.5)>0.5*d).ravel()

        x_downsample1 = x_downsample1[valid_inds]
        y_downsample1 = y_downsample1[valid_inds]

    text_corner_mask = (np.multiply(x_downsample1>6.5,y_downsample1>1))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1>6.3,y_downsample1<-1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample1<-1,y_downsample1>1.5))<1
    x_downsample1 = x_downsample1[text_corner_mask]
    y_downsample1 = y_downsample1[text_corner_mask]

    s=5
    c=2
    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample2 = x[linear_downsample_inds]
        y_downsample2 = y[linear_downsample_inds]

        valid_inds = (np.power(np.power(x_downsample2,2.0)+np.power(y_downsample2,2.0),0.5)>0.5*d).ravel()

        x_downsample2 = x_downsample2[valid_inds]
        y_downsample2 = y_downsample2[valid_inds]

    text_corner_mask = (np.multiply(x_downsample2>6.5,y_downsample2>1))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2>6.3,y_downsample2<-1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]
    text_corner_mask = (np.multiply(x_downsample2<-1,y_downsample2>1.5))<1
    x_downsample2 = x_downsample2[text_corner_mask]
    y_downsample2 = y_downsample2[text_corner_mask]

    s=4
    c=5

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

     # dual log scale error plots
    fig = plot.figure(figsize=(5.85,8.0))
    plot.subplots_adjust(left=0.06,top=0.99,right=0.93,bottom=0.05)
    outer = gridspec.GridSpec(2,2,wspace=0.35,hspace=0.05)
    mid = []
    mid.append(gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=outer[0],wspace=0.05,hspace=0.1,height_ratios=[1,1,1,1,1]))
    inner = []

    MAX_plot_ux_ref = np.nanmax(np.abs(ux_ref.ravel()))
    levels_ux_ref = np.linspace(-MAX_plot_ux_ref,MAX_plot_ux_ref,21)
    MAX_plot_uy_ref = np.nanmax(np.abs(uy_ref.ravel()))
    levels_uy_ref = np.linspace(-MAX_plot_uy_ref,MAX_plot_uy_ref,21)
    MAX_plot_p_ref = np.nanmax(np.abs(p_ref.ravel()))
    levels_p_ref = np.linspace(-MAX_plot_p_ref,MAX_plot_p_ref,21)

    x_ticks = np.array([-2,0,2,4,6,8,10])
    
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[0][0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_ref[:,:,snapshots[sn]],levels=levels_ux_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{x,DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(a)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[0][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux_ref,MAX_plot_ux_ref/2,0.0,-MAX_plot_ux_ref/2,-MAX_plot_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    
    s=3
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[0][1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[1][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{x,PINN}}$',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(b)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux_ref,MAX_plot_ux_ref/2,0.0,-MAX_plot_ux_ref/2,-MAX_plot_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    s=5
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[0][2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[2][0])
    ux_plot = ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{x,PINN}}$',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(c)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[2][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_ux_ref,MAX_plot_ux_ref/2,0.0,-MAX_plot_ux_ref/2,-MAX_plot_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
    
    
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[0][3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[3][0])

    s=3
    e_plot = (ux_rec[s][c][:,:,snapshots[sn]]-ux_ref[:,:,snapshots[sn]])/MAX_plot_ux_ref
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
    ax.text(6.5,1.3,'$\eta(u_{\mathrm{x,PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(d)',fontsize=8)
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
    cax=plot.Subplot(fig,inner[3][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[0][4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[4][0])
    s=5
    e_plot = (ux_rec[s][c][:,:,snapshots[sn]]-ux_ref[:,:,snapshots[sn]])/MAX_plot_ux_ref
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
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\eta(u_{\mathrm{x,PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(e)',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
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

    
    mid.append(gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=outer[1],wspace=0.05,hspace=0.1,height_ratios=[1,1,1,1,1]))

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[1][0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[5][0])
    ux_plot = ax.contourf(X_grid,Y_grid,uy_ref[:,:,snapshots[sn]],levels=levels_uy_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{y,DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(f)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[5][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_uy_ref,MAX_plot_uy_ref/2,0.0,-MAX_plot_uy_ref/2,-MAX_plot_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[1][1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[6][0])
    s=3 # S=8, D/dx = 5
    ux_plot = ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{y,PINN}}$',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(g)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[6][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_uy_ref,MAX_plot_uy_ref/2,0.0,-MAX_plot_uy_ref/2,-MAX_plot_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[1][2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[7][0])
    s=5 # S=32
    ux_plot = ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$u_{\mathrm{y,PINN}}$',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(h)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[7][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_uy_ref,MAX_plot_uy_ref/2,0.0,-MAX_plot_uy_ref/2,-MAX_plot_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)



    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[1][3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[8][0])
    s=3
    e_plot = (uy_rec[s][c][:,:,snapshots[sn]]-uy_ref[:,:,snapshots[sn]])/MAX_plot_uy_ref
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
    ax.text(6.5,1.3,'$\eta(u_{\mathrm{y,PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(i)',fontsize=8)
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
    cax=plot.Subplot(fig,inner[8][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[1][4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[9][0])
    s=5
    e_plot = (uy_rec[s][c][:,:,snapshots[sn]]-uy_ref[:,:,snapshots[sn]])/MAX_plot_uy_ref
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=0)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    if cases_supersample_factor[s]>1:
        dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\eta(u_{\mathrm{y,PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(j)',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
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
    cax=plot.Subplot(fig,inner[9][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)
  
    mid.append(gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=outer[2],wspace=0.05,hspace=0.1,height_ratios=[1,1,1,1,1]))

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[2][0],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[10][0])
    ux_plot = ax.contourf(X_grid,Y_grid,p_ref[:,:,snapshots[sn]],levels=levels_p_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$p_{\mathrm{DNS}}$',fontsize=8)
    ax.text(-1.85,1.45,'(k)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[10][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_p_ref,MAX_plot_p_ref/2,0.0,-MAX_plot_p_ref/2,-MAX_plot_p_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[2][1],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[11][0])
    s=3 # S=8, D/dx = 5
    ux_plot = ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$p_{\mathrm{PINN}}$',fontsize=8)
    ax.text(6.5,-1.8,'$D/\Delta x = 5$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(l)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[11][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_p_ref,MAX_plot_p_ref/2,0.0,-MAX_plot_p_ref/2,-MAX_plot_p_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[2][2],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[12][0])
    s=5 # s=32, D/dx = 1.25
    ux_plot = ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p_ref,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(8,1.4,'$p_{\mathrm{PINN}}$',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(m)',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[12][1])
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_plot_p_ref,MAX_plot_p_ref/2,0.0,-MAX_plot_p_ref/2,-MAX_plot_p_ref],format=tkr.FormatStrFormatter('%.2f'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
            
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[2][3],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[13][0])
    s=3
    e_plot = (p_rec[s][c][:,:,snapshots[sn]]-p_ref[:,:,snapshots[sn]])/MAX_plot_p_ref
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
    ax.text(6.5,1.3,'$\eta(p_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(n)',fontsize=8)
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
    cax=plot.Subplot(fig,inner[13][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[2][4],wspace=0.05,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[14][0])
    s=5
    e_plot = (p_rec[s][c][:,:,snapshots[sn]]-p_ref[:,:,snapshots[sn]])/MAX_plot_p_ref
    e_plot_p =e_plot+1E-30
    e_plot_p[e_plot_p<=0]=np.NaN
    e_plot_n = e_plot
    e_plot_n[e_plot_n>0]=np.NaN
    e_plot_n = np.abs(e_plot_n)

    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_p,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap1,extend='both')
    ux_plot = ax.contourf(X_grid,Y_grid,e_plot_n,levels=levels_mx,norm=matplotlib.colors.LogNorm(),cmap=cmap2,extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8,labelpad=0)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.xaxis.set_tick_params(labelsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    #if cases_supersample_factor[s]>1:
    #    dots = ax.plot(x_downsample2,y_downsample2,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
    ax.text(6.5,1.3,'$\eta(p_{\mathrm{PINN}})$',fontsize=8,color='k')
    ax.text(-1.85,1.45,'(o)',fontsize=8)
    ax.text(6,-1.8,'$D/\Delta x = 1.25$',fontsize=8,color='k')
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[14][1])
    cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
    cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
    fig.add_subplot(cax)

    # error

    error_x_tick_labels = ['40','20','10','5','2.5','1.25']
    error_y_tick_labels = ['1E-4','1E-3','0.01','0.1','1']
    error_y_ticks = [1E-4,1E-3,1E-2,1E-1,1]

    mid.append(gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[3],wspace=0.05,hspace=0.1,height_ratios=[2,3,]))

    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid[3][1],wspace=0.05,hspace=0.1,width_ratios=[0.15,0.85]))
    ax = plot.Subplot(fig,inner[15][1])

    c=5 # use 6 modes

    mean_plt_uxux,=ax.plot(pts_per_d*0.9,mean_err_ux[:,c],linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    max_plt_uxux,=ax.plot(pts_per_d*0.9,max_err_ux[:,c],linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt_uxuy,=ax.plot(pts_per_d*1.0,mean_err_uy[:,c],linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt_uxuy,=ax.plot(pts_per_d*1.0,max_err_uy[:,c],linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    mean_plt_uyuy,=ax.plot(pts_per_d*1.1,mean_err_p[:,c],linewidth=0,marker='o',color='green',markersize=3,markerfacecolor='green')
    max_plt_uyuy,=ax.plot(pts_per_d*1.1,max_err_p[:,c],linewidth=0,marker='v',color='green',markersize=3,markerfacecolor='green')

    ax.set_xscale('log')
    ax.set_xticks(pts_per_d)
    ax.set_xticklabels(error_x_tick_labels,fontsize=8)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.set_yscale('log')
    #ax.text(1.1,5E0,'(a)',fontsize=8)
    ax.set_ylim(1E-4,1E0)
    ax.set_xlim(1.0,55.0)
    ax.set_yticks(error_y_ticks,labels=error_y_tick_labels,fontsize=8)
    ax.set_ylabel("Error ($\eta$)",fontsize=8)
    ax.text(1.5,0.2,'(p)',fontsize=8)
    ax.set_xlabel("$D/\Delta x$",fontsize=8)
    ax.legend([mean_plt_uxux,max_plt_uxux,mean_plt_uxuy,max_plt_uxuy,mean_plt_uyuy,max_plt_uyuy,],['Mean $u_x$','Max $u_x$','Mean $u_y$','Max $u_y$','Mean $p$','Max $p$',],fontsize=8,ncols=2,bbox_to_anchor=(1.0, 1.4))
    ax.grid('on')
        
    fig.add_subplot(ax)


    plot.savefig(figures_dir+'snapshot'+str(snapshots[sn])+'_combined.pdf')
    plot.savefig(figures_dir+'snapshot'+str(snapshots[sn])+'_combined.png',dpi=600)
    plot.close(fig)