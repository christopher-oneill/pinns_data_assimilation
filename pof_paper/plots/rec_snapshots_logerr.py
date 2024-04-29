
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
vorticity_ref = vorticity(ux_ref,uy_ref,X_grid,Y_grid)

ux_ref[cylinder_mask,:]=np.NaN
uy_ref[cylinder_mask,:]=np.NaN
p_ref[cylinder_mask,:]=np.NaN
vorticity_ref[cylinder_mask,:]=np.NaN

# load the reference fourier reconstructions

rec_mode_vec = [2]
cases_supersample_factor = [0,2,4,8,16,32]


ux_rec_ref = []
uy_rec_ref = []
p_rec_ref = []
vorticity_rec_ref = []

for c in range(len(rec_mode_vec)):
    recFile = h5py.File(rec_dir+'rec_fourier_c'+str(rec_mode_vec[c])+'.h5','r')
    ux_rec_ref.append(np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    uy_rec_ref.append(np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    p_rec_ref.append(np.reshape(np.array(recFile['p']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
    vorticity_temp = vorticity(ux_rec_ref[c],uy_rec_ref[c],X_grid,Y_grid)
    vorticity_rec_ref.append(vorticity_temp)
    ux_rec_ref[c][cylinder_mask,:]=np.NaN
    uy_rec_ref[c][cylinder_mask,:]=np.NaN
    p_rec_ref[c][cylinder_mask,:]=np.NaN
    vorticity_rec_ref[c][cylinder_mask,:]=np.NaN



snapshots = [500,1100,2050,3333]

MAX_ux_ref = np.nanmax(np.abs(ux_ref.ravel()))
MAX_uy_ref = np.nanmax(np.abs(uy_ref.ravel()))
MAX_p_ref = np.nanmax(np.abs(p_ref.ravel()))
MAX_vorticity_ref = 2.0#np.nanmax(np.abs(vorticity_ref.ravel()))

levels_ux = 1.1*np.linspace(-MAX_ux_ref,MAX_ux_ref,21)
levels_uy = 1.1*np.linspace(-MAX_uy_ref,MAX_uy_ref,21)
levels_p = 1.1*np.linspace(-MAX_p_ref,MAX_p_ref,21)
levels_vorticity = np.linspace(-MAX_vorticity_ref,MAX_vorticity_ref,21)

s=0 # needed so no points appear in the reference comparison plotss
for c in range(len(rec_mode_vec)):

    for sn in range(len(snapshots)):

        fig = plot.figure(figsize=(7,7))
        plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
        outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        inner = []
        snap_ind = snapshots[sn]
        print('snapshot: ',snap_ind)

        ux_err = (ux_ref[:,:,snap_ind]-ux_rec_ref[c][:,:,snap_ind])/MAX_ux_ref
        uy_err = (uy_ref[:,:,snap_ind]-uy_rec_ref[c][:,:,snap_ind])/MAX_uy_ref
        p_err = (p_ref[:,:,snap_ind]-p_rec_ref[c][:,:,snap_ind])/MAX_p_ref
        vorticity_err = (vorticity_ref[:,:,snap_ind]-vorticity_rec_ref[c][:,:,snap_ind])
        
        MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
        MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
        MAX_p_err = np.nanmax(np.abs(p_err.ravel()))
        MAX_vorticity_err = np.nanmax(np.abs(vorticity_err.ravel()))

        levels_ux_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_ux_err,MAX_ux_err,21)
        levels_uy_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_uy_err,MAX_uy_err,21)
        levels_p_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_p_err,MAX_p_err,21)
        levels_vorticity_err = np.geomspace(1E-3,1,21)#np.linspace(-MAX_vorticity_err,MAX_vorticity_err,21)

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
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$u_{x,DNS}$',fontsize=5)
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
        fig.add_subplot(ax)
        
        cax=plot.Subplot(fig,inner[0][1])
        cax.set(xmargin=0.5)
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
        

        ax = plot.Subplot(fig,inner[0][3])
        ux_plot=ax.contourf(X_grid,Y_grid,ux_rec_ref[c][:,:,snap_ind],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$u_{x,FMD}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][4])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[0][6])
        ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$\\frac{u_{x,DNS}-u_{x,FMD}}{max(|u_{x_DNS}|)}$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][7])
        # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
        cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)


        # quadrant 2

        inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

        ax = plot.Subplot(fig,inner[1][0])
        uy_plot =ax.contourf(X_grid,Y_grid,uy_ref[:,:,snap_ind],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.yaxis.set_tick_params(labelleft=False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$u_{y,DNS}$',fontsize=5)
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][1])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[1][3])
        uy_plot =ax.contourf(X_grid,Y_grid,uy_rec_ref[c][:,:,snap_ind],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.yaxis.set_tick_params(labelleft=False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$u_{y,FMD}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][4])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[1][6])
        uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        t=ax.text(7,1.5,'$\\frac{u_{y,DNS}-u_{y,FMD}}{max(|u_{y,DNS}|)}$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][7])
        # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
        cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
    


        # quadrant 3

        inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

        ax = plot.Subplot(fig,inner[2][0])
        p_plot =ax.contourf(X_grid,Y_grid,p_ref[:,:,snap_ind],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$p_{DNS}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][1])
        cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_p_ref,-MAX_p_ref/2,0.0,MAX_p_ref/2.0,MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
        
        ax = plot.Subplot(fig,inner[2][3])
        p_plot =ax.contourf(X_grid,Y_grid,p_rec_ref[c][:,:,snap_ind],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$p_{FMD}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][4])
        cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_p_ref,-MAX_p_ref/2,0.0,MAX_p_ref/2,MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[2][6])
        p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_ylabel('y/D',fontsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(7,1.5,'$\\frac{p_{DNS}-p_{FMD}}{max(|p_{DNS}|)}$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][7])
        # [-MAX_p_err,-MAX_p_err/2,0.0,MAX_p_err/2.0,MAX_p_err]
        cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        # quadrant 4

        inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

        ax = plot.Subplot(fig,inner[3][0])
        p_plot =ax.contourf(X_grid,Y_grid,vorticity_ref[:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.text(7,1.5,'$\omega_{DNS}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][1])
        cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
        
        ax = plot.Subplot(fig,inner[3][3])
        p_plot =ax.contourf(X_grid,Y_grid,vorticity_rec_ref[c][:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.text(7,1.5,'$\omega_{FMD}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][4])
        cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[3][6])
        p_plot = ax.contourf(X_grid,Y_grid,np.abs(vorticity_err),levels=levels_vorticity_err,cmap= matplotlib.colormaps['hot_r'],norm=matplotlib.colors.LogNorm(),extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(7,1.5,'$\omega_{DNS}-\omega_{FMD}$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][7])
        #[-MAX_vorticity_err,-MAX_vorticity_err/2,0.0,MAX_vorticity_err/2,MAX_vorticity_err]
        # [-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref]
        cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        plot.savefig(figures_dir+'logerr_snapshot'+str(snap_ind)+'_contours_ref_recref_c'+str(rec_mode_vec[c])+'.pdf')
        plot.close(fig)


exit()


# load the reconstuctions and compute the reynolds stresses
ux_rec = []
uy_rec = []
p_rec = []
vorticity_rec = []

for s in range(len(cases_supersample_factor)):
    ux_rec.append([])
    uy_rec.append([])
    p_rec.append([])
    vorticity_rec.append([])

    for c in range(len(rec_mode_vec)):
        recFile = h5py.File(rec_dir+'rec_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.h5','r')
        ux_rec[s].append(np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        uy_rec[s].append(np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        p_rec[s].append(np.reshape(np.array(recFile['p']),[X_grid.shape[0],X_grid.shape[1],L_dft]))
        vorticity_temp = vorticity(ux_rec[s][c],uy_rec[s][c],X_grid,Y_grid)
        vorticity_rec[s].append(vorticity_temp)
        ux_rec[s][c][cylinder_mask,:]=np.NaN
        uy_rec[s][c][cylinder_mask,:]=np.NaN
        p_rec[s][c][cylinder_mask,:]=np.NaN
        vorticity_rec[s][c][cylinder_mask,:]=np.NaN

        



for c in range(len(rec_mode_vec)):

    for s in range(len(cases_supersample_factor)):


        for sn in range(len(snapshots)):
            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            snap_ind = snapshots[sn]
            print('snapshot: ',snap_ind)

            MAX_ux_rec_ref = np.nanmax(np.abs(ux_rec_ref[c].ravel()))
            MAX_uy_rec_ref = np.nanmax(np.abs(uy_rec_ref[c].ravel()))
            MAX_p_rec_ref = np.nanmax(np.abs(p_rec_ref[c].ravel()))
            MAX_vorticity_rec_ref = np.nanmax(np.abs(vorticity_rec_ref[c].ravel()))
            
            MAX_ux_rec = np.nanmax(np.abs(ux_rec[s][c].ravel()))
            MAX_uy_rec = np.nanmax(np.abs(uy_rec[s][c].ravel()))
            MAX_p_rec = np.nanmax(np.abs(p_rec[s][c].ravel()))
            MAX_vorticity_rec = np.nanmax(np.abs(vorticity_rec[s][c].ravel()))

            levels_ux_rec = np.linspace(-MAX_ux_rec,MAX_ux_rec,21)
            levels_uy_rec = np.linspace(-MAX_uy_rec,MAX_uy_rec,21)
            levels_p_rec = np.linspace(-MAX_p_rec,MAX_p_rec,21)
            levels_vorticity_rec = np.linspace(-MAX_vorticity_rec,MAX_vorticity_rec,21)

            ux_err = (ux_ref[:,:,snap_ind]-ux_rec[s][c][:,:,snap_ind])/MAX_ux_ref
            uy_err = (uy_ref[:,:,snap_ind]-uy_rec[s][c][:,:,snap_ind])/MAX_uy_ref
            p_err = (p_ref[:,:,snap_ind]-p_rec[s][c][:,:,snap_ind])/MAX_p_ref
            vorticity_err = (vorticity_ref[:,:,snap_ind]-vorticity_rec[s][c][:,:,snap_ind])
            
            MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
            MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
            MAX_p_err = np.nanmax(np.abs(p_err.ravel()))
            MAX_vorticity_err = np.nanmax(np.abs(vorticity_err.ravel()))

            levels_ux_err = np.linspace(-MAX_ux_err,MAX_ux_err,21)
            levels_uy_err = np.linspace(-MAX_uy_err,MAX_uy_err,21)
            levels_p_err = np.linspace(-MAX_p_err,MAX_p_err,21)
            levels_vorticity_err = np.linspace(-MAX_vorticity_err,MAX_vorticity_err,21)

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
            ux_plot = ax.contourf(X_grid,Y_grid,ux_ref[:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,DNS}$',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\\frac{u_{x,DNS}-u_{x,PINN}}{max(|u_{x,DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_ref[:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,DNS}$',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            t=ax.text(7,1.5,'$\\frac{u_{y,DNS}-u_{y,PINN}}{max(|u_{y,DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,p_ref[:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{DNS}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,cmap= matplotlib.colormaps['nipy_spectral'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(7,1.5,'$\\frac{p_{DNS}-p_{PINN}}{max(|p_{DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            # [MAX_p_err,MAX_p_err/2,0.0,-MAX_p_err/2,-MAX_p_err]
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            p_plot =ax.contourf(X_grid,Y_grid,vorticity_ref[:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.text(7,1.5,'$\omega_{DNS}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[3][3])
            p_plot =ax.contourf(X_grid,Y_grid,vorticity_rec[s][c][:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.text(7,1.5,'$\omega_{PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[3][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs(vorticity_err),levels=levels_vorticity,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(7,1.5,'$\omega_{DNS}-\omega_{PINN}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][7])
            # [-MAX_vorticity_err,-MAX_vorticity_err/2,0.0,MAX_vorticity_err/2,MAX_vorticity_err]
            # [-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref]
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            plot.savefig(figures_dir+'logerr_snapshot'+str(snapshots[sn])+'_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.pdf')
            plot.close(fig)

            # fourier truncated original data versus the pinn data

            ux_err = (ux_rec_ref[c][:,:,snap_ind]-ux_rec[s][c][:,:,snap_ind])/MAX_ux_ref
            uy_err = (uy_rec_ref[c][:,:,snap_ind]-uy_rec[s][c][:,:,snap_ind])/MAX_uy_ref
            p_err = (p_rec_ref[c][:,:,snap_ind]-p_rec[s][c][:,:,snap_ind])/MAX_p_ref
            vorticity_err = (vorticity_rec_ref[c][:,:,snap_ind]-vorticity_rec[s][c][:,:,snap_ind])
            
            MAX_ux_err = np.nanmax(np.abs(ux_err.ravel()))
            MAX_uy_err = np.nanmax(np.abs(uy_err.ravel()))
            MAX_p_err = np.nanmax(np.abs(p_err.ravel()))
            MAX_vorticity_err = np.nanmax(np.abs(vorticity_err.ravel()))

            levels_ux_err = np.linspace(-MAX_ux_err,MAX_ux_err,21)
            levels_uy_err = np.linspace(-MAX_uy_err,MAX_uy_err,21)
            levels_p_err = np.linspace(-MAX_p_err,MAX_p_err,21)
            levels_vorticity_err = np.linspace(-MAX_vorticity_err,MAX_vorticity_err,21)

            fig = plot.figure(figsize=(7,7))
            plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
            outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
            inner = []

            # quadrant 1

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            # (1,(1,1))
            ax = plot.Subplot(fig,inner[0][0])
            ux_plot = ax.contourf(X_grid,Y_grid,ux_rec_ref[c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,FMD}$',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)
            
            cax=plot.Subplot(fig,inner[0][1])
            cax.set(xmargin=0.5)
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            

            ax = plot.Subplot(fig,inner[0][3])
            ux_plot=ax.contourf(X_grid,Y_grid,ux_rec[s][c][:,:,snapshots[sn]],levels=levels_ux,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{x,PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][4])
            cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_ux_ref,MAX_ux_ref/2,0.0,-MAX_ux_ref/2,-MAX_ux_ref],format=tkr.FormatStrFormatter('%.3f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[0][6])
            ux_plot = ax.contourf(X_grid,Y_grid,np.abs(ux_err),levels=levels_ux_err,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\\frac{u_{x,FMD}-u_{x,PINN}}{max(|u_{x,DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[0][7])
            # [MAX_ux_err,MAX_ux_err/2,0.0,-MAX_ux_err/2,-MAX_ux_err]
            cbar = plot.colorbar(ux_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            # quadrant 2

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.9,0.03,0.07]))

            ax = plot.Subplot(fig,inner[1][0])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_rec_ref[c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,FMD}$',fontsize=5)
            if cases_supersample_factor[s]>1:
                dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][1])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][3])
            uy_plot =ax.contourf(X_grid,Y_grid,uy_rec[s][c][:,:,snapshots[sn]],levels=levels_uy,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$u_{y,PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][4])
            cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uy_ref,MAX_uy_ref/2,0.0,-MAX_uy_ref/2,-MAX_uy_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[1][6])
            uy_plot = ax.contourf(X_grid,Y_grid,np.abs(uy_err),levels=levels_uy_err,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            t=ax.text(7,1.5,'$\\frac{u_{y,FMD}-u_{y,PINN}}{max(|u_{y,DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[1][7])
            # [MAX_uy_err,MAX_uy_err/2,0.0,-MAX_uy_err/2,-MAX_uy_err]
            cbar = plot.colorbar(uy_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
        


            # quadrant 3

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[2][0])
            p_plot =ax.contourf(X_grid,Y_grid,p_rec_ref[c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{FMD}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[2][3])
            p_plot =ax.contourf(X_grid,Y_grid,p_rec[s][c][:,:,snapshots[sn]],levels=levels_p,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.set_ylabel('y/D',fontsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$p_{PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[MAX_p_ref,MAX_p_ref/2,0.0,-MAX_p_ref/2,-MAX_p_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[2][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs(p_err),levels=levels_p_err,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.set_ylabel('y/D',fontsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(7,1.5,'$\\frac{p_{FMD}-p_{PINN}}{max(|p_{DNS}|)}$',fontsize=5,color='k')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[2][7])
            # [MAX_p_err,MAX_p_err/2,0.0,-MAX_p_err/2,-MAX_p_err]
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            # quadrant 4

            inner.append(gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.95,0.03,0.07]))

            ax = plot.Subplot(fig,inner[3][0])
            p_plot =ax.contourf(X_grid,Y_grid,vorticity_rec_ref[c][:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\omega_{FMD}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][1])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)
            
            ax = plot.Subplot(fig,inner[3][3])
            p_plot =ax.contourf(X_grid,Y_grid,vorticity_rec[s][c][:,:,snap_ind],levels=levels_vorticity,cmap= matplotlib.colormaps['bwr'],extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.text(7,1.5,'$\omega_{PINN}$',fontsize=5)
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][4])
            cbar = plot.colorbar(p_plot,cax,ticks=[-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref],format=tkr.FormatStrFormatter('%.2f'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)

            ax = plot.Subplot(fig,inner[3][6])
            p_plot = ax.contourf(X_grid,Y_grid,np.abs(vorticity_err),levels=levels_vorticity,cmap= matplotlib.colormaps['nipy_spectral'],norm=matplotlib.colors.LogNorm(),extend='both')
            ax.set_aspect('equal')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_xlabel('x/D',fontsize=5)
            ax.text(7,1.5,'$\omega_{FMD}-\omega_{PINN}$',fontsize=5,color='w')
            fig.add_subplot(ax)

            cax=plot.Subplot(fig,inner[3][7])
            # [-MAX_vorticity_err,-MAX_vorticity_err/2,0.0,MAX_vorticity_err/2,MAX_vorticity_err]
            # [-MAX_vorticity_ref,-MAX_vorticity_ref/2,0.0,MAX_vorticity_ref/2,MAX_vorticity_ref]
            cbar = plot.colorbar(p_plot,cax,ticks=ticks_err,format=tkr.FormatStrFormatter('%.2e'))
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=5)
            fig.add_subplot(cax)


            plot.savefig(figures_dir+'logerr_snapshot'+str(snapshots[sn])+'_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(rec_mode_vec[c])+'.pdf')
            plot.close(fig)