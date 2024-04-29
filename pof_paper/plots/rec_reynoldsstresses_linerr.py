
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

#ux_ref = ux_ref+np.reshape(ux,[ux.shape[0],1])
#uy_ref = uy_ref+np.reshape(uy,[uy.shape[0],1])
#p_ref = p_ref+np.reshape(p,[uy.shape[0],1])

ux_ref = np.reshape(ux_ref,[X_grid.shape[0],X_grid.shape[1],ux_ref.shape[1]])
uy_ref = np.reshape(uy_ref,[X_grid.shape[0],X_grid.shape[1],uy_ref.shape[1]])
p_ref = np.reshape(p_ref,[X_grid.shape[0],X_grid.shape[1],p_ref.shape[1]])



L_dft=4082
fs=10.0
t = np.reshape(np.linspace(0,(L_dft-1)/fs,L_dft),[L_dft])
cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

# crop the fluctuating fields to the first 4082 so they are the same as the fourier data
ux_ref = ux_ref[:,:,0:L_dft]
uy_ref = uy_ref[:,:,0:L_dft]
p_ref = p_ref[:,:,0:L_dft]

# load the reference fourier reconstructions
uxux_rec_ref = []
uxuy_rec_ref = []
uyuy_rec_ref = []

rec_mode_vec = [0,1,2]
cases_supersample_factor = [0,2,4,8,16,32]

for c in rec_mode_vec:
    recFile = h5py.File(rec_dir+'rec_fourier_c'+str(c)+'.h5','r')
    ux_rec_ref = np.reshape(np.array(recFile['ux']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    uy_rec_ref = np.reshape(np.array(recFile['uy']),[X_grid.shape[0],X_grid.shape[1],L_dft])
    ux_rec_ref_m = np.reshape(np.mean(ux_rec_ref,axis=2),[ux_rec_ref.shape[0],ux_rec_ref.shape[1],1])
    uy_rec_ref_m = np.reshape(np.mean(uy_rec_ref,axis=2),[uy_rec_ref.shape[0],uy_rec_ref.shape[1],1])
    uxux_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,ux_rec_ref-ux_rec_ref_m),axis=2))
    uxuy_rec_ref.append(np.mean(np.multiply(ux_rec_ref-ux_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uyuy_rec_ref.append(np.mean(np.multiply(uy_rec_ref-uy_rec_ref_m,uy_rec_ref-uy_rec_ref_m),axis=2))
    uxux_rec_ref[c][cylinder_mask]=np.NaN
    uxux_rec_ref[c][cylinder_mask]=np.NaN
    uxux_rec_ref[c][cylinder_mask]=np.NaN


# compute reference reynolds stresses 
uxux_ref = np.mean(np.multiply(ux_ref,ux_ref),axis=2)
uxuy_ref = np.mean(np.multiply(ux_ref,uy_ref),axis=2)
uyuy_ref = np.mean(np.multiply(uy_ref,uy_ref),axis=2)

uxux_ref[cylinder_mask]=np.NaN
uxuy_ref[cylinder_mask]=np.NaN
uyuy_ref[cylinder_mask]=np.NaN


s=0 # needed so no points appear in the reference comparison plotss
for c in rec_mode_vec:

    if True:

        fig = plot.figure(figsize=(7,7))
        plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
        outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        inner = []

        MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
        MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
        MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
        MAX_uxux_err = np.nanmax(np.abs(uxux_ref.ravel()-uxux_rec_ref[c].ravel())/MAX_uxux_ref)
        MAX_uxuy_err = np.nanmax(np.abs(uxuy_ref.ravel()-uxuy_rec_ref[c].ravel())/MAX_uxuy_ref)
        MAX_uyuy_err = np.nanmax(np.abs(uyuy_ref.ravel()-uyuy_rec_ref[c].ravel())/MAX_uyuy_ref)
        
        levels_uxux = 1.1*np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
        levels_uxuy = 1.1*np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
        levels_uyuy = 1.1*np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)
        levels_uxux_err = np.linspace(-MAX_uxux_err,MAX_uxux_err,21)
        levels_uxuy_err = np.linspace(-MAX_uxuy_err,MAX_uxuy_err,21)
        levels_uyuy_err = np.linspace(-MAX_uyuy_err,MAX_uyuy_err,21)

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
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{x}}_{DNS}$',fontsize=5)
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
        fig.add_subplot(ax)
        
        cax=plot.Subplot(fig,inner[0][1])
        cax.set(xmargin=0.5)
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)
        

        ax = plot.Subplot(fig,inner[0][3])
        ux_plot=ax.contourf(X_grid,Y_grid,uxux_rec_ref[c],levels=levels_uxux,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{x}}_{F'+str(c)+'}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][4])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[0][6])
        ux_plot = ax.contourf(X_grid,Y_grid,(uxux_ref-uxux_rec_ref[c])/MAX_uxux_ref,levels=levels_uxux_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{F'+str(c)+'}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][7])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_err,MAX_uxux_err/2,0.0,-MAX_uxux_err/2,-MAX_uxux_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{y}}_{DNS}$',fontsize=5)
        if cases_supersample_factor[s]>1:
            dots = ax.plot(x_downsample,y_downsample,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
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
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{y}}_{F'+str(c)+'}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][4])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[1][6])
        uy_plot = ax.contourf(X_grid,Y_grid,(uxuy_ref-uxuy_rec_ref[c])/MAX_uxuy_ref,levels=levels_uxuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{F'+str(c)+'}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][7])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_err,MAX_uxuy_err/2,0.0,-MAX_uxuy_err/2,-MAX_uxuy_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(9,1.5,'$\overline{u\'_{y}u\'_{y}}_{DNS}$',fontsize=5)
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
        ax.text(9,1.5,'$\overline{u\'_{y}u\'_{y}}_{F'+str(c)+'}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][4])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[2][6])
        p_plot = ax.contourf(X_grid,Y_grid,(uyuy_ref-uyuy_rec_ref[c])/MAX_uyuy_ref,levels=levels_uyuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_ylabel('y/D',fontsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(8,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{F'+str(c)+'}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='w')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][7])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_err,MAX_uyuy_err/2,0.0,-MAX_uyuy_err/2,-MAX_uyuy_err],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        # quadrant 4

        # for now empty ...

        plot.savefig(figures_dir+'linerr_reynoldsStress_contours_ref_recref_c'+str(c)+'.pdf')
        plot.close('all')




# load the reconstuctions and compute the reynolds stresses
uxux_rec = []
uxuy_rec = []
uyuy_rec = []

for s in range(len(cases_supersample_factor)):
    uxux_rec.append([])
    uxuy_rec.append([])
    uyuy_rec.append([])
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
        

for c in rec_mode_vec:

    for s in range(len(cases_supersample_factor)):


        fig = plot.figure(figsize=(7,7))
        plot.subplots_adjust(left=0.06,top=0.97,right=0.97,bottom=0.05)
        outer = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        inner = []

        MAX_uxux_ref = np.nanmax(np.abs(uxux_ref.ravel()))
        MAX_uxuy_ref = np.nanmax(np.abs(uxuy_ref.ravel()))
        MAX_uyuy_ref = np.nanmax(np.abs(uyuy_ref.ravel()))
        MAX_uxux_err = np.nanmax(np.abs(uxux_ref.ravel()-uxux_rec[s][c].ravel())/MAX_uxux_ref)
        MAX_uxuy_err = np.nanmax(np.abs(uxuy_ref.ravel()-uxuy_rec[s][c].ravel())/MAX_uxuy_ref)
        MAX_uyuy_err = np.nanmax(np.abs(uyuy_ref.ravel()-uyuy_rec[s][c].ravel())/MAX_uyuy_ref)
        
        levels_uxux = 1.1*np.linspace(-MAX_uxux_ref,MAX_uxux_ref,21)
        levels_uxuy = 1.1*np.linspace(-MAX_uxuy_ref,MAX_uxuy_ref,21)
        levels_uyuy = 1.1*np.linspace(-MAX_uyuy_ref,MAX_uyuy_ref,21)
        levels_uxux_err = np.linspace(-MAX_uxux_err,MAX_uxux_err,21)
        levels_uxuy_err = np.linspace(-MAX_uxuy_err,MAX_uxuy_err,21)
        levels_uyuy_err = np.linspace(-MAX_uyuy_err,MAX_uyuy_err,21)


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
        ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][4])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[0][6])
        ux_plot = ax.contourf(X_grid,Y_grid,(uxux_ref-uxux_rec[s][c])/MAX_uxux_ref,levels=levels_uxux_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}_{DNS}-\overline{u\'_{x}u\'_{x}}_{PINN}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][7])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_err,MAX_uxux_err/2,0.0,-MAX_uxux_err/2,-MAX_uxux_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{y}}_{DNS}$',fontsize=5)
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
        ax.text(8,1.5,'$\overline{u\'_{x}u\'_{y}}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][4])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[1][6])
        uy_plot = ax.contourf(X_grid,Y_grid,np.abs((uxuy_ref-uxuy_rec[s][c])/MAX_uxuy_ref)+1E-30,levels=levels_uxuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}_{DNS}-\overline{u\'_{x}u\'_{y}}_{PINN}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][7])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_err,MAX_uxuy_err/2,0.0,-MAX_uxuy_err/2,-MAX_uxuy_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][4])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[2][6])
        p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_ref-uyuy_rec[s][c])/MAX_uyuy_ref)+1E-30,levels=levels_uyuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_ylabel('y/D',fontsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}_{DNS}-\overline{u\'_{y}u\'_{y}}_{PINN}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][7])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_err,MAX_uyuy_err/2,0.0,-MAX_uyuy_err/2,-MAX_uyuy_err],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        # quadrant 4

        # for now empty ...

        plot.savefig(figures_dir+'linerr_reynoldsStress_contours_ref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.pdf')
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
        ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}$',fontsize=5)
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
        ax.text(7,1.5,'$\overline{u\'_{x}u\'_{x}}_{PINN,F'+str(c)+'}$',fontsize=5)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][4])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_ref,MAX_uxux_ref/2,0.0,-MAX_uxux_ref/2,-MAX_uxux_ref],format=tkr.FormatStrFormatter('%.3f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[0][6])
        ux_plot = ax.contourf(X_grid,Y_grid,(uxux_rec_ref[c]-uxux_rec[s][c])/MAX_uxux_ref,levels=levels_uxux_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_ylabel('y/D',fontsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{x}}-\overline{u\'_{x}u\'_{x}}_{PINN}}{max(\overline{u\'_{x}u\'_{x}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][7])
        cbar = plot.colorbar(ux_plot,cax,ticks=[MAX_uxux_err,MAX_uxux_err/2,0.0,-MAX_uxux_err/2,-MAX_uxux_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(7,1.5,'$\overline{u\'_{x}u\'_{y}}$',fontsize=5)
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
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][4])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_ref,MAX_uxuy_ref/2,0.0,-MAX_uxuy_ref/2,-MAX_uxuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[1][6])
        uy_plot = ax.contourf(X_grid,Y_grid,(uxuy_rec_ref[c]-uxuy_rec[s][c])/MAX_uxuy_ref,levels=levels_uxuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.set_yticks(np.array([2.0,1.0,0.0,-1.0,-2.0]))
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        t=ax.text(7,1.5,'$|\\frac{\overline{u\'_{x}u\'_{y}}-\overline{u\'_{x}u\'_{y}}_{PINN}}{max(\overline{u\'_{x}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][7])
        cbar = plot.colorbar(uy_plot,cax,ticks=[MAX_uxuy_err,MAX_uxuy_err/2,0.0,-MAX_uxuy_err/2,-MAX_uxuy_err],format=tkr.FormatStrFormatter('%.2f'))
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
        ax.text(7,1.5,'$\overline{u\'_{y}u\'_{y}}$',fontsize=5)
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
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][4])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_ref,MAX_uyuy_ref/2,0.0,-MAX_uyuy_ref/2,-MAX_uyuy_ref],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        ax = plot.Subplot(fig,inner[2][6])
        p_plot = ax.contourf(X_grid,Y_grid,np.abs((uyuy_rec_ref[c]-uyuy_rec[s][c])/MAX_uyuy_ref)+1E-30,levels=levels_uyuy_err,cmap= matplotlib.colormaps['bwr'],extend='both')
        ax.set_aspect('equal')
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_ylabel('y/D',fontsize=5)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_xlabel('x/D',fontsize=5)
        ax.text(7,1.5,'$|\\frac{\overline{u\'_{y}u\'_{y}}-\overline{u\'_{y}u\'_{y}}_{PINN}}{max(\overline{u\'_{y}u\'_{y}}_{DNS})}|$',fontsize=5,color='k')
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][7])
        cbar = plot.colorbar(p_plot,cax,ticks=[MAX_uyuy_err,MAX_uyuy_err/2,0.0,-MAX_uyuy_err/2,-MAX_uyuy_err],format=tkr.FormatStrFormatter('%.2f'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=5)
        fig.add_subplot(cax)

        # quadrant 4

        # for now empty ...

        plot.savefig(figures_dir+'linerr_reynoldsStress_contours_recref_pinn_S'+str(cases_supersample_factor[s])+'_c'+str(c)+'.pdf')
        plot.close('all')