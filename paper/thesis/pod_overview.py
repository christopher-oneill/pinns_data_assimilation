

import numpy as np
import sys
import h5py
import platform
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

node_name = platform.node()

LOCAL_NODE = 'DESKTOP-L3FA8HC'
if node_name==LOCAL_NODE:
    import matplotlib.pyplot as plot
    import matplotlib.colors as mplcolors
    useGPU=False    
    HOMEDIR = 'F:/projects/pinns_narval/sync/'
    sys.path.append('F:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists
from pinns_data_assimilation.lib.dft import dft

# read the data

base_dir = HOMEDIR+'data/mazi_fixed_grid/'
time_data_dir = 'F:/projects/fixed_cylinder/grid/data/'
figures_dir = 'F:/projects/paper_figures/t010_f2/thesis/'


configFile = h5py.File(base_dir+'configuration.mat','r')
POD_file = h5py.File(base_dir+'POD_data_m16.mat','r')

fs = 10.0


Phi = np.array(POD_file['Phi'])
Ak = np.array(POD_file['Ak'])
Ak = Ak[0:4082,:] # truncate to the same length as the fourier example
t = np.arange(0,Ak.shape[0])/fs

x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

phi_xr = np.reshape(Phi[0:x.shape[0],:],[X_grid.shape[0],X_grid.shape[1],Phi.shape[1]])
phi_yr = np.reshape(Phi[x.shape[0]:2*x.shape[0],:],[X_grid.shape[0],X_grid.shape[1],Phi.shape[1]])

print(Ak.shape)

Ak_hat,f_hat = dft(np.transpose(Ak),fs=fs)
Ak_hat = np.abs(Ak_hat.transpose())/Ak.shape[0]
Ak_hat = 2*Ak_hat[0:Ak.shape[0]//2,:]#+Ak_hat[Ak.shape[1]:-1:Ak.shape[1]//2]
f_hat = f_hat[0:Ak.shape[0]//2]

Ak_hat[0,:]=np.NaN # remove the zero frequency because it is approximately meps
print(Ak_hat.shape)
print(f_hat.shape)

if False:
    plot.figure(1)
    plot.subplot(2,1,1)
    plot.xlim([0,20])
    plot.plot(t,Ak[:,0])
    plot.plot(t,Ak[:,1],'r')
    plot.subplot(2,1,2)
    plot.plot(f_hat,Ak_hat[:,0])
    plot.plot(f_hat,Ak_hat[:,1],'r')
    plot.yscale('log')
    plot.show()
    exit()


def spectrumplot_line(fig,mid,plot_tuple):
    Ak_hat1,Ak_hat2,label1,label2,subplot_label = plot_tuple

    MAX_Ak_hat = np.nanmax([np.nanmax(Ak_hat1),np.nanmax(Ak_hat2)])
    MIN_Ak_hat = np.nanmin([np.nanmin(Ak_hat1),np.nanmin(Ak_hat2)])

    frequency_plot_x_ticks = [0,1,2,3,4,5]
    frequency_plot_y_ticks = np.power(10,np.arange(np.floor(np.log10(MIN_Ak_hat))-1,np.ceil(np.log10(MAX_Ak_hat)))+1)

    # compute the DFT of the two signals
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid,wspace=0.0,hspace=0.1,width_ratios=[0.2,0.8]))
    ax = plot.Subplot(fig,inner[len(inner)-1][1])
    axes.append(ax)
    caxes.append([])

    
    # plot
    ax.plot(f_hat,Ak_hat1,'k',linewidth=0.5)
    ax.plot(f_hat,Ak_hat2,'r',linewidth=0.5)
    ax.set_ylim([frequency_plot_y_ticks[0],frequency_plot_y_ticks[-1]])
    ax.set_yscale('log')
    ax.text(3.5,frequency_plot_y_ticks[-1]/10.0,label1,fontsize=8,color='black')
    ax.text(3.5,frequency_plot_y_ticks[-1]/100.0,label2,fontsize=8,color='red')
    ax.text(1,frequency_plot_y_ticks[-1]/10.0,subplot_label,fontsize=8)
    ax.set_xticks(frequency_plot_x_ticks)
    ax.set_yticks(frequency_plot_y_ticks)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlabel('$fD/U_{\infty}$',fontsize=8)
    fig.add_subplot(ax)

def contour_wcbar(fig,mid,plot_tuple,ylabel=True,xlabel=False,xtick_label=False,ytick_label=True):
    field,field_label,subplot_label,scale = plot_tuple

    levels = np.linspace(-scale,scale,21)
    inner.append(gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=mid,wspace=0.01,hspace=0.1,width_ratios=[0.97,0.03]))
    ax = plot.Subplot(fig,inner[len(inner)-1][0])
    axes.append(ax)
    ux_plot = ax.contourf(X_grid,Y_grid,field,levels=levels,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    if ylabel:
        ax.set_ylabel('y',fontsize=8,labelpad=-5)
    if xlabel:
        ax.set_xlabel('x',fontsize=8,labelpad=-1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(x_ticks)
    if xtick_label==False:
        ax.xaxis.set_tick_params(labelbottom=False)
    if ytick_label==False:
        ax.yaxis.set_tick_params(labelleft=False)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.text(7.5,1.2,field_label,fontsize=8)
    ax.text(-1.85,1.45,subplot_label,fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    fig.add_subplot(ax)
            
    cax=plot.Subplot(fig,inner[len(inner)-1][1])
    caxes.append(cax)
    cax.set(xmargin=0.5)
    cbar = plot.colorbar(ux_plot,cax,ticks=[scale,scale/2,0.0,-scale/2,-scale],format=tkr.FormatStrFormatter('%.1e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

if True:
       # mode 3 summary dual log version
    x_ticks = np.array([-2,0,2,4,6,8,10])

    # define the vector of quantity tuples for plotting
    contour_quantities = [(phi_xr[:,:,0],'$\Phi_{\mathrm{1x}}$','(a)',np.nanmax(np.abs(phi_xr[:,:,0]))),
                          (phi_yr[:,:,0],'$\Phi_{\mathrm{1y}}$','(b)',np.nanmax(np.abs(phi_yr[:,:,0]))),
                          (phi_xr[:,:,1],'$\Phi_{\mathrm{2x}}$','(d)',np.nanmax(np.abs(phi_xr[:,:,1]))),
                          (phi_yr[:,:,1],'$\Phi_{\mathrm{2y}}$','(e)',np.nanmax(np.abs(phi_yr[:,:,1]))),
                          (phi_xr[:,:,2],'$\Phi_{\mathrm{3x}}$','(f)',np.nanmax(np.abs(phi_xr[:,:,2]))),
                          (phi_yr[:,:,2],'$\Phi_{\mathrm{3y}}$','(g)',np.nanmax(np.abs(phi_yr[:,:,2]))),
                          (phi_xr[:,:,3],'$\Phi_{\mathrm{4x}}$','(i)',np.nanmax(np.abs(phi_xr[:,:,3]))),
                          (phi_yr[:,:,3],'$\Phi_{\mathrm{4y}}$','(j)',np.nanmax(np.abs(phi_yr[:,:,3]))),
                          (phi_xr[:,:,4],'$\Phi_{\mathrm{5x}}$','(k)',np.nanmax(np.abs(phi_xr[:,:,4]))),
                          (phi_yr[:,:,4],'$\Phi_{\mathrm{5y}}$','(l)',np.nanmax(np.abs(phi_yr[:,:,4]))),
                          (phi_xr[:,:,5],'$\Phi_{\mathrm{6x}}$','(n)',np.nanmax(np.abs(phi_xr[:,:,5]))),
                          (phi_yr[:,:,5],'$\Phi_{\mathrm{6y}}$','(o)',np.nanmax(np.abs(phi_yr[:,:,5]))),
                          (phi_xr[:,:,6],'$\Phi_{\mathrm{7x}}$','(p)',np.nanmax(np.abs(phi_xr[:,:,6]))),
                          (phi_yr[:,:,6],'$\Phi_{\mathrm{7y}}$','(q)',np.nanmax(np.abs(phi_yr[:,:,6]))),
                          (phi_xr[:,:,7],'$\Phi_{\mathrm{8x}}$','(s)',np.nanmax(np.abs(phi_xr[:,:,7]))),
                          (phi_yr[:,:,7],'$\Phi_{\mathrm{8y}}$','(t)',np.nanmax(np.abs(phi_yr[:,:,7]))),
                          (phi_xr[:,:,8],'$\Phi_{\mathrm{9x}}$','(u)',np.nanmax(np.abs(phi_xr[:,:,8]))),
                          (phi_yr[:,:,8],'$\Phi_{\mathrm{9y}}$','(v)',np.nanmax(np.abs(phi_yr[:,:,8]))),
                          (phi_xr[:,:,9],'$\Phi_{\mathrm{10x}}$','(x)',np.nanmax(np.abs(phi_xr[:,:,9]))),
                          (phi_yr[:,:,9],'$\Phi_{\mathrm{10y}}$','(y)',np.nanmax(np.abs(phi_yr[:,:,9]))),
                          (phi_xr[:,:,10],'$\Phi_{\mathrm{11x}}$','(z)',np.nanmax(np.abs(phi_xr[:,:,10]))),
                          (phi_yr[:,:,10],'$\Phi_{\mathrm{11y}}$','(aa)',np.nanmax(np.abs(phi_yr[:,:,10]))),
                          (phi_xr[:,:,11],'$\Phi_{\mathrm{12x}}$','(ac)',np.nanmax(np.abs(phi_xr[:,:,11]))),
                          (phi_yr[:,:,11],'$\Phi_{\mathrm{12y}}$','(ad)',np.nanmax(np.abs(phi_yr[:,:,11]))),]

    timeplot_quantities = [(Ak[:,0],Ak[:,1],'$a_{\mathrm{1}}$','$a_{\mathrm{2}}$','(c)'),
                           (Ak[:,2],Ak[:,3],'$a_{\mathrm{3}}$','$a_{\mathrm{4}}$','(i)'),
                           (Ak[:,2],Ak[:,3],'$a_{\mathrm{5}}$','$a_{\mathrm{6}}$','(o)'),]
    
    spectraplot_quantities = [(Ak_hat[:,0],Ak_hat[:,1],'$|\hat{a}_{\mathrm{1}}|$','$|\hat{a}_{\mathrm{2}}|$','(c)'),
                              (Ak_hat[:,2],Ak_hat[:,3],'$|\hat{a}_{\mathrm{3}}|$','$|\hat{a}_{\mathrm{4}}|$','(h)'),
                              (Ak_hat[:,4],Ak_hat[:,5],'$|\hat{a}_{\mathrm{5}}|$','$|\hat{a}_{\mathrm{6}}|$','(m)'),
                              (Ak_hat[:,6],Ak_hat[:,7],'$|\hat{a}_{\mathrm{7}}|$','$|\hat{a}_{\mathrm{8}}|$','(r)'),
                              (Ak_hat[:,8],Ak_hat[:,9],'$|\hat{a}_{\mathrm{9}}|$','$|\hat{a}_{\mathrm{10}}|$','(w)'),
                              (Ak_hat[:,10],Ak_hat[:,11],'$|\hat{a}_{\mathrm{11}}|$','$|\hat{a}_{\mathrm{12}}|$','(ab)'),]

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(7.5,9))
    plot.subplots_adjust(left=0.04,top=0.99,right=0.99,bottom=0.04)
    outer = gridspec.GridSpec(12,1,wspace=0.1,hspace=0.15)
    mid = []
    inner = []

    axes = []
    caxes = []

    wspace_temp = 0.5

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))   
    contour_wcbar(fig,mid[0][0],contour_quantities[0])
    contour_wcbar(fig,mid[0][1],contour_quantities[1],ylabel=False)
    #timeplot_line(fig,mid[0][2],timeplot_quantities[0])
    spectrumplot_line(fig,mid[0][2],spectraplot_quantities[0])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[1][0],contour_quantities[2])
    contour_wcbar(fig,mid[1][1],contour_quantities[3],ylabel=False)
    

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[2][0],contour_quantities[4])
    contour_wcbar(fig,mid[2][1],contour_quantities[5],ylabel=False)
    #timeplot_line(fig,mid[2][2],timeplot_quantities[1])
    spectrumplot_line(fig,mid[2][2],spectraplot_quantities[1])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[3][0],contour_quantities[6])
    contour_wcbar(fig,mid[3][1],contour_quantities[7],ylabel=False)
    

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[4][0],contour_quantities[8])
    contour_wcbar(fig,mid[4][1],contour_quantities[9],ylabel=False)
    #timeplot_line(fig,mid[4][2],timeplot_quantities[2])
    spectrumplot_line(fig,mid[4][2],spectraplot_quantities[2])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[5][0],contour_quantities[10],)
    contour_wcbar(fig,mid[5][1],contour_quantities[11],ylabel=False)
        
    
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[6],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[6][0],contour_quantities[12])
    contour_wcbar(fig,mid[6][1],contour_quantities[13],ylabel=False)
    #timeplot_line(fig,mid[4][2],timeplot_quantities[2])
    spectrumplot_line(fig,mid[6][2],spectraplot_quantities[3])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[7],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[7][0],contour_quantities[14],)
    contour_wcbar(fig,mid[7][1],contour_quantities[15],ylabel=False)

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[8],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[8][0],contour_quantities[16])
    contour_wcbar(fig,mid[8][1],contour_quantities[17],ylabel=False)
    #timeplot_line(fig,mid[4][2],timeplot_quantities[2])
    spectrumplot_line(fig,mid[8][2],spectraplot_quantities[4])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[9],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[9][0],contour_quantities[18],)
    contour_wcbar(fig,mid[9][1],contour_quantities[19],ylabel=False)

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[10],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[10][0],contour_quantities[20])
    contour_wcbar(fig,mid[10][1],contour_quantities[21],ylabel=False)
    #timeplot_line(fig,mid[4][2],timeplot_quantities[2])
    spectrumplot_line(fig,mid[10][2],spectraplot_quantities[5])
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[11],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[11][0],contour_quantities[22],xlabel=True,xtick_label=True,)
    contour_wcbar(fig,mid[11][1],contour_quantities[23],xlabel=True,xtick_label=True,ylabel=False)


    plot.savefig(figures_dir+'pod_overview1.png',dpi=300)
    plot.close(fig)

    # mode 0 summary, dual log scale error plots
    fig = plot.figure(figsize=(7.5,5))
    plot.subplots_adjust(left=0.04,top=0.99,right=0.99,bottom=0.04)
    outer = gridspec.GridSpec(6,1,wspace=0.1,hspace=0.15)
    mid = []
    inner = []

    axes = []
    caxes = []

    wspace_temp = 0.5

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))   
    contour_wcbar(fig,mid[0][0],contour_quantities[0])
    contour_wcbar(fig,mid[0][1],contour_quantities[1],ylabel=False)
    spectrumplot_line(fig,mid[0][2],spectraplot_quantities[0])
   

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[1][0],contour_quantities[4])
    contour_wcbar(fig,mid[1][1],contour_quantities[5],ylabel=False)
    spectrumplot_line(fig,mid[1][2],spectraplot_quantities[1])
    

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[2][0],contour_quantities[8])
    contour_wcbar(fig,mid[2][1],contour_quantities[9],ylabel=False)
    spectrumplot_line(fig,mid[2][2],spectraplot_quantities[2])
        
    
    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[3][0],contour_quantities[12])
    contour_wcbar(fig,mid[3][1],contour_quantities[13],ylabel=False)
    spectrumplot_line(fig,mid[3][2],spectraplot_quantities[3])

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[4][0],contour_quantities[16])
    contour_wcbar(fig,mid[4][1],contour_quantities[17],ylabel=False)
    spectrumplot_line(fig,mid[4][2],spectraplot_quantities[4])

    mid.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=wspace_temp,hspace=0.1,height_ratios=[1],width_ratios=[2,2,1]))
    contour_wcbar(fig,mid[5][0],contour_quantities[20],xlabel=True,xtick_label=True,)
    contour_wcbar(fig,mid[5][1],contour_quantities[21],xlabel=True,xtick_label=True,ylabel=False)
    spectrumplot_line(fig,mid[5][2],spectraplot_quantities[5])

    plot.savefig(figures_dir+'pod_overview1_short.png',dpi=300)
    plot.close(fig)
