

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

figures_dir = 'C:/projects/paper_figures/energy_exchange/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

cases_list_m = ['mfg_fbc003_001_S0/mfg_fbc003_001_S0_ep72927_pred.mat','mfg_fbc003_001_S2/mfg_fbc003_001_S2_ep74925_pred.mat','mfg_fbc003_001_S4/mfg_fbc003_001_S4_ep86913_pred.mat','mfg_fbc003_001_S8/mfg_fbc003_001_S8_ep101898_pred.mat','mfg_fbc003_001_S16/mfg_fbc003_001_S16_ep69930_pred.mat','mfg_fbc003_001_S32/mfg_fbc003_001_S32_ep72927_pred.mat']
cases_list_f = []
cases_list_f.append(['mfg_fbcf007_f0_S0_j001_output/mfg_fbcf007_f0_S0_j001_ep153846_pred.mat','mfg_fbcf007_f0_S2_j001_output/mfg_fbcf007_f0_S2_j001_ep265734_pred.mat','mfg_fbcf007_f0_S4_j001_output/mfg_fbcf007_f0_S4_j001_ep164835_pred.mat','mfg_fbcf007_f0_S8_j001_output/mfg_fbcf007_f0_S8_j001_ep167832_pred.mat','mfg_fbcf007_f0_S16_j001_output/mfg_fbcf007_f0_S16_j001_ep175824_pred.mat','mfg_fbcf007_f0_S32_j001_output/mfg_fbcf007_f0_S32_j001_ep164835_pred.mat'])
cases_list_f.append(['mfg_fbcf007_f1_S0_j001_output/mfg_fbcf007_f1_S0_j001_ep153846_pred.mat','mfg_fbcf007_f1_S2_j001_output/mfg_fbcf007_f1_S2_j001_ep163836_pred.mat','mfg_fbcf007_f1_S4_j001_output/mfg_fbcf007_f1_S4_j001_ep164835_pred.mat','mfg_fbcf007_f1_S8_j001_output/mfg_fbcf007_f1_S8_j001_ep164835_pred.mat','mfg_fbcf007_f1_S16_j001_output/mfg_fbcf007_f1_S16_j001_ep164835_pred.mat','mfg_fbcf007_f1_S32_j001_output/mfg_fbcf007_f1_S32_j001_ep161838_pred.mat'])
cases_list_f.append(['mfg_fbcf007_f2_S0_j001_output/mfg_fbcf007_f2_S0_j001_ep154845_pred.mat','mfg_fbcf007_f2_S2_j001_output/mfg_fbcf007_f2_S2_j001_ep157842_pred.mat','mfg_fbcf007_f2_S4_j001_output/mfg_fbcf007_f2_S4_j001_ep164835_pred.mat','mfg_fbcf007_f2_S8_j001_output/mfg_fbcf007_f2_S8_j001_ep165834_pred.mat','mfg_fbcf007_f2_S16_j001_output/mfg_fbcf007_f2_S16_j001_ep164835_pred.mat','mfg_fbcf007_f2_S32_j001_output/mfg_fbcf007_f2_S32_j001_ep171828_pred.mat'])
cases_supersample_factor = [0,2,4,8,16,32]


# load the reference data

# get the constants for all the modes
class UserScalingParameters(object):
    pass
ScalingParameters = UserScalingParameters()
ScalingParameters.f =[]
ScalingParameters.f.append(UserScalingParameters())

# load the reference data
base_dir = data_dir

meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')

#fluctuatingVelocityFile = h5py.File(base_dir+'fluctuatingVelocity.mat','r')
#fluctuatingPressureFile = h5py.File(base_dir+'fluctuatingPressure.mat','r')


x = np.array(configFile['X_vec'][0,:])
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
d = np.array(configFile['cylinderDiameter'])

ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

X_grid_plot = X_grid
Y_grid_plot = Y_grid
X_plot = np.stack((X_grid_plot.flatten(),Y_grid_plot.flatten()),axis=1)

cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

ScalingParameters.f[0].fs = 10.0
ScalingParameters.f[0].MAX_x = 10.0
ScalingParameters.f[0].MAX_y = 10.0
ScalingParameters.f[0].MAX_p = 1.0
ScalingParameters.f[0].nu_mol = 0.0066667
ScalingParameters.f[0].MAX_ux = np.max(ux)
ScalingParameters.f[0].MAX_uy = np.max(uy)

phi_x_ref = []
phi_y_ref = []

# mode zero is the mean fields
phi_x_ref.append(np.reshape(np.array(meanVelocityFile['meanVelocity'][0,:],dtype=np.complex128).transpose(),[X_grid.shape[0],X_grid.shape[1]]))
phi_y_ref.append(np.reshape(np.array(meanVelocityFile['meanVelocity'][1,:],dtype=np.complex128).transpose(),[X_grid.shape[0],X_grid.shape[1]]))

fourierModeFile = h5py.File(base_dir+'fourierModes.mat','r')

nM_ref = 7
# load reference data
for mode_number in range(nM_ref):
    
    phi_x = np.array(np.reshape(fourierModeFile['velocityModes'][:,mode_number,0],[X_grid.shape[0],X_grid.shape[1]]),dtype=np.complex128)
    phi_x_ref.append(phi_x)
    phi_y = np.array(np.reshape(fourierModeFile['velocityModes'][:,mode_number,1],[X_grid.shape[0],X_grid.shape[1]]),dtype=np.complex128)
    phi_y_ref.append(phi_y)
 
    fs = 10.0 #np.array(configFile['fs'])
    omega_0 = np.array(fourierModeFile['modeFrequencies'][0])*2*np.pi
    omega = np.array(fourierModeFile['modeFrequencies'][mode_number])*2*np.pi

    ScalingParameters.f.append(UserScalingParameters())
    ScalingParameters.f[mode_number+1].MAX_x = 20.0
    ScalingParameters.f[mode_number+1].MAX_y = 20.0 # we use the larger of the two spatial scalings
    ScalingParameters.f[mode_number+1].MAX_phi_xr = np.max((np.real(phi_x)).flatten())
    ScalingParameters.f[mode_number+1].MAX_phi_xi = np.max((np.imag(phi_x)).flatten())
    ScalingParameters.f[mode_number+1].MAX_phi_yr = np.max((np.real(phi_y)).flatten())
    ScalingParameters.f[mode_number+1].MAX_phi_yi = np.max((np.imag(phi_y)).flatten())
    ScalingParameters.f[mode_number+1].MAX_psi= 0.2*np.power((omega_0/omega),2.0) # chosen based on abs(max(psi)) # since this decays with frequency, we multiply by the inverse to prevent a scaling issue
    ScalingParameters.f[mode_number+1].omega = omega
    ScalingParameters.f[mode_number+1].f = np.array(fourierModeFile['modeFrequencies'][mode_number])
    ScalingParameters.f[mode_number+1].nu_mol = 0.0066667

for mode_number in range(1,nM_ref+1):
    # append the conjugates as 4:8
    phi_x_ref.append(np.conj(phi_x_ref[mode_number]))
    phi_y_ref.append(np.conj(phi_y_ref[mode_number]))

mode_symbols = ['\overline{u}','\phi_{1}','\phi_{2}','\phi_{3}','\phi^*_{1}','\phi^*_{2}','\phi^*_{3}',]
mode_numbers_star = ['0','1','2','3','1°','2°','3°']

def plot_e_ij(i,j):
    # we assume i and j are from 0 to 3
    mode_i = abs(i)+nM_ref*(i>0) # 0 if 0, if positive use conjugate; if negative use regular. 
    mode_i_less_j = abs(i-j)+nM_ref*((i-j)<0) # if negative we use the conj 
    mode_j = abs(j)+nM_ref*(j<0)  # this one we just use as is
    print(i,' ',i-j,' ',j)
    print(mode_i,' ',mode_i_less_j,'',mode_j)


    # compute the gradient of mode j
    d_phi_j_x_x = np.gradient(phi_x_ref[mode_j],X_grid[:,0],axis=0)
    d_phi_j_x_y = np.gradient(phi_x_ref[mode_j],Y_grid[0,:],axis=1)
    d_phi_j_y_x = np.gradient(phi_y_ref[mode_j],X_grid[:,0],axis=0)
    d_phi_j_y_y = np.gradient(phi_y_ref[mode_j],Y_grid[0,:],axis=1)

    e_ij_1 = phi_x_ref[mode_i]*phi_x_ref[mode_i_less_j]*d_phi_j_x_x 
    e_ij_2 = phi_x_ref[mode_i]*phi_y_ref[mode_i_less_j]*d_phi_j_x_y 
    e_ij_3 = phi_y_ref[mode_i]*phi_x_ref[mode_i_less_j]*d_phi_j_y_x 
    e_ij_4 = phi_y_ref[mode_i]*phi_y_ref[mode_i_less_j]*d_phi_j_y_y 

    e_ij = e_ij_1 + e_ij_2 + e_ij_3 + e_ij_4

    # set inside values zero
    e_ij[cylinder_mask]=np.NaN

    MAX_e_ij = np.nanmax([np.nanmax(np.abs(np.real(e_ij))),np.nanmax(np.abs(np.imag(e_ij)))])
    levels_e_ij = np.linspace(-MAX_e_ij,MAX_e_ij,21)

    cbar_ticks = [-MAX_e_ij,-0.5*MAX_e_ij,0,0.5*MAX_e_ij,MAX_e_ij]

    fig = plot.figure(figsize=(3.37,(4/12)*3.37*2+0.25))
    plot.subplots_adjust(left=0.13,top=0.99,right=0.88,bottom=0.12)
    outer = gridspec.GridSpec(2,1,wspace=0.1,hspace=0.05)
    inner = []       

    # quadrant 1

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.15,hspace=0.1,width_ratios=[0.93,0.02,0.05]))

    # (1,(1,1))
    ax = plot.Subplot(fig,inner[0][0])
    ux_plot = ax.contourf(X_grid,Y_grid,np.real(e_ij),levels=levels_e_ij,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xticks(np.array([-2,0,2,4,6,8,10]))
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.text(5,1.4,'$R('+mode_symbols[mode_i]+'\cdot ('+mode_symbols[mode_i_less_j]+'\cdot \\nabla)'+mode_symbols[mode_j]+')$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(a)',fontsize=8)
    fig.add_subplot(ax)
        
    cax=plot.Subplot(fig,inner[0][1])
    cbar = plot.colorbar(ux_plot,cax,ticks=cbar_ticks, format=tkr.FormatStrFormatter('%.1e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)
        
    # quadrant 2

    inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.15,hspace=0.1,width_ratios=[0.93,0.02,0.05]))

    ax = plot.Subplot(fig,inner[1][0])
    uy_plot =ax.contourf(X_grid,Y_grid,np.imag(e_ij),levels=levels_e_ij,cmap= matplotlib.colormaps['bwr'],extend='both')
    ax.set_aspect('equal')
    #ax.set_yticks(np.array([2.0,0.0,-1.0,-2.0]))
    ax.set_xticks(np.array([-2,0,2,4,6,8,10]))
    ax.yaxis.set_tick_params(labelleft=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
    ax.set_xlabel('x/D',fontsize=8)    
    ax.text(5,1.4,'$Im('+mode_symbols[mode_i]+'\cdot ('+mode_symbols[mode_i_less_j]+'\cdot \\nabla)'+mode_symbols[mode_j]+')$',fontsize=8)
    circle = plot.Circle((0,0),0.5,color='k',fill=False)
    ax.add_patch(circle)
    ax.text(-1.75,1.4,'(b)',fontsize=8)
    fig.add_subplot(ax)

    cax=plot.Subplot(fig,inner[1][1])
    cbar = plot.colorbar(uy_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.1e'))
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)

    filename = 'spatial_modes/triad_'+mode_numbers_star[mode_i]+'_'+mode_numbers_star[mode_i_less_j]+'_'+mode_numbers_star[mode_j]
    plot.savefig(figures_dir+filename+'.pdf')
    plot.savefig(figures_dir+filename+'.png',dpi=300)

    plot.close(fig)

def compute_valid_triads(m,r):
    triads = []
    for ri in range(-r,r):
        if ((m-ri)>=-r) and ((m-ri)<=r):
            triads.append([m,m-ri,ri])

    return triads

def compute_integral_eij(i,j):
    # we assume i and j are from 0 to 3
    mode_i = abs(i)+nM_ref*(i>0) # 0 if 0, if positive use conjugate; if negative use regular. 
    mode_i_less_j = abs(i-j)+nM_ref*((i-j)<0) # if negative we use the conj 
    mode_j = abs(j)+nM_ref*(j<0)  # this one we just use as is

    # compute the gradient of mode j
    d_phi_j_x_x = np.gradient(phi_x_ref[mode_j],X_grid[:,0],axis=0)
    d_phi_j_x_y = np.gradient(phi_x_ref[mode_j],Y_grid[0,:],axis=1)
    d_phi_j_y_x = np.gradient(phi_y_ref[mode_j],X_grid[:,0],axis=0)
    d_phi_j_y_y = np.gradient(phi_y_ref[mode_j],Y_grid[0,:],axis=1)

    e_ij_1 = phi_x_ref[mode_i]*phi_x_ref[mode_i_less_j]*d_phi_j_x_x 
    e_ij_2 = phi_x_ref[mode_i]*phi_y_ref[mode_i_less_j]*d_phi_j_x_y 
    e_ij_3 = phi_y_ref[mode_i]*phi_x_ref[mode_i_less_j]*d_phi_j_y_x 
    e_ij_4 = phi_y_ref[mode_i]*phi_y_ref[mode_i_less_j]*d_phi_j_y_y 

    e_ij = e_ij_1 + e_ij_2 + e_ij_3 + e_ij_4

    e_ij[cylinder_mask] = 0+0j

    t1r = np.trapz(np.real(e_ij),X_grid[:,0],axis=0)
    t1i = np.trapz(np.imag(e_ij),X_grid[:,0],axis=0)
    t2r = np.trapz(t1r,Y_grid[0,:],axis=0)
    t2i = np.trapz(t1i,Y_grid[0,:],axis=0)
    return t2r+t2i*1j, e_ij


mode_0_triads = compute_valid_triads(0,3)
mode_1_triads = compute_valid_triads(1,3)
mode_2_triads = compute_valid_triads(2,3)
mode_3_triads = compute_valid_triads(3,3)







#for m in range(len(mode_0_triads)):
#    plot_e_ij(mode_0_triads[m][0],mode_0_triads[m][2])


#for m in range(len(mode_1_triads)):
#    plot_e_ij(mode_1_triads[m][0],mode_1_triads[m][2])


#for m in range(len(mode_2_triads)):
#    plot_e_ij(mode_2_triads[m][0],mode_2_triads[m][2])


#for m in range(len(mode_3_triads)):
#    plot_e_ij(mode_3_triads[m][0],mode_3_triads[m][2])

if False:
    # plot all E_ij for 0:7
    e_ij_matrix = np.zeros([X_grid.shape[0],X_grid.shape[1],15,15],np.complex128)
    E_ij_matrix = np.zeros([15,15],np.complex128)
    for i in range(-7,7+1):
        for j in range(-7,7+1):
            if (i-j)>=7 or (i-j)<=-7:
                # out of range frequencies
                E_ij_matrix[i+7,j+7]=np.NaN
            else:
                E_ij_matrix[i+7,j+7],e_ij_matrix[:,:,i+7,j+7]=compute_integral_eij(i,j)

    freq_indices = np.arange(-7,7+1)
    freq_mesh_i,freq_mesh_j = np.meshgrid(freq_indices,freq_indices)

    temp_E_ij_real_pos = np.real(1.0*E_ij_matrix)
    temp_E_ij_real_pos[temp_E_ij_real_pos<0]=np.NaN
    temp_E_ij_real_pos[(temp_E_ij_real_pos>0)*(temp_E_ij_real_pos<1E-4)]=0.0
    temp_E_ij_real_neg = np.real(1.0*E_ij_matrix)
    temp_E_ij_real_neg[temp_E_ij_real_neg>=0]=np.NaN
    temp_E_ij_real_neg = np.abs(temp_E_ij_real_neg)
    temp_E_ij_real_neg[(temp_E_ij_real_neg>0)*(temp_E_ij_real_neg<1E-4)]=0.0

    temp_E_ij_imag_pos = np.imag(1.0*E_ij_matrix)
    temp_E_ij_imag_pos[temp_E_ij_imag_pos<0]=np.NaN
    temp_E_ij_imag_pos[(temp_E_ij_imag_pos>0)*(temp_E_ij_imag_pos<1E-4)]=0.0
    temp_E_ij_imag_neg = np.imag(1.0*E_ij_matrix)
    temp_E_ij_imag_neg[temp_E_ij_imag_neg>=0]=np.NaN
    temp_E_ij_imag_neg = np.abs(temp_E_ij_imag_neg)
    temp_E_ij_imag_neg[(temp_E_ij_imag_neg>0)*(temp_E_ij_imag_neg<1E-4)]=0.0

    MAX_E_ij = np.nanmax([np.nanmax(temp_E_ij_real_pos.ravel()),np.nanmax(temp_E_ij_real_neg.ravel()),np.nanmax(temp_E_ij_imag_pos.ravel()),np.nanmax(temp_E_ij_imag_neg.ravel())])


    cmap1 = matplotlib.colormaps['Reds']
    cmap1.set_bad('white',alpha=0.0)
    cmap2 = matplotlib.colormaps['Blues']
    cmap2.set_bad('white',alpha=0.0)


    # make the mode energy crossplot
    fig = plot.figure(figsize=(3.37,3.37*2+0.25))
    plot.subplots_adjust(left=0.15,top=0.99,right=0.97,bottom=0.01)
    outer = gridspec.GridSpec(3,1,wspace=0.1,hspace=0.05,height_ratios=[0.45,0.45,0.1])
    inner = []   

    cbar_ticks = [1E-4,1E-3,1E-2,1E-1,1]
    xy_ticks = [-6,-4,-2,0,2,4,6]
    xy_minor_ticks = [-7,-5,-3,-1,1,3,5,7]

    ax = plot.Subplot(fig,outer[0])
    ax.set_aspect('equal')
    p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
    n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
    ax.text(-6,6,'(a)',fontsize=8)
    ax.set_yticks(xy_ticks)
    ax.set_yticks(xy_minor_ticks,minor=True)
    ticklabs = ax.get_yticklabels()
    ax.set_yticklabels(ticklabs, fontsize=8)
    ax.set_xticks(xy_ticks)
    ax.set_xticks(xy_minor_ticks,minor=True)
    ticklabs = ax.get_xticklabels()
    ax.set_xticklabels(ticklabs, fontsize=8)
    ax.set_ylabel('$f/f_{vs}$',fontsize=8)
    fig.add_subplot(ax)


    ax = plot.Subplot(fig,outer[1])
    ax.set_aspect('equal')
    p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
    n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
    ax.text(-6,6,'(b)',fontsize=8)
    ax.set_yticks(xy_ticks)
    ax.set_yticks(xy_minor_ticks,minor=True)
    ticklabs = ax.get_yticklabels()
    ax.set_yticklabels(ticklabs, fontsize=8)
    ax.set_xticks(xy_ticks)
    ax.set_xticks(xy_minor_ticks,minor=True)
    ticklabs = ax.get_xticklabels()
    ax.set_xticklabels(ticklabs, fontsize=8)
    ax.set_ylabel('$f/f_{vs}$',fontsize=8)
    ax.set_xlabel('$f/f_{vs}$',fontsize=8)
    fig.add_subplot(ax)


    inner.append(gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=outer[2],wspace=0.15,hspace=0.3,height_ratios=[0.04,0.02,0.01,0.02,0.03]))
    cax=plot.Subplot(fig,inner[0][1])
    cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'),orientation='horizontal')
    cbar.ax.xaxis.set_tick_params(labelbottom=False)
    #ticklabs = cbar.ax.get_xticklabels()
    #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)


    cax=plot.Subplot(fig,inner[0][3])
    cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'),orientation='horizontal')
    ticklabs = cbar.ax.get_xticklabels()
    cbar.ax.set_xticklabels(ticklabs, fontsize=8)
    fig.add_subplot(cax)


    filename = 'simplified_energy_exchange'
    plot.savefig(figures_dir+filename+'.pdf')
    plot.savefig(figures_dir+filename+'.png',dpi=300)
    plot.close(fig)

def compute_integral_eij_downsample(i,j,S):
    # we assume i and j are from 0 to 3
    mode_i = abs(i)+nM_ref*(i>0) # 0 if 0, if positive use conjugate; if negative use regular. 
    mode_i_less_j = abs(i-j)+nM_ref*((i-j)<0) # if negative we use the conj 
    mode_j = abs(j)+nM_ref*(j<0)  # this one we just use as is

    phi_i_x = phi_x_ref[mode_i]
    phi_i_y = phi_y_ref[mode_i]
    phi_i_less_j_x = phi_x_ref[mode_i_less_j]
    phi_i_less_j_y = phi_y_ref[mode_i_less_j]
    phi_j_x = phi_x_ref[mode_j]
    phi_j_y = phi_y_ref[mode_j]

    # downsample indices
    if S==0:
        X_grid_ds = X_grid
        Y_grid_ds = Y_grid
        phi_i_x_ds = phi_i_x
        phi_i_y_ds = phi_i_y
        phi_i_less_j_x_ds = phi_i_less_j_x
        phi_i_less_j_y_ds = phi_i_less_j_y
        phi_j_x_ds = phi_j_x
        phi_j_y_ds = phi_j_y
        cylinder_mask_ds = cylinder_mask
    else:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(S,X_grid[:,0],Y_grid[0,:].transpose())
        cylinder_mask_ds = np.reshape((cylinder_mask.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        X_grid_ds = np.reshape((X_grid.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        Y_grid_ds = np.reshape((Y_grid.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_i_x_ds = np.reshape((phi_i_x.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_i_y_ds = np.reshape((phi_i_y.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_i_less_j_x_ds = np.reshape((phi_i_less_j_x.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_i_less_j_y_ds = np.reshape((phi_i_less_j_y.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_j_x_ds = np.reshape((phi_j_x.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()
        phi_j_y_ds = np.reshape((phi_j_y.ravel())[linear_downsample_inds],[ndy,ndx]).transpose()

    # compute the gradient of mode j
    phi_j_x_ds_x = np.gradient(phi_j_x_ds,X_grid_ds[:,0],axis=0)
    phi_j_x_ds_y = np.gradient(phi_j_x_ds,Y_grid_ds[0,:],axis=1)
    phi_j_y_ds_x = np.gradient(phi_j_y_ds,X_grid_ds[:,0],axis=0)
    phi_j_y_ds_y = np.gradient(phi_j_y_ds,Y_grid_ds[0,:],axis=1)   

    e_ij_1 = phi_i_x_ds*phi_i_less_j_x_ds*phi_j_x_ds_x 
    e_ij_2 = phi_i_x_ds*phi_i_less_j_y_ds*phi_j_x_ds_y 
    e_ij_3 = phi_i_y_ds*phi_i_less_j_x_ds*phi_j_y_ds_x 
    e_ij_4 =phi_i_y_ds*phi_i_less_j_y_ds*phi_j_y_ds_y 

    e_ij = e_ij_1 + e_ij_2 + e_ij_3 + e_ij_4

    e_ij[cylinder_mask_ds] = 0+0j

    t1r = np.trapz(np.real(e_ij),X_grid_ds[:,0],axis=0)
    t1i = np.trapz(np.imag(e_ij),X_grid_ds[:,0],axis=0)
    t2r = np.trapz(t1r,Y_grid_ds[0,:],axis=0)
    t2i = np.trapz(t1i,Y_grid_ds[0,:],axis=0)
    return t2r+t2i*1j, e_ij

dx = [] # array for supersample spacing
x_ds_grid = []
y_ds_grid = []
ds_inds = []
for s in range(len(cases_supersample_factor)):
    dx.append([])
    if cases_supersample_factor[s]==0:
        dx[s].append(X_grid[1,0]-X_grid[0,0])
        x_ds_grid.append(X_grid)
        y_ds_grid.append(Y_grid)
        ds_inds.append(np.arange(X_grid.size))

    if cases_supersample_factor[s]>0:
        linear_downsample_inds, ndx,ndy = compute_downsample_inds_center(cases_supersample_factor[s],X_grid[:,0],Y_grid[0,:].transpose())

        x_downsample = x[linear_downsample_inds]
        y_downsample = y[linear_downsample_inds]
        
        ds_inds.append(linear_downsample_inds)
        x_ds_grid.append((np.reshape(x_downsample,(ndy,ndx))).transpose())
        y_ds_grid.append((np.reshape(y_downsample,(ndy,ndx))).transpose())
        dx[s].append(x_ds_grid[s][1,0]-x_ds_grid[s][0,0])


pts_per_d = 1.0/np.array(dx)
pts_per_d = pts_per_d[:,0]


if True:
    supersample_factors_ds = [2,4,8,16,32]

    mean_error_ds_real = np.zeros([len(supersample_factors_ds),])
    max_error_ds_real = np.zeros([len(supersample_factors_ds),])
    mean_error_ds_imag = np.zeros([len(supersample_factors_ds),])
    max_error_ds_imag = np.zeros([len(supersample_factors_ds),])

    E_ij_matrix = np.zeros([15,15],np.complex128)
    e_ij_matrix = np.zeros([X_grid.shape[0],X_grid.shape[1],15,15],np.complex128)
    for i in range(-7,7+1):
            for j in range(-7,7+1):
                if (i-j)>=7 or (i-j)<=-7:
                    # out of range frequencies
                    E_ij_matrix[i+7,j+7]=np.NaN
                else:
                    E_ij_matrix[i+7,j+7],e_ij_matrix[:,:,i+7,j+7]=compute_integral_eij(i,j)

    e_ij_ds_matrix =[]
    for s in range(len(supersample_factors_ds)):
        e_ij_ds_matrix.append(np.zeros([x_ds_grid[s+1].shape[0],x_ds_grid[s+1].shape[1],15,15],np.complex128))
    
    E_ij_ds_matrix = np.zeros([15,15,len(supersample_factors_ds)],np.complex128)
    for s in range(len(supersample_factors_ds)):
        # plot error in downsampled E_ij for 0:7, S=[]
        for i in range(-7,7+1):
            for j in range(-7,7+1):
                if (i-j)>=7 or (i-j)<=-7:
                    # out of range frequencies
                    E_ij_ds_matrix[i+7,j+7,s]=np.NaN
                else:
                    E_ij_ds_matrix[i+7,j+7,s],e_ij_ds_matrix[s][:,:,i+7,j+7]=compute_integral_eij_downsample(i,j,supersample_factors_ds[s])

        freq_indices = np.arange(-7,7+1)
        freq_mesh_i,freq_mesh_j = np.meshgrid(freq_indices,freq_indices)

        temp_E_ij_real_pos = np.real(1.0*E_ij_matrix)
        temp_E_ij_real_pos[temp_E_ij_real_pos<0]=np.NaN
        temp_E_ij_real_pos[(temp_E_ij_real_pos>0)*(temp_E_ij_real_pos<1E-4)]=0.0
        temp_E_ij_real_neg = np.real(1.0*E_ij_matrix)
        temp_E_ij_real_neg[temp_E_ij_real_neg>=0]=np.NaN
        temp_E_ij_real_neg = np.abs(temp_E_ij_real_neg)
        temp_E_ij_real_neg[(temp_E_ij_real_neg>0)*(temp_E_ij_real_neg<1E-4)]=0.0

        temp_E_ij_imag_pos = np.imag(1.0*E_ij_matrix)
        temp_E_ij_imag_pos[temp_E_ij_imag_pos<0]=np.NaN
        temp_E_ij_imag_pos[(temp_E_ij_imag_pos>0)*(temp_E_ij_imag_pos<1E-4)]=0.0
        temp_E_ij_imag_neg = np.imag(1.0*E_ij_matrix)
        temp_E_ij_imag_neg[temp_E_ij_imag_neg>=0]=np.NaN
        temp_E_ij_imag_neg = np.abs(temp_E_ij_imag_neg)
        temp_E_ij_imag_neg[(temp_E_ij_imag_neg>0)*(temp_E_ij_imag_neg<1E-4)]=0.0

        temp_E_ij_ds_real_pos = np.real(1.0*E_ij_ds_matrix[:,:,s])
        temp_E_ij_ds_real_pos[temp_E_ij_ds_real_pos<0]=np.NaN
        temp_E_ij_ds_real_pos[(temp_E_ij_ds_real_pos>0)*(temp_E_ij_ds_real_pos<1E-4)]=0.0
        temp_E_ij_ds_real_neg = np.real(1.0*E_ij_ds_matrix[:,:,s])
        temp_E_ij_ds_real_neg[temp_E_ij_ds_real_neg>=0]=np.NaN
        temp_E_ij_ds_real_neg = np.abs(temp_E_ij_ds_real_neg)
        temp_E_ij_ds_real_neg[(temp_E_ij_ds_real_neg>0)*(temp_E_ij_ds_real_neg<1E-4)]=0.0

        temp_E_ij_ds_imag_pos = np.imag(1.0*E_ij_ds_matrix[:,:,s])
        temp_E_ij_ds_imag_pos[temp_E_ij_ds_imag_pos<0]=np.NaN
        temp_E_ij_ds_imag_pos[(temp_E_ij_ds_imag_pos>0)*(temp_E_ij_ds_imag_pos<1E-4)]=0.0
        temp_E_ij_ds_imag_neg = np.imag(1.0*E_ij_ds_matrix[:,:,s])
        temp_E_ij_ds_imag_neg[temp_E_ij_ds_imag_neg>=0]=np.NaN
        temp_E_ij_ds_imag_neg = np.abs(temp_E_ij_ds_imag_neg)
        temp_E_ij_ds_imag_neg[(temp_E_ij_ds_imag_neg>0)*(temp_E_ij_ds_imag_neg<1E-4)]=0.0

        err_E_ij_real = (np.real(E_ij_matrix)-np.real(E_ij_ds_matrix[:,:,s]))#/np.nanmax(np.abs(E_ij_matrix))
        err_E_ij_imag = (np.imag(E_ij_matrix)-np.imag(E_ij_ds_matrix[:,:,s]))#/np.nanmax(np.abs(E_ij_matrix))

        mean_error_ds_real[s]=np.nanmean(np.abs(err_E_ij_real))
        max_error_ds_real[s] = np.nanmax(np.abs(err_E_ij_real))
        mean_error_ds_imag[s]=np.nanmean(np.abs(err_E_ij_imag))
        max_error_ds_imag[s] = np.nanmax(np.abs(err_E_ij_imag))

        err_E_ij_ds_real_pos = 1.0*err_E_ij_real
        err_E_ij_ds_real_pos[err_E_ij_ds_real_pos<0]=np.NaN
        #err_E_ij_ds_real_pos[(err_E_ij_ds_real_pos>0)*(err_E_ij_ds_real_pos<1E-4)]=0.0
        err_E_ij_ds_real_neg = 1.0*err_E_ij_real
        err_E_ij_ds_real_neg[err_E_ij_ds_real_neg>=0]=np.NaN
        err_E_ij_ds_real_neg = np.abs(err_E_ij_ds_real_neg)
        #err_E_ij_ds_real_neg[(err_E_ij_ds_real_neg>0)*(err_E_ij_ds_real_neg<1E-4)]=0.0

        err_E_ij_ds_imag_pos = 1.0*err_E_ij_imag
        err_E_ij_ds_imag_pos[err_E_ij_ds_imag_pos<0]=np.NaN
        #err_E_ij_ds_imag_pos[(err_E_ij_ds_imag_pos>0)*(err_E_ij_ds_imag_pos<1E-4)]=0.0
        err_E_ij_ds_imag_neg = 1.0*err_E_ij_imag
        err_E_ij_ds_imag_neg[err_E_ij_ds_imag_neg>=0]=np.NaN
        err_E_ij_ds_imag_neg = np.abs(err_E_ij_ds_imag_neg)
        #err_E_ij_ds_imag_neg[(err_E_ij_ds_imag_neg>0)*(err_E_ij_ds_imag_neg<1E-4)]=0.0
        
        

        MAX_E_ij = np.nanmax([np.nanmax(temp_E_ij_real_pos.ravel()),np.nanmax(temp_E_ij_real_neg.ravel()),np.nanmax(temp_E_ij_imag_pos.ravel()),np.nanmax(temp_E_ij_imag_neg.ravel()),np.nanmax(temp_E_ij_ds_real_pos.ravel()),np.nanmax(temp_E_ij_ds_real_neg.ravel()),np.nanmax(temp_E_ij_ds_imag_pos.ravel()),np.nanmax(temp_E_ij_ds_imag_neg.ravel())]) 

        MAX_err_E_ij = np.nanmax([np.nanmax(err_E_ij_ds_real_pos.ravel()),np.nanmax(err_E_ij_ds_real_neg.ravel()),np.nanmax(err_E_ij_ds_imag_pos.ravel()),np.nanmax(err_E_ij_ds_imag_neg.ravel()),])

        cmap1 = matplotlib.colormaps['Reds']
        cmap1.set_bad('white',alpha=0.0)
        cmap2 = matplotlib.colormaps['Blues']
        cmap2.set_bad('white',alpha=0.0)


        # make the mode energy crossplot
        fig = plot.figure(figsize=(6.69,7.5))
        plot.subplots_adjust(left=0.05,top=0.99,right=0.9,bottom=0.05)
        outer = gridspec.GridSpec(3,3,wspace=0.05,hspace=0.1,height_ratios=[1,1,1],width_ratios=[0.45,0.45,0.05])
         

        cbar_ticks = [1E-4,1E-3,1E-2,1E-1,1]
        xy_ticks = [-6,-4,-2,0,2,4,6]
        xy_minor_ticks = [-7,-5,-3,-1,1,3,5,7]

        ax = plot.Subplot(fig,outer[0])
        ax.set_aspect('equal')
        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(aa)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[1])
        ax.set_aspect('equal')
        p_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ba)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        #ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[3])
        ax.set_aspect('equal')
        p_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ab)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[4])
        ax.set_aspect('equal')
        p_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(bb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[6])
        ax.set_aspect('equal')
        p_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(ca)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[7])
        ax.set_aspect('equal')
        p_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(cb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)



        inner = []  
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[0][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[0][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[1][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[1][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[8],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[2][0])
        cbar = plot.colorbar(p_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[2][2])
        cbar = plot.colorbar(n_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        #ticklabs = cbar.ax.get_yticklabels()
        #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)



        filename = 'simplified_energy_exchange_error_S'+str(supersample_factors_ds[s])
        plot.savefig(figures_dir+filename+'.pdf')
        plot.savefig(figures_dir+filename+'.png',dpi=300)
        plot.close(fig)



    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3.37)
    plot.subplots_adjust(left=0.17,top=0.95,right=0.9,bottom=0.15)

    y_ticks=[1E-5,1E-4,1E-3,1E-2,1E-1]
    y_tick_labels=['1E-5','1E-4','1E-3','1E-2','1E-1']
    error_x_tick_labels = ['20','10','5','2.5','1.25',]

    mean_plt,=axs.plot(pts_per_d[1:]*0.97,mean_error_ds_real,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    mean_plt2,=axs.plot(pts_per_d[1:]*1.03,mean_error_ds_imag,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    max_plt,=axs.plot(pts_per_d[1:]*0.97,max_error_ds_real,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    max_plt2,=axs.plot(pts_per_d[1:]*1.03,max_error_ds_imag,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs.set_xscale('log')
    axs.set_xticks(pts_per_d[1:])
    axs.xaxis.set_tick_params(labelbottom=True)
    axs.set_yscale('log')
    axs.set_ylim(5E-6,1E0)
    axs.set_yticks(y_ticks,labels=y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    #axs.set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs.legend([mean_plt,mean_plt2,max_plt,max_plt2],['Mean (Re)','Mean (Im)','Max (Re)','Max (Im)'],fontsize=8)
    axs.grid('on')
    axs.set_xticks(pts_per_d[1:])
    axs.set_xticklabels(error_x_tick_labels)
    axs.set_xlabel('Pts/D',fontsize=8)
    #axs[0].text(0.45,10.0,'(a)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'energy_downsample_error.pdf')
    plot.savefig(figures_dir+'energy_downsample_error.png',dpi=300)
    plot.close(fig)


# now load the modes
phi_x_pred = []
phi_y_pred = []

# then the modes
for s in range(len(cases_supersample_factor)):
    phi_x_pred.append([])
    phi_y_pred.append([])
    # mean
    pred_m_file = h5py.File(output_dir+cases_list_m[s],'r')
    phi_x = np.array(pred_m_file['pred'][:,0],dtype=np.complex128)*ScalingParameters.f[0].MAX_ux
    phi_y = np.array(pred_m_file['pred'][:,1],dtype=np.complex128)*ScalingParameters.f[0].MAX_uy
    phi_x_pred[s].append(np.reshape(phi_x,[X_grid.shape[0],X_grid.shape[1]]))
    phi_y_pred[s].append(np.reshape(phi_y,[X_grid.shape[0],X_grid.shape[1]]))
    # modes
    for c in [0,1,2]:
        pred_f_file = h5py.File(output_dir+cases_list_f[c][s],'r')
        phi_x = np.array(pred_f_file['pred'][:,0],dtype=np.complex128)*ScalingParameters.f[c+1].MAX_phi_xr + np.array(1j*pred_f_file['pred'][:,1],dtype=np.complex128)*ScalingParameters.f[c+1].MAX_phi_xi
        phi_y = np.array(pred_f_file['pred'][:,2],dtype=np.complex128)*ScalingParameters.f[c+1].MAX_phi_yr + np.array(1j*pred_f_file['pred'][:,3],dtype=np.complex128)*ScalingParameters.f[c+1].MAX_phi_yi
        phi_x_pred[s].append(np.reshape(phi_x,[X_grid.shape[0],X_grid.shape[1]]))
        phi_y_pred[s].append(np.reshape(phi_y,[X_grid.shape[0],X_grid.shape[1]]))
    for c in [1,2,3]:
        # add the conjugate of the last 3 modes
        phi_x_pred[s].append(np.conj(phi_x_pred[s][c]))
        phi_y_pred[s].append(np.conj(phi_y_pred[s][c]))



n_LOM=3
def compute_integral_eij_pred(i,j,s):
    # we assume i and j are from 0 to 3
    mode_i = abs(i)+n_LOM*(i>0) # 0 if 0, if positive use conjugate; if negative use regular. 
    mode_i_less_j = abs(i-j)+n_LOM*((i-j)<0) # if negative we use the conj 
    mode_j = abs(j)+n_LOM*(j<0)  # this one we just use as is

    phi_i_x = phi_x_pred[s][mode_i]
    phi_i_y = phi_y_pred[s][mode_i]
    phi_i_less_j_x = phi_x_pred[s][mode_i_less_j]
    phi_i_less_j_y = phi_y_pred[s][mode_i_less_j]
    phi_j_x = phi_x_pred[s][mode_j]
    phi_j_y = phi_y_pred[s][mode_j]

    # compute the gradient of mode j
    phi_j_x_x = np.gradient(phi_j_x,X_grid[:,0],axis=0)
    phi_j_x_y = np.gradient(phi_j_x,Y_grid[0,:],axis=1)
    phi_j_y_x = np.gradient(phi_j_y,X_grid[:,0],axis=0)
    phi_j_y_y = np.gradient(phi_j_y,Y_grid[0,:],axis=1)   

    e_ij_1 = phi_i_x*phi_i_less_j_x*phi_j_x_x 
    e_ij_2 = phi_i_x*phi_i_less_j_y*phi_j_x_y 
    e_ij_3 = phi_i_y*phi_i_less_j_x*phi_j_y_x 
    e_ij_4 =phi_i_y*phi_i_less_j_y*phi_j_y_y 

    e_ij = e_ij_1 + e_ij_2 + e_ij_3 + e_ij_4

    e_ij[cylinder_mask] = 0+0j

    t1r = np.trapz(np.real(e_ij),X_grid[:,0],axis=0)
    t1i = np.trapz(np.imag(e_ij),X_grid[:,0],axis=0)
    t2r = np.trapz(t1r,Y_grid[0,:],axis=0)
    t2i = np.trapz(t1i,Y_grid[0,:],axis=0)
    return t2r+t2i*1j, e_ij


if True:
    e_ij_pred_matrix = np.zeros([X_grid.shape[0],X_grid.shape[1],2*n_LOM+1,2*n_LOM+1,len(cases_supersample_factor)],np.complex128)
    e_ij_ds_LOM_matrix = []
    for s in range(len(supersample_factors_ds)):
        e_ij_ds_LOM_matrix.append(np.zeros([x_ds_grid[s+1].shape[0],x_ds_grid[s+1].shape[1],2*n_LOM+1,2*n_LOM+1],np.complex128))
    e_ij_LOM_matrix = np.zeros([X_grid.shape[0],X_grid.shape[1],2*n_LOM+1,2*n_LOM+1],np.complex128)

    E_ij_pred_matrix = np.zeros([2*n_LOM+1,2*n_LOM+1,len(cases_supersample_factor)],np.complex128)
    E_ij_ds_LOM_matrix = np.zeros([2*n_LOM+1,2*n_LOM+1,len(supersample_factors_ds)],np.complex128)
    E_ij_LOM_matrix = np.zeros([2*n_LOM+1,2*n_LOM+1],np.complex128)

    mean_error_PINN_real = np.zeros([len(cases_supersample_factor),])
    max_error_PINN_real = np.zeros([len(cases_supersample_factor),])
    mean_error_PINN_imag = np.zeros([len(cases_supersample_factor),])
    max_error_PINN_imag = np.zeros([len(cases_supersample_factor),])

    mean_error_PINN = np.zeros([len(cases_supersample_factor),])
    max_error_PINN = np.zeros([len(cases_supersample_factor),])



    for s in range(len(supersample_factors_ds)):
        for i in range(-n_LOM,n_LOM+1):
            for j in range(-n_LOM,n_LOM+1):
                if (i-j)>=n_LOM or (i-j)<=-n_LOM:
                    # out of range frequencies
                    E_ij_ds_LOM_matrix[i+n_LOM,j+n_LOM,s]=np.NaN # downsampled 
                    e_ij_ds_LOM_matrix[s][:,:,i+n_LOM,j+n_LOM]=np.NaN
                else:
                    E_ij_ds_LOM_matrix[i+n_LOM,j+n_LOM,s]=E_ij_ds_matrix[i+nM_ref,j+nM_ref,s]
                    e_ij_ds_LOM_matrix[s][:,:,i+n_LOM,j+n_LOM] = e_ij_ds_matrix[s][:,:,i+nM_ref,j+nM_ref]


    for s in range(len(cases_supersample_factor)):
        # plot error in downsampled E_ij for 0:7, S=[]
        for i in range(-n_LOM,n_LOM+1):
            for j in range(-n_LOM,n_LOM+1):
                if (i-j)>=n_LOM or (i-j)<=-n_LOM:
                    # out of range frequencies
                    E_ij_pred_matrix[i+n_LOM,j+n_LOM,s]=np.NaN # PINN
                    E_ij_LOM_matrix[i+n_LOM,j+n_LOM]=np.NaN # true
                    e_ij_pred_matrix[:,:,i+n_LOM,j+n_LOM,s]=np.NaN # PINN
                    e_ij_LOM_matrix[:,:,i+n_LOM,j+n_LOM]=np.NaN # true
                else:
                    E_ij_pred_matrix[i+n_LOM,j+n_LOM,s],e_ij_pred_matrix[:,:,i+n_LOM,j+n_LOM,s]=compute_integral_eij_pred(i,j,s)
                    E_ij_LOM_matrix[i+n_LOM,j+n_LOM]=E_ij_matrix[i+nM_ref,j+nM_ref]
                    e_ij_LOM_matrix[:,:,i+n_LOM,j+n_LOM]=e_ij_matrix[:,:,i+nM_ref,j+nM_ref]

        freq_indices = np.arange(-n_LOM,n_LOM+1)
        freq_mesh_i,freq_mesh_j = np.meshgrid(freq_indices,freq_indices)

        temp_E_ij_real_pos = np.real(1.0*E_ij_LOM_matrix)
        temp_E_ij_real_pos[temp_E_ij_real_pos<0]=np.NaN
        temp_E_ij_real_pos[(temp_E_ij_real_pos>0)*(temp_E_ij_real_pos<1E-4)]=0.0
        temp_E_ij_real_neg = np.real(1.0*E_ij_LOM_matrix)
        temp_E_ij_real_neg[temp_E_ij_real_neg>=0]=np.NaN
        temp_E_ij_real_neg = np.abs(temp_E_ij_real_neg)
        temp_E_ij_real_neg[(temp_E_ij_real_neg>0)*(temp_E_ij_real_neg<1E-4)]=0.0

        temp_E_ij_imag_pos = np.imag(1.0*E_ij_LOM_matrix)
        temp_E_ij_imag_pos[temp_E_ij_imag_pos<0]=np.NaN
        temp_E_ij_imag_pos[(temp_E_ij_imag_pos>0)*(temp_E_ij_imag_pos<1E-4)]=0.0
        temp_E_ij_imag_neg = np.imag(1.0*E_ij_LOM_matrix)
        temp_E_ij_imag_neg[temp_E_ij_imag_neg>=0]=np.NaN
        temp_E_ij_imag_neg = np.abs(temp_E_ij_imag_neg)
        temp_E_ij_imag_neg[(temp_E_ij_imag_neg>0)*(temp_E_ij_imag_neg<1E-4)]=0.0

        temp_E_ij_pred_real_pos = np.real(1.0*E_ij_pred_matrix[:,:,s]) #
        temp_E_ij_pred_real_pos[temp_E_ij_pred_real_pos<0]=np.NaN
        temp_E_ij_pred_real_pos[(temp_E_ij_pred_real_pos>0)*(temp_E_ij_pred_real_pos<1E-4)]=0.0
        temp_E_ij_pred_real_neg = np.real(1.0*E_ij_pred_matrix[:,:,s]) # 
        temp_E_ij_pred_real_neg[temp_E_ij_pred_real_neg>=0]=np.NaN
        temp_E_ij_pred_real_neg = np.abs(temp_E_ij_pred_real_neg)
        temp_E_ij_pred_real_neg[(temp_E_ij_pred_real_neg>0)*(temp_E_ij_pred_real_neg<1E-4)]=0.0

        temp_E_ij_pred_imag_pos = np.imag(1.0*E_ij_pred_matrix[:,:,s])
        temp_E_ij_pred_imag_pos[temp_E_ij_pred_imag_pos<0]=np.NaN
        temp_E_ij_pred_imag_pos[(temp_E_ij_pred_imag_pos>0)*(temp_E_ij_pred_imag_pos<1E-4)]=0.0
        temp_E_ij_pred_imag_neg = np.imag(1.0*E_ij_pred_matrix[:,:,s])
        temp_E_ij_pred_imag_neg[temp_E_ij_pred_imag_neg>=0]=np.NaN
        temp_E_ij_pred_imag_neg = np.abs(temp_E_ij_pred_imag_neg)
        temp_E_ij_pred_imag_neg[(temp_E_ij_pred_imag_neg>0)*(temp_E_ij_pred_imag_neg<1E-4)]=0.0

        err_E_ij_real = (np.real(E_ij_LOM_matrix)-np.real(E_ij_pred_matrix[:,:,s]))/np.nanmax(np.abs(np.real(E_ij_LOM_matrix)),axis=0)
        err_E_ij_imag = (np.imag(E_ij_LOM_matrix)-np.imag(E_ij_pred_matrix[:,:,s]))/np.nanmax(np.abs(np.imag(E_ij_LOM_matrix)),axis=0)
        err_E_ij = (E_ij_pred_matrix[:,:,s]-E_ij_LOM_matrix)/np.nanmax(np.abs(E_ij_LOM_matrix),axis=0)

        mean_error_PINN_real[s]=np.nanmean(np.abs(err_E_ij_real))
        max_error_PINN_real[s] = np.nanmax(np.abs(err_E_ij_real))
        mean_error_PINN_imag[s]=np.nanmean(np.abs(err_E_ij_imag))
        max_error_PINN_imag[s] = np.nanmax(np.abs(err_E_ij_imag))

        mean_error_PINN[s] = np.nanmean(np.abs(err_E_ij))
        max_error_PINN[s] = np.nanmax(np.abs(err_E_ij))

        #err_E_ij_pred_real_pos = 1.0*err_E_ij_real
        #err_E_ij_pred_real_pos[err_E_ij_pred_real_pos<0]=np.NaN
        #err_E_ij_pred_real_pos[(err_E_ij_pred_real_pos>0)*(err_E_ij_pred_real_pos<1E-4)]=0.0
        #err_E_ij_pred_real_neg = 1.0*err_E_ij_real
        #err_E_ij_pred_real_neg[err_E_ij_pred_real_neg>=0]=np.NaN
        #err_E_ij_pred_real_neg = np.abs(err_E_ij_pred_real_neg)
        #err_E_ij_pred_real_neg[(err_E_ij_pred_real_neg>0)*(err_E_ij_pred_real_neg<1E-4)]=0.0

        #err_E_ij_pred_imag_pos = 1.0*err_E_ij_imag
        #err_E_ij_pred_imag_pos[err_E_ij_pred_imag_pos<0]=np.NaN
        #err_E_ij_pred_imag_pos[(err_E_ij_pred_imag_pos>0)*(err_E_ij_pred_imag_pos<1E-4)]=0.0
        #err_E_ij_pred_imag_neg = 1.0*err_E_ij_imag
        #err_E_ij_pred_imag_neg[err_E_ij_pred_imag_neg>=0]=np.NaN
        #err_E_ij_pred_imag_neg = np.abs(err_E_ij_pred_imag_neg)
        #err_E_ij_pred_imag_neg[(err_E_ij_pred_imag_neg>0)*(err_E_ij_pred_imag_neg<1E-4)]=0.0
        
        

        MAX_E_ij = np.nanmax([np.nanmax(temp_E_ij_real_pos.ravel()),np.nanmax(temp_E_ij_real_neg.ravel()),np.nanmax(temp_E_ij_imag_pos.ravel()),np.nanmax(temp_E_ij_imag_neg.ravel()),np.nanmax(temp_E_ij_pred_real_pos.ravel()),np.nanmax(temp_E_ij_pred_real_neg.ravel()),np.nanmax(temp_E_ij_pred_imag_pos.ravel()),np.nanmax(temp_E_ij_pred_imag_neg.ravel())]) 

        #MAX_err_E_ij = np.nanmax([np.nanmax(err_E_ij_pred_real_pos.ravel()),np.nanmax(err_E_ij_pred_real_neg.ravel()),np.nanmax(err_E_ij_pred_imag_pos.ravel()),np.nanmax(err_E_ij_pred_imag_neg.ravel()),])
        MAX_err_E_ij = np.nanmax([np.nanmax(np.abs(err_E_ij_real.ravel())),np.nanmax(np.abs(err_E_ij_imag.ravel()))])

        cmap1 = matplotlib.colormaps['Reds']
        cmap1.set_bad('white',alpha=0.0)
        cmap2 = matplotlib.colormaps['Blues']
        cmap2.set_bad('white',alpha=0.0)

        cmaperr = matplotlib.colormaps['jet']
        cmaperr.set_bad('white',alpha=0.0)

        # make the mode energy crossplot
        fig = plot.figure(figsize=(6.69,7.5))
        plot.subplots_adjust(left=0.05,top=0.99,right=0.9,bottom=0.05)
        outer = gridspec.GridSpec(3,3,wspace=0.05,hspace=0.1,height_ratios=[1,1,1],width_ratios=[0.45,0.45,0.05])
         

        cbar_ticks = [1E-4,1E-3,1E-2,1E-1,1]
        xy_ticks = [-2,0,2,]
        xy_minor_ticks = [-3,-1,1,3,]

        ax = plot.Subplot(fig,outer[0])
        ax.set_aspect('equal')
        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(aa)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[1])
        ax.set_aspect('equal')
        p_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ba)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        #ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[3])
        ax.set_aspect('equal')
        p_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_pred_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_pred_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ab)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[4])
        ax.set_aspect('equal')
        p_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_pred_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_pred_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(bb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[6])
        ax.set_aspect('equal')
        p_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,np.abs(err_E_ij_real),cmap=cmaperr,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        #n_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_pred_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(ca)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[7])
        ax.set_aspect('equal')
        p_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,np.abs(err_E_ij_imag),cmap=cmaperr,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        #n_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_pred_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(cb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)



        inner = []  
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[0][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[0][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[1][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[1][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[8],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[2][0])
        cbar = plot.colorbar(p_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=True)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        #cax=plot.Subplot(fig,inner[2][2])
        #cbar = plot.colorbar(n_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        #ticklabs = cbar.ax.get_yticklabels()
        #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        #fig.add_subplot(cax)



        filename = 'simplified_energy_exchange_error_PINN_S'+str(cases_supersample_factor[s])
        plot.savefig(figures_dir+filename+'.pdf')
        plot.savefig(figures_dir+filename+'.png',dpi=300)
        plot.close(fig)

    mean_error_ds_real = np.zeros([len(supersample_factors_ds),])
    max_error_ds_real = np.zeros([len(supersample_factors_ds),])
    mean_error_ds_imag = np.zeros([len(supersample_factors_ds),])
    max_error_ds_imag = np.zeros([len(supersample_factors_ds),])
    mean_error_ds = np.zeros([len(supersample_factors_ds),])
    max_error_ds = np.zeros([len(supersample_factors_ds),])
    
    for s in range(len(supersample_factors_ds)):


        # the downsampled
        freq_indices = np.arange(-n_LOM,n_LOM+1)
        freq_mesh_i,freq_mesh_j = np.meshgrid(freq_indices,freq_indices)

        temp_E_ij_real_pos = np.real(1.0*E_ij_LOM_matrix)
        temp_E_ij_real_pos[temp_E_ij_real_pos<0]=np.NaN
        temp_E_ij_real_pos[(temp_E_ij_real_pos>0)*(temp_E_ij_real_pos<1E-4)]=0.0
        temp_E_ij_real_neg = np.real(1.0*E_ij_LOM_matrix)
        temp_E_ij_real_neg[temp_E_ij_real_neg>=0]=np.NaN
        temp_E_ij_real_neg = np.abs(temp_E_ij_real_neg)
        temp_E_ij_real_neg[(temp_E_ij_real_neg>0)*(temp_E_ij_real_neg<1E-4)]=0.0

        temp_E_ij_imag_pos = np.imag(1.0*E_ij_LOM_matrix)
        temp_E_ij_imag_pos[temp_E_ij_imag_pos<0]=np.NaN
        temp_E_ij_imag_pos[(temp_E_ij_imag_pos>0)*(temp_E_ij_imag_pos<1E-4)]=0.0
        temp_E_ij_imag_neg = np.imag(1.0*E_ij_LOM_matrix)
        temp_E_ij_imag_neg[temp_E_ij_imag_neg>=0]=np.NaN
        temp_E_ij_imag_neg = np.abs(temp_E_ij_imag_neg)
        temp_E_ij_imag_neg[(temp_E_ij_imag_neg>0)*(temp_E_ij_imag_neg<1E-4)]=0.0

        temp_E_ij_ds_real_pos = np.real(1.0*E_ij_ds_LOM_matrix[:,:,s])
        temp_E_ij_ds_real_pos[temp_E_ij_ds_real_pos<0]=np.NaN
        temp_E_ij_ds_real_pos[(temp_E_ij_ds_real_pos>0)*(temp_E_ij_ds_real_pos<1E-4)]=0.0
        temp_E_ij_ds_real_neg = np.real(1.0*E_ij_ds_LOM_matrix[:,:,s])
        temp_E_ij_ds_real_neg[temp_E_ij_ds_real_neg>=0]=np.NaN
        temp_E_ij_ds_real_neg = np.abs(temp_E_ij_ds_real_neg)
        temp_E_ij_ds_real_neg[(temp_E_ij_ds_real_neg>0)*(temp_E_ij_ds_real_neg<1E-4)]=0.0

        temp_E_ij_ds_imag_pos = np.imag(1.0*E_ij_ds_LOM_matrix[:,:,s])
        temp_E_ij_ds_imag_pos[temp_E_ij_ds_imag_pos<0]=np.NaN
        temp_E_ij_ds_imag_pos[(temp_E_ij_ds_imag_pos>0)*(temp_E_ij_ds_imag_pos<1E-4)]=0.0
        temp_E_ij_ds_imag_neg = np.imag(1.0*E_ij_ds_LOM_matrix[:,:,s])
        temp_E_ij_ds_imag_neg[temp_E_ij_ds_imag_neg>=0]=np.NaN
        temp_E_ij_ds_imag_neg = np.abs(temp_E_ij_ds_imag_neg)
        temp_E_ij_ds_imag_neg[(temp_E_ij_ds_imag_neg>0)*(temp_E_ij_ds_imag_neg<1E-4)]=0.0

        err_E_ij_real = (np.real(E_ij_LOM_matrix)-np.real(E_ij_ds_LOM_matrix[:,:,s]))/np.nanmax(np.abs(np.real(E_ij_LOM_matrix)),axis=0)
        err_E_ij_imag = (np.imag(E_ij_LOM_matrix)-np.imag(E_ij_ds_LOM_matrix[:,:,s]))/np.nanmax(np.abs(np.imag(E_ij_LOM_matrix)),axis=0)
        err_E_ij = np.abs(E_ij_LOM_matrix-E_ij_ds_LOM_matrix[:,:,s])/np.nanmax(np.abs(E_ij_LOM_matrix),axis=0)

        mean_error_ds_real[s]=np.nanmean(np.abs(err_E_ij_real))
        max_error_ds_real[s] = np.nanmax(np.abs(err_E_ij_real))
        mean_error_ds_imag[s]=np.nanmean(np.abs(err_E_ij_imag))
        max_error_ds_imag[s] = np.nanmax(np.abs(err_E_ij_imag))

        mean_error_ds[s]=np.nanmean(err_E_ij)
        max_error_ds[s] = np.nanmax(err_E_ij)

        #err_E_ij_ds_real_pos = 1.0*err_E_ij_real
        #err_E_ij_ds_real_pos[err_E_ij_ds_real_pos<0]=np.NaN
        #err_E_ij_ds_real_pos[(err_E_ij_ds_real_pos>0)*(err_E_ij_ds_real_pos<1E-4)]=0.0
        #err_E_ij_ds_real_neg = 1.0*err_E_ij_real
        #err_E_ij_ds_real_neg[err_E_ij_ds_real_neg>=0]=np.NaN
        #err_E_ij_ds_real_neg = np.abs(err_E_ij_ds_real_neg)
        #err_E_ij_ds_real_neg[(err_E_ij_ds_real_neg>0)*(err_E_ij_ds_real_neg<1E-4)]=0.0

        #err_E_ij_ds_imag_pos = 1.0*err_E_ij_imag
        #err_E_ij_ds_imag_pos[err_E_ij_ds_imag_pos<0]=np.NaN
        #err_E_ij_ds_imag_pos[(err_E_ij_ds_imag_pos>0)*(err_E_ij_ds_imag_pos<1E-4)]=0.0
        #err_E_ij_ds_imag_neg = 1.0*err_E_ij_imag
        #err_E_ij_ds_imag_neg[err_E_ij_ds_imag_neg>=0]=np.NaN
        #err_E_ij_ds_imag_neg = np.abs(err_E_ij_ds_imag_neg)
        #err_E_ij_ds_imag_neg[(err_E_ij_ds_imag_neg>0)*(err_E_ij_ds_imag_neg<1E-4)]=0.0
        
        

        MAX_E_ij = np.nanmax([np.nanmax(temp_E_ij_real_pos.ravel()),np.nanmax(temp_E_ij_real_neg.ravel()),np.nanmax(temp_E_ij_imag_pos.ravel()),np.nanmax(temp_E_ij_imag_neg.ravel()),np.nanmax(temp_E_ij_ds_real_pos.ravel()),np.nanmax(temp_E_ij_ds_real_neg.ravel()),np.nanmax(temp_E_ij_ds_imag_pos.ravel()),np.nanmax(temp_E_ij_ds_imag_neg.ravel())]) 

        #MAX_err_E_ij = np.nanmax([np.nanmax(err_E_ij_ds_real_pos.ravel()),np.nanmax(err_E_ij_ds_real_neg.ravel()),np.nanmax(err_E_ij_ds_imag_pos.ravel()),np.nanmax(err_E_ij_ds_imag_neg.ravel()),])
        MAX_err_E_ij = np.nanmax([np.nanmax(np.abs(err_E_ij_real.ravel())),np.nanmax(np.abs(np.abs(err_E_ij_imag.ravel())))])

        cmap1 = matplotlib.colormaps['Reds']
        cmap1.set_bad('white',alpha=0.0)
        cmap2 = matplotlib.colormaps['Blues']
        cmap2.set_bad('white',alpha=0.0)

        cmaperr = matplotlib.colormaps['jet']
        cmaperr.set_bad('white',alpha=0.0)

        # make the mode energy crossplot
        fig = plot.figure(figsize=(6.69,7.5))
        plot.subplots_adjust(left=0.05,top=0.99,right=0.9,bottom=0.05)
        outer = gridspec.GridSpec(3,3,wspace=0.05,hspace=0.1,height_ratios=[1,1,1],width_ratios=[0.45,0.45,0.05])
         

        cbar_ticks = [1E-4,1E-3,1E-2,1E-1,1]
        xy_ticks = [-2,0,2,]
        xy_minor_ticks = [-3,-1,1,3,]

        ax = plot.Subplot(fig,outer[0])
        ax.set_aspect('equal')
        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(aa)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[1])
        ax.set_aspect('equal')
        p_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot2 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ba)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        #ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[3])
        ax.set_aspect('equal')
        p_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_real_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot3 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(ab)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[4])
        ax.set_aspect('equal')
        p_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_imag_pos,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        n_plot4 = ax.pcolor(freq_mesh_i,freq_mesh_j,temp_E_ij_ds_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmax=MAX_E_ij))
        ax.text(-6,6,'(bb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        #ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[6])
        ax.set_aspect('equal')
        p_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,np.abs(err_E_ij_real),cmap=cmaperr,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        #n_plot5 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_real_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(ca)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f/f_{vs}$',fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)

        ax = plot.Subplot(fig,outer[7])
        ax.set_aspect('equal')
        p_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,np.abs(err_E_ij_imag),cmap=cmaperr,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        #n_plot6 = ax.pcolor(freq_mesh_i,freq_mesh_j,err_E_ij_ds_imag_neg,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-6,6,'(cb)',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_xlabel('$f/f_{vs}$',fontsize=8)
        fig.add_subplot(ax)



        inner = []  
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[0][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[0][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[5],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[1][0])
        cbar = plot.colorbar(p_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=False)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        cax=plot.Subplot(fig,inner[1][2])
        cbar = plot.colorbar(n_plot,cax,ticks=cbar_ticks,format=tkr.FormatStrFormatter('%.0e'))
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[8],wspace=0.15,hspace=0.3,width_ratios=[0.02,0.01,0.02]))
        cax=plot.Subplot(fig,inner[2][0])
        cbar = plot.colorbar(p_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        cbar.ax.yaxis.set_tick_params(labelright=True)
        #ticklabs = cbar.ax.get_xticklabels()
        #cbar.ax.set_xticklabels(ticklabs, fontsize=8)
        fig.add_subplot(cax)


        #cax=plot.Subplot(fig,inner[2][2])
        #cbar = plot.colorbar(n_plot5,cax,format=tkr.FormatStrFormatter('%.0e'))
        #ticklabs = cbar.ax.get_yticklabels()
        #cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        #fig.add_subplot(cax)



        filename = 'simplified_energy_exchange_error_ds_S'+str(supersample_factors_ds[s])
        plot.savefig(figures_dir+filename+'.pdf')
        plot.savefig(figures_dir+filename+'.png',dpi=300)
        plot.close(fig)

    # combined plot with both DS and PINN
        freq_indices = np.arange(-n_LOM,n_LOM+1)
        freq_mesh_i,freq_mesh_j = np.meshgrid(freq_indices,freq_indices)

        from matplotlib.colors import LinearSegmentedColormap

        cdict3 = {'red':   [(0.0,  1.0, 1.0), # white
                            (1.0,  1.0, 1.0)], # red 

                'green': [(0.0,  0.0, 1.0),
                            (1.0,  0.0, 0.0)],

                'blue':  [(0.0,  0.0, 1.0),
                            (1.0,  0.0, 0.0)]}


        cmap1 = LinearSegmentedColormap('WhiteRed',cdict3)
        cmap1.set_bad('white',alpha=0.0)

        
        cdict4 = {'red':   [(0.0,  1.0, 1.0),
                            (1.0,  0.0, 0.0)],

                'green': [(0.0,  0.0, 1.0),
                            (1.0,  0.0, 0.0)],

                'blue':  [(0.0,  0.0, 1.0),
                            (1.0,  1.0, 1.0)]}

        cmap2 = LinearSegmentedColormap('WhiteBlue',cdict4)
        cmap2.set_bad('white',alpha=0.0)
        
        cdict5 = {'red':   [(0.0,  0.0, 0.0),
                            (0.5,  1.0, 1.0), # white
                            (1.0,  1.0, 1.0)], # red 

                'green': [(0.0,  0.0, 0.0),
                            (0.5,  1.0, 1.0),
                            (1.0,  0.0, 0.0)],

                'blue':  [(0.0,  1.0, 1.0),
                            (0.5,  1.0, 1.0),
                            (1.0,  0.0, 0.0)]}
        # for the colobars only
        cmap3 = LinearSegmentedColormap('BlueWhiteRed',cdict5)
        cmap3.set_bad('white',alpha=0.0)

        cdict13 = {'red':   [(0.0,  1.0, 1.0), # white
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


        cmap11 = LinearSegmentedColormap('WhiteYellowPinkRed',cdict13)
        cmap1.set_bad('white',alpha=0.0)

        
        cdict14 = {'red':   [(0.0,  1.0, 1.0),
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

        cmap12 = LinearSegmentedColormap('WhiteGreenCyanBlue',cdict14)
        cmap12.set_bad('white',alpha=0.0)
        
        cdict15 = {'red':   [(0.0,  0.0, 0.0),
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
        cmap13 = LinearSegmentedColormap('BlueCyanGreenWhiteYellowPinkRed',cdict15)
        cmap13.set_bad('white',alpha=0.0)

        dual_log_cbar_norm = matplotlib.colors.CenteredNorm(0.0,1.0)
        dual_log_cbar_ticks = [-1,-0.5,0,0.5,1]
        dual_log_cbar_labels = ['-1','-1e-2','$\pm$1e-4','1e-2','1']

        err_log_cbar_norm = matplotlib.colors.CenteredNorm(0.0,1.0)
        err_log_cbar_ticks = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
        err_log_cbar_labels = ['-1','-1e-1','-1e-2','-1e-3','$\pm$1e-4','1e-3','1e-2','1e-1','1']



        MAX_E_ij = np.nanmax(np.abs(E_ij_LOM_matrix.ravel()))
        print(MAX_E_ij)

        # make the mode energy crossplot
        fig = plot.figure(figsize=(3.37,6.5))
        plot.subplots_adjust(left=0.1,top=0.99,right=0.85,bottom=0.07)
        outer = gridspec.GridSpec(5,1,wspace=0.05,hspace=0.1,height_ratios=[1,1,1,1,1])
        inner = []

        xy_ticks = [-2,0,2,]
        xy_minor_ticks = [-3,-1,1,3,]

        ylabelpad = -5
        xlabelpad = 0       

        # reference
        e_plot = np.real(E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=[0.48,0.48,0.04]))
        ax = plot.Subplot(fig,inner[0][0])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(a)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f_{j}/f_{1}$',fontsize=8,labelpad=ylabelpad)
        fig.add_subplot(ax)

        e_plot = np.imag(E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        ax = plot.Subplot(fig,inner[0][1])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(b)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[0][2])
        cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
        cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
        fig.add_subplot(cax)

        # downsampled
        e_plot = np.real(E_ij_ds_LOM_matrix[:,:,s])/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=[0.48,0.48,0.04]))
        ax = plot.Subplot(fig,inner[1][0])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(c)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f_{j}/f_{1}$',fontsize=8,labelpad=ylabelpad)
        fig.add_subplot(ax)

        e_plot = np.imag(E_ij_ds_LOM_matrix[:,:,s])/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        ax = plot.Subplot(fig,inner[1][1])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(d)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[1][2])
        cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
        cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
        fig.add_subplot(cax)

        # PINN
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=[0.48,0.48,0.04]))
        ax = plot.Subplot(fig,inner[2][0])
        ax.set_aspect('equal')

        e_plot = np.real(E_ij_pred_matrix[:,:,s])/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(e)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f_{j}/f_{1}$',fontsize=8,labelpad=ylabelpad)
        fig.add_subplot(ax)

        e_plot = np.imag(E_ij_pred_matrix[:,:,s])/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        ax = plot.Subplot(fig,inner[2][1])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap1,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap2,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(f)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[2][2])
        cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap3),cax,ticks=dual_log_cbar_ticks,extend='both')
        cbar.ax.set_yticklabels(dual_log_cbar_labels, fontsize=8)
        fig.add_subplot(cax)



        # err DS
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=[0.48,0.48,0.04]))
        ax = plot.Subplot(fig,inner[3][0])
        ax.set_aspect('equal')

        e_plot = np.real(E_ij_ds_LOM_matrix[:,:,s]-E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap11,norm=matplotlib.colors.LogNorm(vmin=1E-6,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap12,norm=matplotlib.colors.LogNorm(vmin=1E-6,vmax=1))
        ax.text(-3,2.5,'(g)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f_{j}/f_{1}$',fontsize=8,labelpad=ylabelpad)
        fig.add_subplot(ax)

        e_plot = np.imag(E_ij_ds_LOM_matrix[:,:,s]-E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        ax = plot.Subplot(fig,inner[3][1])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap11,norm=matplotlib.colors.LogNorm(vmin=1E-6,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap12,norm=matplotlib.colors.LogNorm(vmin=1E-6,vmax=1))
        ax.text(-3,2.5,'(h)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[3][2])
        cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap13),cax,ticks=err_log_cbar_ticks,extend='both')
        cbar.ax.set_yticklabels(err_log_cbar_labels, fontsize=8)
        fig.add_subplot(cax)

        # err PINN
        inner.append(gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=[0.48,0.48,0.04]))
        ax = plot.Subplot(fig,inner[4][0])
        ax.set_aspect('equal')

        e_plot = np.real(E_ij_pred_matrix[:,:,s]-E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap11,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap12,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(i)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ticklabs = ax.get_yticklabels()
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        ticklabs = ax.get_xticklabels()
        ax.set_xticklabels(ticklabs, fontsize=8)
        ax.set_ylabel('$f_{j}/f_{1}$',fontsize=8,labelpad=ylabelpad)
        ax.set_xlabel('$f_{i}/f_{1}$',fontsize=8,labelpad=xlabelpad)
        fig.add_subplot(ax)

        e_plot = np.imag(E_ij_pred_matrix[:,:,s]-E_ij_LOM_matrix)/MAX_E_ij
        e_plot_p =e_plot+1E-30
        e_plot_p[e_plot_p<=0]=np.NaN
        e_plot_n = e_plot
        e_plot_n[e_plot_n>0]=np.NaN
        e_plot_n = np.abs(e_plot_n)

        ax = plot.Subplot(fig,inner[4][1])
        ax.set_aspect('equal')

        p_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_p,cmap=cmap11,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        n_plot = ax.pcolor(freq_mesh_i,freq_mesh_j,e_plot_n,cmap=cmap12,norm=matplotlib.colors.LogNorm(vmin=1E-4,vmax=1))
        ax.text(-3,2.5,'(j)',fontsize=8)
        #ax.text(-0.5,-3.4,'$R(\Delta E_{ij,DNS})$',fontsize=8)
        ax.set_yticks(xy_ticks)
        ax.set_yticks(xy_minor_ticks,minor=True)
        ax.set_xticks(xy_ticks)
        ax.set_xticks(xy_minor_ticks,minor=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xlabel('$f_{i}/f_{1}$',fontsize=8,labelpad=xlabelpad)
        fig.add_subplot(ax)

        cax=plot.Subplot(fig,inner[4][2])
        cbar = plot.colorbar(matplotlib.cm.ScalarMappable(norm=dual_log_cbar_norm, cmap=cmap13),cax,ticks=err_log_cbar_ticks,extend='both')
        cbar.ax.set_yticklabels(err_log_cbar_labels, fontsize=8)
        fig.add_subplot(cax)




        filename = 'simplified_energy_exchange_error_ds_PINN_S'+str(supersample_factors_ds[s])
        plot.savefig(figures_dir+filename+'.pdf')
        plot.savefig(figures_dir+filename+'.png',dpi=300)
        plot.close(fig)


    # error summary plot

    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,3.37)
    plot.subplots_adjust(left=0.17,top=0.95,right=0.9,bottom=0.15)

    y_ticks=[1E-5,1E-4,1E-3,1E-2,1E-1,1]
    y_tick_labels=['1E-5','1E-4','1E-3','1E-2','1E-1','1']
    error_x_tick_labels = ['40','20','10','5','2.5','1.25',]

    mean_plt,=axs.plot(pts_per_d[1:]*0.97,mean_error_ds_real,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    mean_plt2,=axs.plot(pts_per_d[1:]*1.03,mean_error_ds_imag,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
    mean_plt3,=axs.plot(pts_per_d*0.97,mean_error_PINN_real,linewidth=0,marker='*',color='black',markersize=3,markerfacecolor='black')
    mean_plt4,=axs.plot(pts_per_d*1.03,mean_error_PINN_imag,linewidth=0,marker='*',color='black',markersize=3,markerfacecolor='black')
    max_plt,=axs.plot(pts_per_d[1:]*0.97,max_error_ds_real,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    max_plt2,=axs.plot(pts_per_d[1:]*1.03,max_error_ds_imag,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    max_plt3,=axs.plot(pts_per_d*0.97,max_error_PINN_real,linewidth=0,marker='^',color='red',markersize=3,markerfacecolor='red')
    max_plt4,=axs.plot(pts_per_d*1.03,max_error_PINN_imag,linewidth=0,marker='^',color='red',markersize=3,markerfacecolor='red')
    axs.set_xscale('log')
    axs.xaxis.set_tick_params(labelbottom=True)
    axs.set_yscale('log')
    axs.set_ylim(1E-5,1E1)
    axs.set_yticks(y_ticks,labels=y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    #axs.set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs.legend([mean_plt,mean_plt3,max_plt,max_plt3],['Mean DS','Mean PINN','Max DS','Max PINN'],fontsize=8,loc='upper center',ncol=2)
    axs.grid('on')
    axs.set_xticks(pts_per_d)
    axs.set_xticklabels(error_x_tick_labels)
    axs.set_xlabel('$D/\Delta x$',fontsize=8)
    #axs[0].text(0.45,10.0,'(a)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'energy_PINN_error.pdf')
    plot.savefig(figures_dir+'energy_PINN_error.png',dpi=300)
    plot.close(fig)

    fig,axs = plot.subplots(1,1)
    fig.set_size_inches(3.37,2.5)
    plot.subplots_adjust(left=0.17,top=0.99,right=0.9,bottom=0.17)

    y_ticks=[1E-4,1E-3,1E-2,1E-1,1]
    y_tick_labels=['1E-4','1E-3','1E-2','1E-1','1']
    error_x_tick_labels = ['40','20','10','5','2.5','1.25',]

    mean_plt,=axs.plot(pts_per_d[1:]*0.97,mean_error_ds,linewidth=0,marker='o',color='blue',markersize=3,markerfacecolor='blue')
    mean_plt2,=axs.plot(pts_per_d*1.03,mean_error_PINN,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
    max_plt,=axs.plot(pts_per_d[1:]*0.97,max_error_ds,linewidth=0,marker='v',color='blue',markersize=3,markerfacecolor='blue')
    max_plt2,=axs.plot(pts_per_d*1.03,max_error_PINN,linewidth=0,marker='v',color='red',markersize=3,markerfacecolor='red')
    axs.set_xscale('log')
    axs.xaxis.set_tick_params(labelbottom=True)
    axs.set_yscale('log')
    axs.set_ylim(1E-4,1E1)
    axs.set_yticks(y_ticks,labels=y_tick_labels,fontsize=8)
    axs.set_ylabel("Relative Error",fontsize=8)
    #axs.set_title('$\overline{u\'_{x}u\'_{x}}$')
    axs.legend([mean_plt,mean_plt2,max_plt,max_plt2],['Mean DS','Mean PINN','Max DS','Max PINN'],fontsize=8,loc='upper center',ncol=2)
    axs.grid('on')
    axs.set_xticks(pts_per_d)
    axs.set_xticklabels(error_x_tick_labels)
    axs.set_xlabel('$D/\Delta x$',fontsize=8)
    #axs[0].text(0.45,10.0,'(a)',fontsize=10)

    #fig.tight_layout()
    plot.savefig(figures_dir+'energy_PINN_error_mag.pdf')
    plot.savefig(figures_dir+'energy_PINN_error_mag.png',dpi=300)
    plot.close(fig)



exit()
# spatial error plots for the different energy terms

for s in range(len(supersample_factors_ds)):

    for i in range(-n_LOM,n_LOM+1):
        for j in range(-n_LOM,n_LOM+1):
            if (i-j)>=n_LOM or (i-j)<=-n_LOM:
                # plot nothing, invalid frequency
                pass
            else:
                print(i)
                print(j)
                percent_energy_real = np.abs(np.real(E_ij_LOM_matrix[i+n_LOM,j+n_LOM]))/np.nansum(np.abs(np.real(E_ij_LOM_matrix)))
                percent_energy_imag = np.abs(np.imag(E_ij_LOM_matrix[i+n_LOM,j+n_LOM]))/np.nansum(np.abs(np.imag(E_ij_LOM_matrix)))


                c_ref_real = np.real(e_ij_LOM_matrix[:,:,i+n_LOM,j+n_LOM])
                c_ref_imag = np.imag(e_ij_LOM_matrix[:,:,i+n_LOM,j+n_LOM])
                #c_ref_real[cylinder_mask] = np.NaN
                #c_ref_imag[cylinder_mask] = np.NaN

                MAX_c_ref_real = np.nanmax(np.abs(c_ref_real.ravel()))
                MAX_c_ref_imag = np.nanmax(np.abs(c_ref_imag.ravel()))

                c_ds_real = np.real(e_ij_ds_LOM_matrix[s][:,:,i+n_LOM,j+n_LOM])
                c_ds_imag = np.imag(e_ij_ds_LOM_matrix[s][:,:,i+n_LOM,j+n_LOM])

                MAX_c_ds_real = np.nanmax(np.abs(c_ds_real.ravel()))
                MAX_c_ds_imag = np.nanmax(np.abs(c_ds_imag.ravel()))

                x_ds_grid_plot = x_ds_grid[s+1]
                y_ds_grid_plot = y_ds_grid[s+1]
                ds_inds_plot = ds_inds[s+1]
                
                x_dots_plot_ref = x_ds_grid_plot
                y_dots_plot_ref = y_ds_grid_plot
                dots_mask = np.power(np.power(x_dots_plot_ref,2.0)+np.power(y_dots_plot_ref,2.0),0.5)>0.5
                x_dots_plot_ref = x_dots_plot_ref[dots_mask]
                y_dots_plot_ref = y_dots_plot_ref[dots_mask]
                dots_mask_2 = (np.multiply(x_dots_plot_ref>5.5,y_dots_plot_ref>0.9)<1)
                x_dots_plot_ref = x_dots_plot_ref[dots_mask_2]
                y_dots_plot_ref = y_dots_plot_ref[dots_mask_2]
                dots_mask_3 = (np.multiply(x_dots_plot_ref>2.8,y_dots_plot_ref<-1)<1)
                x_dots_plot_ref = x_dots_plot_ref[dots_mask_3]
                y_dots_plot_ref = y_dots_plot_ref[dots_mask_3]
                dots_mask_4 = (np.multiply(x_dots_plot_ref<-0.5,y_dots_plot_ref>1.2)<1)
                x_dots_plot_ref = x_dots_plot_ref[dots_mask_4]
                y_dots_plot_ref = y_dots_plot_ref[dots_mask_4]

                x_dots_plot_err = x_ds_grid_plot
                y_dots_plot_err = y_ds_grid_plot
                dots_mask = np.power(np.power(x_dots_plot_err,2.0)+np.power(y_dots_plot_err,2.0),0.5)>0.5
                x_dots_plot_err = x_dots_plot_err[dots_mask]
                y_dots_plot_err = y_dots_plot_err[dots_mask]
                dots_mask_2 = (np.multiply(x_dots_plot_err>4.8,y_dots_plot_err>0.9)<1)
                x_dots_plot_err = x_dots_plot_err[dots_mask_2]
                y_dots_plot_err = y_dots_plot_err[dots_mask_2]
                dots_mask_3 = (np.multiply(x_dots_plot_err<-0.5,y_dots_plot_err>1.2)<1)
                x_dots_plot_err = x_dots_plot_err[dots_mask_3]
                y_dots_plot_err = y_dots_plot_err[dots_mask_3]

                cylinder_mask_ds = np.reshape((cylinder_mask.ravel())[ds_inds_plot],[x_ds_grid_plot.shape[1],x_ds_grid_plot.shape[0]]).transpose()
                #c_ds_real[cylinder_mask_ds] = np.NaN
                #c_ds_imag[cylinder_mask_ds] = np.NaN

                c_pinn_real = np.real(e_ij_pred_matrix[:,:,i+n_LOM,j+n_LOM,s+1])
                c_pinn_imag = np.imag(e_ij_pred_matrix[:,:,i+n_LOM,j+n_LOM,s+1])
                #c_pinn_real[cylinder_mask]=np.NaN
                #c_pinn_imag[cylinder_mask]=np.NaN

                MAX_c_pinn_real = np.nanmax(np.abs(c_pinn_real.ravel()))
                MAX_c_pinn_imag = np.nanmax(np.abs(c_pinn_imag.ravel()))

                MAX_c_real = np.nanmax([MAX_c_ref_real,MAX_c_ds_real,MAX_c_pinn_real])
                MAX_c_imag = np.nanmax([MAX_c_ref_imag,MAX_c_ds_imag,MAX_c_pinn_imag])
                if i==0 and j==0:
                    MAX_c_imag = 1.0 # needs to be nonzero even tho the imaginary part of mode 0 is zero
                    MAX_c_ref_imag=1.0
                levels_c_real = np.linspace(-MAX_c_real,MAX_c_real,21)
                levels_c_imag = np.linspace(-MAX_c_imag,MAX_c_imag,21)

                c_ref_ds_real = np.reshape((c_ref_real.ravel())[ds_inds_plot],[x_ds_grid_plot.shape[1],x_ds_grid_plot.shape[0]]).transpose()
                c_ref_ds_imag = np.reshape((c_ref_imag.ravel())[ds_inds_plot],[x_ds_grid_plot.shape[1],x_ds_grid_plot.shape[0]]).transpose()

                c_ds_err_real = (c_ds_real - c_ref_ds_real)/MAX_c_ref_real
                c_ds_err_imag = (c_ds_imag - c_ref_ds_imag)/MAX_c_ref_imag

                MAX_c_ds_err_real = np.nanmax(np.abs(c_ds_err_real.ravel()))
                levels_c_ds_err_real = np.linspace(-MAX_c_ds_err_real,MAX_c_ds_err_real,21)
                MAX_c_ds_err_imag = np.nanmax(np.abs(c_ds_err_imag.ravel()))
                if i==0 and j==0:
                    MAX_c_ds_err_imag = 1.0
                levels_c_ds_err_imag = np.linspace(-MAX_c_ds_err_imag,MAX_c_ds_err_imag,21)

                c_pinn_err_real = (c_pinn_real - c_ref_real)/MAX_c_ref_real
                c_pinn_err_imag = (c_pinn_imag - c_ref_imag)/MAX_c_ref_imag

                MAX_c_pinn_err_real = np.nanmax(np.abs(c_pinn_err_real.ravel()))
                levels_c_pinn_err_real = np.linspace(-MAX_c_pinn_err_real,MAX_c_pinn_err_real,21)
                MAX_c_pinn_err_imag = np.nanmax(np.abs(c_pinn_err_imag.ravel()))
                if i==0 and j==0:
                    MAX_c_pinn_err_imag = 1.0
                levels_c_pinn_err_imag = np.linspace(-MAX_c_pinn_err_imag,MAX_c_pinn_err_imag,21)

                fig = plot.figure(figsize=(6.69,5))
                plot.subplots_adjust(left=0.05,top=0.97,right=0.99,bottom=0.07)
                outer = gridspec.GridSpec(5,1,wspace=0.1,hspace=0.1)
                inner = []

                width_ratios = [0.4,0.01,0.09,0.4,0.01,0.09]
                y_ticks = [-2,0,2]
                x_ticks = [-2,0,2,4,6,8,10]
                c_ticks_real = [MAX_c_real,0.5*MAX_c_real,0.0,-0.5*MAX_c_real,-1.0*MAX_c_real]
                c_ticks_imag = [MAX_c_imag,0.5*MAX_c_imag,0.0,-0.5*MAX_c_imag,-1.0*MAX_c_imag]

                inner.append(gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=outer[0],wspace=0.05,hspace=0.1,width_ratios=width_ratios))
                ax = plot.Subplot(fig,inner[0][0])
                c_ref_real_plot = ax.contourf(X_grid,Y_grid,c_ref_real,levels=levels_c_real,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.xaxis.set_tick_params(labelbottom=False)
                dots = ax.plot(x_dots_plot_ref,y_dots_plot_ref,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ticklabs = ax.get_yticklabels()
                ax.set_yticklabels(ticklabs, fontsize=8)
                ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
                ax.text(-1.8,1.4,'(aa)',fontsize=8)
                ax.text(6,1.2,'$R(\Delta e_{'+str(i)+','+str(j)+',DNS})$',fontsize=8)
                ax.text(3,-1.6,str(round(100*percent_energy_real,1))+'% of $\Sigma |R(\Delta E_{ij,DNS})|$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[0][1])
                cbar = plot.colorbar(c_ref_real_plot,cax,ticks=c_ticks_real,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[0][3])
                c_ref_imag_plot = ax.contourf(X_grid,Y_grid,c_ref_imag,levels=levels_c_imag,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_aspect('equal')
                dots = ax.plot(x_dots_plot_ref,y_dots_plot_ref,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.text(-1.8,1.4,'(ba)',fontsize=8)
                ax.text(6,1.2,'$Im(\Delta e_{'+str(i)+','+str(j)+',DNS})$',fontsize=8)
                ax.text(3,-1.6,str(round(100*percent_energy_imag,1))+'% of $\Sigma |Im(\Delta E_{ij,DNS})|$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[0][4])
                cbar = plot.colorbar(c_ref_imag_plot,cax,ticks=c_ticks_imag,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                inner.append(gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=outer[1],wspace=0.05,hspace=0.1,width_ratios=width_ratios))
                ax = plot.Subplot(fig,inner[1][0])
                c_ds_real_plot = ax.contourf(x_ds_grid_plot,y_ds_grid_plot,c_ds_real,levels=levels_c_real,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.xaxis.set_tick_params(labelbottom=False)
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
                ticklabs = ax.get_yticklabels()
                ax.set_yticklabels(ticklabs, fontsize=8)
                ax.text(-1.8,1.4,'(ab)',fontsize=8)
                ax.text(6,1.2,'$R(\Delta e_{'+str(i)+','+str(j)+',DS})$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[1][1])
                cbar = plot.colorbar(c_ds_real_plot,cax,ticks=c_ticks_real,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[1][3])
                c_ds_imag_plot = ax.contourf(x_ds_grid_plot,y_ds_grid_plot,c_ds_imag,levels=levels_c_imag,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_aspect('equal')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.text(-1.8,1.4,'(bb)',fontsize=8)
                ax.text(6,1.2,'$Im(\Delta e_{'+str(i)+','+str(j)+',DS})$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[1][4])
                cbar = plot.colorbar(c_ds_imag_plot,cax,ticks=c_ticks_imag,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                inner.append(gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=outer[2],wspace=0.05,hspace=0.1,width_ratios=width_ratios))
                ax = plot.Subplot(fig,inner[2][0])
                c_pinn_real_plot = ax.contourf(X_grid,Y_grid,c_pinn_real,levels=levels_c_real,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.xaxis.set_tick_params(labelbottom=False)
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
                ticklabs = ax.get_yticklabels()
                ax.set_yticklabels(ticklabs, fontsize=8)
                ax.text(-1.8,1.4,'(ac)',fontsize=8)
                ax.text(6,1.2,'$R(\Delta e_{'+str(i)+','+str(j)+',PINN})$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[2][1])
                cbar = plot.colorbar(c_pinn_real_plot,cax,ticks=c_ticks_real,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[2][3])
                c_pinn_imag_plot = ax.contourf(X_grid,Y_grid,c_pinn_imag,levels=levels_c_imag,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_aspect('equal')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.text(-1.8,1.4,'(bc)',fontsize=8)
                ax.text(6,1.2,'$Im(\Delta e_{'+str(i)+','+str(j)+',PINN})$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[2][4])
                cbar = plot.colorbar(c_pinn_imag_plot,cax,ticks=c_ticks_imag,format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                inner.append(gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=outer[3],wspace=0.05,hspace=0.1,width_ratios=width_ratios))
                ax = plot.Subplot(fig,inner[3][0])
                c_ds_err_real_plot = ax.contourf(x_ds_grid_plot,y_ds_grid_plot,c_ds_err_real,levels=levels_c_ds_err_real,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                ax.xaxis.set_tick_params(labelbottom=False)
                dots = ax.plot(x_dots_plot_err,y_dots_plot_err,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ticklabs = ax.get_yticklabels()
                ax.set_yticklabels(ticklabs, fontsize=8)
                ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
                ax.text(-1.8,1.4,'(ad)',fontsize=8)
                ax.text(5,1.2,'$\\frac{R(\Delta e_{'+str(i)+','+str(j)+',DS}-\Delta e_{'+str(i)+','+str(j)+',DNS})}{max(R(\Delta e_{'+str(i)+','+str(j)+',DNS}))}$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[3][1])
                cbar = plot.colorbar(c_ds_err_real_plot,cax,ticks=[MAX_c_ds_err_real,MAX_c_ds_err_real/2,0.0,-MAX_c_ds_err_real/2,-MAX_c_ds_err_real],format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[3][3])
                c_ds_err_imag_plot = ax.contourf(x_ds_grid_plot,y_ds_grid_plot,c_ds_err_imag,levels=levels_c_ds_err_imag,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_aspect('equal')
                dots = ax.plot(x_dots_plot_err,y_dots_plot_err,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.text(-1.8,1.4,'(bd)',fontsize=8)
                ax.text(5,1.2,'$\\frac{Im(\Delta e_{'+str(i)+','+str(j)+',DS}-\Delta e_{'+str(i)+','+str(j)+',DNS})}{max(Im(\Delta e_{'+str(i)+','+str(j)+',DNS}))}$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[3][4])
                cbar = plot.colorbar(c_ds_err_imag_plot,cax,ticks=[MAX_c_ds_err_imag,MAX_c_ds_err_imag/2,0.0,-MAX_c_ds_err_imag/2,-MAX_c_ds_err_imag],format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                inner.append(gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=outer[4],wspace=0.05,hspace=0.1,width_ratios=width_ratios))
                ax = plot.Subplot(fig,inner[4][0])
                c_pinn_err_real_plot = ax.contourf(X_grid,Y_grid,c_pinn_err_real,levels=levels_c_pinn_err_real,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.set_aspect('equal')
                dots = ax.plot(x_dots_plot_err,y_dots_plot_err,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_ylabel('y/D',fontsize=8,labelpad=-5)
                ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
                ticklabs = ax.get_yticklabels()
                ax.set_yticklabels(ticklabs, fontsize=8)
                ticklabs = ax.get_xticklabels()
                ax.set_xticklabels(ticklabs, fontsize=8)
                ax.text(-1.8,1.4,'(ae)',fontsize=8)
                ax.text(5,1.2,'$\\frac{R(\Delta e_{'+str(i)+','+str(j)+',PINN}-\Delta e_{'+str(i)+','+str(j)+',DNS})}{max(R(\Delta e_{'+str(i)+','+str(j)+',DNS}))}$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[4][1])
                cbar = plot.colorbar(c_pinn_err_real_plot,cax,ticks=[MAX_c_pinn_err_real,MAX_c_pinn_err_real/2,0.0,-MAX_c_pinn_err_real/2,-MAX_c_pinn_err_real],format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                ax = plot.Subplot(fig,inner[4][3])
                c_pinn_err_imag_plot = ax.contourf(X_grid,Y_grid,c_pinn_err_imag,levels=levels_c_pinn_err_imag,cmap= matplotlib.colormaps['bwr'],extend='both')
                ax.yaxis.set_tick_params(labelleft=False)
                dots = ax.plot(x_dots_plot_err,y_dots_plot_err,markersize=2,linewidth=0,color='k',marker='.',fillstyle='full',markeredgecolor='none')
                circle = plot.Circle((0,0),0.5,color='white',fill=True)
                ax.add_patch(circle)
                circle = plot.Circle((0,0),0.5,color='k',fill=False)
                ax.add_patch(circle)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ticklabs = ax.get_xticklabels()
                ax.set_xticklabels(ticklabs, fontsize=8)
                ax.set_xlabel('x/D',fontsize=8,labelpad=-1)
                ax.set_aspect('equal')
                ax.text(-1.8,1.4,'(be)',fontsize=8)
                ax.text(5,1.2,'$\\frac{Im(\Delta e_{'+str(i)+','+str(j)+',PINN}-\Delta e_{'+str(i)+','+str(j)+',DNS})}{max(Im(\Delta e_{'+str(i)+','+str(j)+',DNS}))}$',fontsize=8)
                fig.add_subplot(ax)
                cax=plot.Subplot(fig,inner[4][4])
                cbar = plot.colorbar(c_pinn_err_imag_plot,cax,ticks=[MAX_c_pinn_err_imag,MAX_c_pinn_err_imag/2,0.0,-MAX_c_pinn_err_imag/2,-MAX_c_pinn_err_imag],format=tkr.FormatStrFormatter('%.1e'))
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=8)
                fig.add_subplot(cax)

                plot.savefig(figures_dir+'spatial_err/S'+str(supersample_factors_ds[s])+'/spatial_err_S'+str(supersample_factors_ds[s])+'_mode'+str(i)+'_mode'+str(j)+'.pdf')
                plot.savefig(figures_dir+'spatial_err/S'+str(supersample_factors_ds[s])+'/spatial_err_S'+str(supersample_factors_ds[s])+'_mode'+str(i)+'_mode'+str(j)+'.png',dpi=300)
                plot.close(fig)
