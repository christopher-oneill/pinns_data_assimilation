

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import sys
sys.path.append('C:/projects/pinns_local/code/')
from pinns_galerkin_viv.lib.downsample import compute_downsample_inds

from pinns_galerkin_viv.lib.file_util import extract_matching_integers
from pinns_galerkin_viv.lib.file_util import find_highest_numbered_file
from pinns_galerkin_viv.lib.file_util import create_directory_if_not_exists

# script

figures_dir = 'C:/projects/paper_figures/mean_field/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
output_dir = 'C:/projects/pinns_narval/sync/output/'

cases_list = ['mfg_vdnn_mean002_S1_L10N100_output/mfg_vdnn_mean002_S1_L10N100_ep12000_pred.mat','mfg_vdnn_mean004_S4_L10N100_output/mfg_vdnn_mean004_S4_L10N100_ep20000_pred.mat','mfg_vdnn_mean008_S8_L10N100_output/mfg_vdnn_mean008_S8_L10N100_ep189000_pred.mat','mfg_vdnn_mean008_S16_L10N100_output/mfg_vdnn_mean008_S16_L10N100_ep171000_pred.mat','mfg_vdnn_mean008_S32_L10N100_output/mfg_vdnn_mean008_S32_L10N100_ep186000_pred.mat']
cases_supersample_factor = [1,4,8,16,32]
n_cases = len(cases_list)

# summary plot quantities
mean_error_ux = np.zeros((len(cases_supersample_factor),1))
p95_error_ux = np.zeros((len(cases_supersample_factor),1))
mean_error_uy = np.zeros((len(cases_supersample_factor),1))
p95_error_uy = np.zeros((len(cases_supersample_factor),1))
mean_error_p = np.zeros((len(cases_supersample_factor),1))
p95_error_p = np.zeros((len(cases_supersample_factor),1))


# load the reference data
meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
configFile = h5py.File(data_dir+'configuration.mat','r')
meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
                
ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]

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
p_grid = np.reshape(p,X_grid.shape)
p_grid[cylinder_mask] = np.NaN


for c in range(n_cases):
    predFile = h5py.File(output_dir+cases_list[c],'r')

    ux_pred = np.array(predFile['pred'][:,0])*MAX_ux
    uy_pred = np.array(predFile['pred'][:,1])*MAX_uy
    p_pred = np.array(predFile['pred'][:,5])*MAX_p 

    ux_pred_grid = np.reshape(ux_pred,X_grid.shape)
    ux_pred_grid[cylinder_mask] = np.NaN
    uy_pred_grid = np.reshape(uy_pred,X_grid.shape)
    uy_pred_grid[cylinder_mask] = np.NaN
    p_pred_grid = np.reshape(p_pred,X_grid.shape)
    p_pred_grid[cylinder_mask] = np.NaN

    ux_err = np.log10(100*np.abs((ux_grid-ux_pred_grid)/MAX_ux))
    uy_err = np.log10(100*np.abs((uy_grid-uy_pred_grid)/MAX_uy))
    p_err = np.log10(100*np.abs((p_grid-p_pred_grid)/MAX_p))

    # compute error values for summary

    summary_err_ux = 100*np.abs((ux_grid-ux_pred_grid)/MAX_ux)
    summary_err_uy = 100*np.abs((uy_grid-uy_pred_grid)/MAX_uy)
    summary_err_p = 100*np.abs((p_grid-p_pred_grid)/MAX_p)
    mean_error_ux[c] = np.nanmean(summary_err_ux.ravel())
    mean_error_uy[c] = np.nanmean(summary_err_uy.ravel())
    mean_error_p[c] = np.nanmean(summary_err_p.ravel())
    p95_error_ux[c] = np.nanpercentile(summary_err_ux.ravel(),95)
    p95_error_uy[c] = np.nanpercentile(summary_err_uy.ravel(),95)
    p95_error_p[c] = np.nanpercentile(summary_err_p.ravel(),95)
            

fig,axs = plot.subplots(3,1)
fig.set_size_inches(3.37,5.5)



supersample_factors = np.array(cases_supersample_factor)
axs[0].plot(supersample_factors,mean_error_ux,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
axs[0].plot(supersample_factors,p95_error_ux,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
axs[0].set_xscale('log')
axs[0].set_xticks([1,4,8,16,32])
axs[0].set_xticklabels([1,4,8,16,32])
axs[0].xaxis.set_tick_params(labelbottom=False)
axs[0].set_yscale('log')
axs[0].set_yticks([0.1,0.5,1.0,2.0])
axs[0].set_yticklabels([0.1,0.5,1.0,2.0])
axs[0].set_ylabel("Percent Error")
axs[0].set_title('$u_x$')
axs[0].legend({'95th Percentile','Mean'},fontsize=7)
axs[0].grid('on')
axs[0].text(1,3,'(a)',fontsize=10)

axs[1].plot(supersample_factors,mean_error_uy,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
axs[1].plot(supersample_factors,p95_error_uy,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
axs[1].set_xscale('log')
axs[1].set_xticks([1,4,8,16,32])
axs[1].set_xticklabels([1,4,8,16,32])
axs[1].xaxis.set_tick_params(labelbottom=False)
axs[1].set_yscale('log')
axs[1].set_yticks([0.1,0.5,1.0])
axs[1].set_yticklabels([0.1,0.5,1.0])
axs[1].set_ylabel("Percent Error")
axs[1].set_title('$u_y$')
axs[1].grid('on')
axs[1].text(1,1.5,'(b)',fontsize=10)

axs[2].plot(supersample_factors,mean_error_p,linewidth=0,marker='o',color='black',markersize=3,markerfacecolor='black')
axs[2].plot(supersample_factors,p95_error_p,linewidth=0,marker='o',color='red',markersize=3,markerfacecolor='red')
axs[2].set_xscale('log')
axs[2].set_xticks([1,4,8,16,32])
axs[2].set_xticklabels([1,4,8,16,32])
axs[2].set_yscale('log')
axs[2].set_yticks([0.1,0.5,1.0,2.0])
axs[2].set_yticklabels([0.1,0.5,1.0,2.0])
axs[2].set_xlabel('Supersample Factor')
axs[2].set_ylabel("Percent Error")
axs[2].set_title('$p$')
axs[2].grid('on')
axs[2].text(1,4.5,'(c)',fontsize=10)

fig.tight_layout()
plot.savefig(figures_dir+'meanFieldAssimilation_error.pdf')
plot.show()


