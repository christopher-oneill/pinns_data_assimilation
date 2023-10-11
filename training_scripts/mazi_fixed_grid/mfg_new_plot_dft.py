import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp
import os
import re
import sys



mode_number_array = np.array([0,1,2,3,4,5])
supersample_factor_list = [1]

sys.path.append('C:/projects/pinns_local/code/')
from pinns_galerkin_viv.lib.downsample import compute_downsample_inds

from pinns_galerkin_viv.lib import file_util

for mn in range(mode_number_array.size):
    
    for p in range(len(supersample_factor_list)):
        # script
        mode_number = mode_number_array[mn]
        supersample_factor = supersample_factor_list[p]

        base_dir = 'C:/projects/pinns_narval/sync/'
        data_dir = base_dir+'data/mazi_fixed_grid/'
        case_prefix = 'mfg_dft'+str(mode_number)+'_S'+str(supersample_factor)+'_j'
        output_base_dir = base_dir+'output/'


        training_cases = file_util.extract_matching_integers(output_base_dir+case_prefix,'[0-9][0-9][0-9]','_output')

        for k in training_cases:
            case_name = case_prefix + "{:03d}".format(k)
            print('This case is: ',case_name)
            output_dir = output_base_dir+case_name + '_output/'

            meanVelocityFile = h5py.File(data_dir+'meanVelocity.mat','r')
            configFile = h5py.File(data_dir+'configuration.mat','r')
            meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
            reynoldsStressFile = h5py.File(data_dir+'reynoldsStress.mat','r')
            fourierModeFile = h5py.File(data_dir+'fourier_data_DFT.mat','r')

            predfilename,epoch_number = file_util.find_highest_numbered_file(output_dir+case_name+'_ep','[0-9]*','_pred.mat')

            predFile =  h5py.File(predfilename,'r')
            figures_folder = output_dir+'figures/'
            file_util.create_directory_if_not_exists(figures_folder)
            figure_prefix = figures_folder + case_name+'_ep'+str(epoch_number)

            SaveFig = True
            PlotFig = False

            ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
            uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()
            p = np.array(meanPressureFile['meanPressure']).transpose()
            p = p[:,0]
            upup = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
            upvp = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
            vpvp = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

            phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
            phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
            phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
            phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()

            psi_r = np.array(fourierModeFile['pressureModesShortReal'][mode_number,:]).transpose()
            psi_i = np.array(fourierModeFile['pressureModesShortImag'][mode_number,:]).transpose()

            tau_xx_r = np.array(fourierModeFile['stressModesShortReal'][0,mode_number,:]).transpose()
            tau_xx_i = np.array(fourierModeFile['stressModesShortImag'][0,mode_number,:]).transpose()
            tau_xy_r = np.array(fourierModeFile['stressModesShortReal'][1,mode_number,:]).transpose()
            tau_xy_i = np.array(fourierModeFile['stressModesShortImag'][1,mode_number,:]).transpose()
            tau_yy_r = np.array(fourierModeFile['stressModesShortReal'][2,mode_number,:]).transpose()
            tau_yy_i = np.array(fourierModeFile['stressModesShortImag'][2,mode_number,:]).transpose()

            omega = np.array(fourierModeFile['fShort'][0,mode_number])*2*np.pi


            x = np.array(configFile['X_vec'][0,:])
            X_grid = np.array(configFile['X_grid'])
            y = np.array(configFile['X_vec'][1,:])
            Y_grid = np.array(configFile['Y_grid'])
            d = np.array(configFile['cylinderDiameter'])[0]

            if supersample_factor>1:
                n_x = X_grid.shape[0]
                n_y = X_grid.shape[1]
                linear_downsample_inds, n_d_x, n_d_y = compute_downsample_inds(supersample_factor,n_x,n_y)

                x_downsample = x[linear_downsample_inds]
                y_downsample = y[linear_downsample_inds]
                valid_inds = np.power(np.power(x_downsample,2.0)+np.power(y_downsample,2.0),0.5)>0.5*d
                x_downsample = x_downsample[valid_inds]
                y_downsample = y_downsample[valid_inds]


            MAX_ux = np.max(ux)
            MAX_uy = np.max(uy)
            MAX_upup = np.max(upup)
            MAX_upvp = np.max(upvp) # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
            MAX_vpvp = np.max(vpvp)
            MAX_p= 1 # estimated maximum pressure, we should 
            MAX_psi= 0.1 # chosen based on abs(max(psi))

            MAX_phi_xr = np.max(phi_xr.flatten())
            MAX_phi_xi = np.max(phi_xi.flatten())
            MAX_phi_yr = np.max(phi_yr.flatten())
            MAX_phi_yi = np.max(phi_yi.flatten())

            MAX_psi_r = np.max(psi_r.flatten())
            MAX_psi_i = np.max(psi_i.flatten())

            MAX_tau_xx_r = np.max(tau_xx_r.flatten())
            MAX_tau_xx_i = np.max(tau_xx_i.flatten())
            MAX_tau_xy_r = np.max(tau_xy_r.flatten())
            MAX_tau_xy_i = np.max(tau_xy_i.flatten())
            MAX_tau_yy_r = np.max(tau_yy_r.flatten())
            MAX_tau_yy_i = np.max(tau_yy_i.flatten())

            nu_mol = 0.0066667
            
            # velocity fourier modes
            phi_xr_pred = np.array(predFile['pred'][:,0])*MAX_phi_xr
            phi_xi_pred = np.array(predFile['pred'][:,1])*MAX_phi_xi
            phi_yr_pred = np.array(predFile['pred'][:,2])*MAX_phi_yr
            phi_yi_pred = np.array(predFile['pred'][:,3])*MAX_phi_yi
            
                # fourier coefficients of the fluctuating field
            tau_xx_r_pred = np.array(predFile['pred'][:,4])*MAX_tau_xx_r
            tau_xx_i_pred = np.array(predFile['pred'][:,5])*MAX_tau_xx_i
            tau_xy_r_pred = np.array(predFile['pred'][:,6])*MAX_tau_xy_r
            tau_xy_i_pred = np.array(predFile['pred'][:,7])*MAX_tau_xy_i
            tau_yy_r_pred = np.array(predFile['pred'][:,8])*MAX_tau_yy_r
            tau_yy_i_pred = np.array(predFile['pred'][:,9])*MAX_tau_yy_i
            # unknowns, pressure fourier modes
            psi_r_pred = np.array(predFile['pred'][:,10])*MAX_psi
            psi_i_pred = np.array(predFile['pred'][:,11])*MAX_psi
            # compute the estimated reynolds stress

            #upvp_pred = -np.multiply(np.reshape(nu_pred+nu_mol,[nu_pred.shape[0],1]),dudy+dvdx)

            print('ux.shape: ',ux.shape)
            print('uy.shape: ',uy.shape)
            print('p.shape: ',p.shape)
            print('upvp.shape: ',upvp.shape)


            print('x.shape: ',x.shape)
            print('y.shape: ',y.shape)
            print('X_grid.shape: ',X_grid.shape)
            print('Y_grid.shape: ',Y_grid.shape)
            print('d: ',d.shape)

            print('phi_xr_pred.shape: ',phi_xr_pred.shape)
            print('phi_yr_pred.shape: ',phi_yr_pred.shape)
            print('psi_r_pred.shape: ',psi_r_pred.shape)
            #print('nu_pred.shape: ',nu_pred.shape)
            print('tau_xx_r_pred.shape: ',tau_xx_r_pred.shape)

            # note that the absolute value of the pressure doesnt matter, only grad p and grad2 p, so subtract the mean 
            #p_pred = p_pred-(1/3)*(upup+vpvp)#p_pred - (1/3)*(upup+vpvp)

            cylinder_mask = (np.power(np.power(X_grid,2)+np.power(Y_grid,2),0.5)<(0.5*d))

            ux_grid = np.reshape(ux,X_grid.shape)
            ux_grid[cylinder_mask] = np.NaN
            uy_grid = np.reshape(uy,X_grid.shape)
            uy_grid[cylinder_mask] = np.NaN
            p_grid = np.reshape(p,X_grid.shape)
            p_grid[cylinder_mask] = np.NaN

            # given values / reference values
            phi_xr_grid = np.reshape(phi_xr,X_grid.shape)
            phi_xr_grid[cylinder_mask] = np.NaN
            phi_xi_grid = np.reshape(phi_xi,X_grid.shape)
            phi_xi_grid[cylinder_mask] = np.NaN
            phi_yr_grid = np.reshape(phi_yr,X_grid.shape)
            phi_yr_grid[cylinder_mask] = np.NaN
            phi_yi_grid = np.reshape(phi_yi,X_grid.shape)
            phi_yi_grid[cylinder_mask] = np.NaN

            # fourier coefficients of the fluctuating field
            tau_xx_r_grid = np.reshape(tau_xx_r,X_grid.shape)
            tau_xx_r_grid[cylinder_mask] = np.NaN
            tau_xx_i_grid = np.reshape(tau_xx_i,X_grid.shape)
            tau_xx_i_grid[cylinder_mask] = np.NaN
            tau_xy_r_grid = np.reshape(tau_xy_r,X_grid.shape)
            tau_xy_r_grid[cylinder_mask] = np.NaN
            tau_xy_i_grid = np.reshape(tau_xy_i,X_grid.shape)
            tau_xy_i_grid[cylinder_mask] = np.NaN
            tau_yy_r_grid = np.reshape(tau_yy_r,X_grid.shape)
            tau_yy_r_grid[cylinder_mask] = np.NaN
            tau_yy_i_grid = np.reshape(tau_yy_i,X_grid.shape)
            tau_yy_i_grid[cylinder_mask] = np.NaN
            # unknowns, pressure fourier modes
            psi_r_grid = np.reshape(psi_r,X_grid.shape)
            psi_r_grid[cylinder_mask] = np.NaN
            psi_i_grid = np.reshape(psi_i,X_grid.shape)
            psi_i_grid[cylinder_mask] = np.NaN

            # predicted values
            phi_xr_pred_grid = np.reshape(phi_xr_pred,X_grid.shape)
            phi_xr_pred_grid[cylinder_mask] = np.NaN
            phi_xi_pred_grid = np.reshape(phi_xi_pred,X_grid.shape)
            phi_xi_pred_grid[cylinder_mask] = np.NaN
            phi_yr_pred_grid = np.reshape(phi_yr_pred,X_grid.shape)
            phi_yr_pred_grid[cylinder_mask] = np.NaN
            phi_yi_pred_grid = np.reshape(phi_yi_pred,X_grid.shape)
            phi_yi_pred_grid[cylinder_mask] = np.NaN

            # fourier coefficients of the fluctuating field
            tau_xx_r_pred_grid = np.reshape(tau_xx_r_pred,X_grid.shape)
            tau_xx_r_pred_grid[cylinder_mask] = np.NaN
            tau_xx_i_pred_grid = np.reshape(tau_xx_i_pred,X_grid.shape)
            tau_xx_i_pred_grid[cylinder_mask] = np.NaN
            tau_xy_r_pred_grid = np.reshape(tau_xy_r_pred,X_grid.shape)
            tau_xy_r_pred_grid[cylinder_mask] = np.NaN
            tau_xy_i_pred_grid = np.reshape(tau_xy_i_pred,X_grid.shape)
            tau_xy_i_pred_grid[cylinder_mask] = np.NaN
            tau_yy_r_pred_grid = np.reshape(tau_yy_r_pred,X_grid.shape)
            tau_yy_r_pred_grid[cylinder_mask] = np.NaN
            tau_yy_i_pred_grid = np.reshape(tau_yy_i_pred,X_grid.shape)
            tau_yy_i_pred_grid[cylinder_mask] = np.NaN
            # unknowns, pressure fourier modes
            psi_r_pred_grid = np.reshape(psi_r_pred,X_grid.shape)
            psi_r_pred_grid[cylinder_mask] = np.NaN
            psi_i_pred_grid = np.reshape(psi_i_pred,X_grid.shape)
            psi_i_pred_grid[cylinder_mask] = np.NaN


            x_lim_vec = [-2,10.0]
            y_lim_vec = [-2.0,2.0]
            f1_levels = np.linspace(-1.2*MAX_phi_xr,1.2*MAX_phi_xr,21)
            fig = plot.figure(1)
            ax = fig.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,phi_xr_grid,levels=f1_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,phi_xr_pred_grid,levels=f1_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.axis('equal')
            fig.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(phi_xr_grid-phi_xr_pred_grid)/MAX_phi_xr,levels=21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_phi_xr.png',dpi=300)

            f2_levels = np.linspace(-1.2*MAX_phi_xi,1.2*MAX_phi_xi,21)
            fig2 = plot.figure(2)
            fig2.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,phi_xi_grid,levels=f2_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig2.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,phi_xi_pred_grid,levels=f2_levels)
            plot.set_cmap('bwr')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            fig2.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(phi_xi_grid-phi_xi_pred_grid)/MAX_phi_xi,levels=21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.axis('equal')
            if SaveFig:
                plot.savefig(figure_prefix+'_phi_xi.png',dpi=300)


            f3_levels = np.linspace(-1.2*MAX_phi_yr,1.2*MAX_phi_yr,21)
            fig3 = plot.figure(3)
            fig3.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,phi_yr_grid,f3_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig3.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,phi_yr_pred_grid,f3_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig3.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(phi_yr_grid-phi_yr_pred_grid)/MAX_phi_yr,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.xlabel('x/D')
            if SaveFig:
                plot.savefig(figure_prefix+'_phi_yr.png',dpi=300)


            f4_levels = np.linspace(-1.2*MAX_phi_yi,1.2*MAX_phi_yi,21)
            fig4 = plot.figure(4)
            fig4.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,phi_yi_grid,f4_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig4.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,phi_yi_pred_grid,f4_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig4.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(phi_yi_grid-phi_yi_pred_grid)/MAX_phi_yi,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_phi_yi.png',dpi=300)


            MAX_psi_r = np.nanmax(psi_r)
            f5_levels = np.linspace(-1.2*MAX_psi_r,1.2*MAX_psi_r,21)
            fig5 = plot.figure(5)
            fig5.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,psi_r_grid,f5_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig5.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,psi_r_pred_grid,f5_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig5.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(psi_r_grid-psi_r_pred_grid)/MAX_psi_r,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_psi_r.png',dpi=300)


            MAX_psi_i = np.nanmax(psi_i)
            print(MAX_psi_i)
            f6_levels = np.linspace(-1.2*MAX_psi_i,1.2*MAX_psi_i,21)
            fig6 = plot.figure(6)
            fig6.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,psi_i_grid,f6_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig6.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,psi_i_pred_grid,f6_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig6.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(psi_i_grid-psi_i_pred_grid)/MAX_psi_i,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_psi_i.png',dpi=300)

            fig7_levels = np.linspace(-1.2*MAX_tau_xx_r,1.2*MAX_tau_xx_r,21)
            fig7 = plot.figure(7)
            fig7.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_xx_r_grid,fig7_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig7.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_xx_r_pred_grid,fig7_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig7.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_xx_r_grid-tau_xx_r_pred_grid)/MAX_tau_xx_r,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_xx_r.png',dpi=300)


            fig8_levels = np.linspace(-1.2*MAX_tau_xx_i,1.2*MAX_tau_xx_i,21)
            fig8 = plot.figure(8)
            fig8.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_xx_i_grid,fig8_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig8.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_xx_i_pred_grid,fig8_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig8.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_xx_i_grid-tau_xx_i_pred_grid)/MAX_tau_xx_i,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_xx_i.png',dpi=300)

            fig9_levels = np.linspace(-1.2*MAX_tau_xy_r,1.2*MAX_tau_xy_r,21)
            fig9 = plot.figure(9)
            fig9.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_xy_r_grid,fig9_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig9.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_xy_r_pred_grid,fig9_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig9.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_xy_r_grid-tau_xy_r_pred_grid)/MAX_tau_xy_r,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_xy_r.png',dpi=300)

            fig10_levels = np.linspace(-1.2*MAX_tau_xy_i,1.2*MAX_tau_xy_i,21)
            fig10 = plot.figure(10)
            fig10.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_xy_i_grid,fig10_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig10.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_xy_i_pred_grid,fig10_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig10.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_xy_i_grid-tau_xy_i_pred_grid)/MAX_tau_xy_i,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_xy_i.png',dpi=300)

            fig11_levels = np.linspace(-1.2*MAX_tau_yy_r,1.2*MAX_tau_yy_r,21)
            fig11 = plot.figure(11)
            fig11.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_yy_r_grid,fig11_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig11.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_yy_r_pred_grid,fig11_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig11.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_yy_r_grid-tau_yy_r_pred_grid)/MAX_tau_yy_r,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_yy_r.png',dpi=300)

            fig12_levels = np.linspace(-1.2*MAX_tau_yy_i,1.2*MAX_tau_yy_i,21)
            fig12 = plot.figure(12)
            fig12.add_subplot(3,1,1)
            plot.axis('equal')
            plot.contourf(X_grid,Y_grid,tau_yy_i_grid,fig12_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.ylabel('y/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            fig12.add_subplot(3,1,2)
            plot.contourf(X_grid,Y_grid,tau_yy_i_pred_grid,fig12_levels)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            plot.ylabel('y/D')
            fig12.add_subplot(3,1,3)
            plot.contourf(X_grid,Y_grid,(tau_yy_i_grid-tau_yy_i_pred_grid)/MAX_tau_yy_i,21)
            plot.set_cmap('bwr')
            plot.colorbar()
            plot.axis('equal')
            plot.ylabel('y/D')
            plot.xlabel('x/D')
            ax=plot.gca()
            ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
            ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
            if SaveFig:
                plot.savefig(figure_prefix+'_tau_yy_i.png',dpi=300)

            if PlotFig:
                plot.show()

            plot.close('all')

            errorfilename,epoch_number = file_util.find_highest_numbered_file(output_dir+case_name+'_ep','[0-9]*','_error.mat')
            if errorfilename is None:
                continue
            else:
                errorFile =  h5py.File(errorfilename,'r')
                figures_folder = output_dir+'figures/'
                file_util.create_directory_if_not_exists(figures_folder)
                figure_prefix = figures_folder + case_name+'_ep'+str(epoch_number)

                SaveFig = True
                PlotFig = False


                nu_mol = 0.0066667
                
                # velocity fourier modes
                mxr = np.array(errorFile['mxr_grid'])
                mxi = np.array(errorFile['mxi_grid'])
                myr = np.array(errorFile['myr_grid'])
                myi = np.array(errorFile['myi_grid'])
                massr = np.array(errorFile['massr_grid'])
                massi = np.array(errorFile['massi_grid'])

                if mxr.size!=X_grid.size:
                    print('Skipped case due to size mismatch: ',case_name)
                    continue
                # fourier coefficients of the fluctuating field
                mxr_grid = np.reshape(mxr,X_grid.shape)
                mxr_grid[cylinder_mask] = np.NaN
                mxi_grid = np.reshape(mxi,X_grid.shape)
                mxi_grid[cylinder_mask] = np.NaN
                myr_grid = np.reshape(myr,X_grid.shape)
                myr_grid[cylinder_mask] = np.NaN
                myi_grid = np.reshape(myi,X_grid.shape)
                myi_grid[cylinder_mask] = np.NaN
                massr_grid = np.reshape(massr,X_grid.shape)
                massr_grid[cylinder_mask] = np.NaN
                massi_grid = np.reshape(massi,X_grid.shape)
                massi_grid[cylinder_mask] = np.NaN

                print(np.mean(mxr_grid/phi_xr_grid))

                x_lim_vec = [-2,10.0]
                y_lim_vec = [-2.0,2.0]
                fig = plot.figure(1)
                ax = fig.add_subplot(3,1,1)
                plot.axis('equal')
                plot.contourf(X_grid,Y_grid,mxr_grid,levels=21)
                plot.set_cmap('bwr')
                plot.colorbar()
                ax=plot.gca()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                plot.ylabel('y/D')
                fig.add_subplot(3,1,2)
                plot.contourf(X_grid,Y_grid,mxi_grid,levels=21)
                plot.set_cmap('bwr')
                plot.colorbar()
                plot.ylabel('y/D')
                ax=plot.gca()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                plot.axis('equal')
                fig.add_subplot(3,1,3)
                plot.contourf(X_grid,Y_grid,myr_grid,levels=21)
                plot.set_cmap('bwr')
                plot.colorbar()
                plot.ylabel('y/D')
                plot.xlabel('x/D')
                plot.axis('equal')
                ax=plot.gca()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                if SaveFig:
                    plot.savefig(figure_prefix+'_error1.png',dpi=300)


                fig2 = plot.figure(2)
                fig2.add_subplot(3,1,1)
                plot.axis('equal')
                plot.contourf(X_grid,Y_grid,myi_grid,levels=21)
                plot.set_cmap('bwr')
                plot.colorbar()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                plot.ylabel('y/D')
                fig2.add_subplot(3,1,2)
                plot.contourf(X_grid,Y_grid,massr_grid,levels=21)
                plot.set_cmap('bwr')
                ax=plot.gca()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                plot.colorbar()
                plot.axis('equal')
                plot.ylabel('y/D')
                fig2.add_subplot(3,1,3)
                plot.contourf(X_grid,Y_grid,massi_grid,levels=21)
                plot.set_cmap('bwr')
                plot.colorbar()
                plot.ylabel('y/D')
                plot.xlabel('x/D')
                ax=plot.gca()
                ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
                ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
                plot.axis('equal')
                if SaveFig:
                    plot.savefig(figure_prefix+'_error2.png',dpi=300)

                if PlotFig:
                    plot.show()

                plot.close('all')



            
