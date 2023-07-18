import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp
import os
import re

# functions
def find_highest_numbered_file(path_prefix, number_pattern, suffix):
    # Get the directory path and file prefix
    directory, file_prefix = os.path.split(path_prefix)
    
    # Compile the regular expression pattern
    pattern = re.compile(f'{file_prefix}({number_pattern}){suffix}')
    
    # Initialize variables to track the highest number and file path
    highest_number = 0
    highest_file_path = None
    
    # Iterate over the files in the directory
    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            file_number = int(match.group(1))
            if file_number > highest_number:
                highest_number = file_number
                highest_file_path = os.path.join(directory, file)
    
    return highest_file_path, highest_number

def extract_matching_integers(prefix, number_pattern, suffix):
    # Get the directory path and file name prefix
    directory, file_prefix = os.path.split(prefix)
    
    # Compile the regular expression pattern
    pattern = re.compile(f'{file_prefix}({number_pattern}){suffix}')
    
    # Initialize a NumPy array to store the matching integers
    matching_integers = np.array([], dtype=int)
    
    # Iterate over the files in the directory

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            matching_integers = np.append(matching_integers, int(match.group(1)))
    
    return matching_integers

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# script

output_base_dir = 'C:/projects/pinns_narval/sync/output/'
data_dir = 'C:/projects/pinns_narval/sync/data/mazi_fixed/'
data_dir_grid = 'C:/projects/pinns_narval/sync/data/mazi_fixed_grid/'
case_prefix = 'mf_new_mean'



training_cases = extract_matching_integers(output_base_dir+case_prefix,'[0-9][0-9][0-9]','_output')

for k in training_cases:
    case_name = case_prefix + "{:03d}".format(k)
    print('This case is: ', case_name)
    output_dir = output_base_dir+case_name + '_output/'


    meanVelocityFile = h5py.File(data_dir+'meanField.mat','r')
    configFile = h5py.File(data_dir+'configuration.mat','r')
    meanPressureFile = h5py.File(data_dir+'meanPressure.mat','r')
    reynoldsStressFile = h5py.File(data_dir+'reynoldsStresses.mat','r')
    configFileGrid = h5py.File(data_dir_grid+'configuration.mat')

    predfilename,epoch_number = find_highest_numbered_file(output_dir+case_name+'_ep','[0-9]*','_pred.mat')
    predFile =  h5py.File(predfilename,'r')

    figures_folder = output_dir+'figures/'
    create_directory_if_not_exists(figures_folder)
    figure_prefix = figures_folder + case_name+'_ep'+str(epoch_number)

    SaveFig = True
    PlotFig = False

    ux = np.array(meanVelocityFile['meanField'][0,:]).transpose()
    ux = ux[:,0]
    uy = np.array(meanVelocityFile['meanField'][1,:]).transpose()
    uy = uy[:,0]
    p = np.array(meanPressureFile['meanPressure']).transpose()
    p = p[:,0]
    upup = np.array(reynoldsStressFile['reynoldsStresses'][0,:]).transpose()
    upvp = np.array(reynoldsStressFile['reynoldsStresses'][1,:]).transpose()
    vpvp = np.array(reynoldsStressFile['reynoldsStresses'][2,:]).transpose()
    print(configFile.keys())
    x = np.array(configFile['X'][0,:])
    X_grid = np.array(configFileGrid['X_grid'])
    y = np.array(configFile['X'][1,:])
    Y_grid = np.array(configFileGrid['Y_grid'])
    d = np.array(configFile['cylinderDiameter'])[0]
    x_vec_grid = np.array(configFileGrid['X_vec'][0,:])
    y_vec_grid = np.array(configFileGrid['X_vec'][1,:])

    MAX_ux = np.max(ux)
    MAX_uy = np.max(uy)
    MAX_upup = np.max(upup)
    MAX_upvp = np.max(upvp) # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
    MAX_vpvp = np.max(vpvp)
    MAX_p= 1 # estimated maximum pressure

    nu_mol = 0.0066667
    ux_pred = np.array(predFile['pred'][:,0])*MAX_ux
    uy_pred = np.array(predFile['pred'][:,1])*MAX_uy
    p_pred = np.array(predFile['pred'][:,5])*MAX_p 
    #nu_pred =  np.power(np.array(predFile['pred'][:,3]),2)*MAX_nut
    upup_pred = np.array(predFile['pred'][:,2])*MAX_upup
    upvp_pred = np.array(predFile['pred'][:,3])*MAX_upvp
    vpvp_pred = np.array(predFile['pred'][:,4])*MAX_vpvp
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

    print('ux_pred.shape: ',ux_pred.shape)
    print('uy_pred.shape: ',uy_pred.shape)
    print('p_pred.shape: ',p_pred.shape)
    #print('nu_pred.shape: ',nu_pred.shape)
    print('upvp_pred.shape: ',upvp_pred.shape)

    # note that the absolute value of the pressure doesnt matter, only grad p and grad2 p, so subtract the mean 
    #p_pred = p_pred-(1/3)*(upup+vpvp)#p_pred - (1/3)*(upup+vpvp)
    cylinder_mask = np.power(np.power(X_grid,2.0)+np.power(Y_grid,2.0),0.5)<0.5*d
    ux_grid = sp.interpolate.griddata((x,y),ux,(X_grid,Y_grid),method='cubic')
    ux_grid[cylinder_mask] = np.NaN
    ux_pred_grid = sp.interpolate.griddata((x,y),ux_pred,(X_grid,Y_grid),method='cubic')
    ux_pred_grid[cylinder_mask] = np.NaN
    ux_err_grid = sp.interpolate.griddata((x,y),ux_pred-ux,(X_grid,Y_grid),method='cubic')
    ux_err_grid[cylinder_mask] = np.NaN

    uy_grid = sp.interpolate.griddata((x,y),uy,(X_grid,Y_grid),method='cubic')
    uy_grid[cylinder_mask] = np.NaN
    uy_pred_grid = sp.interpolate.griddata((x,y),uy_pred,(X_grid,Y_grid),method='cubic')
    uy_pred_grid[cylinder_mask] = np.NaN
    uy_err_grid = sp.interpolate.griddata((x,y),uy_pred-uy,(X_grid,Y_grid),method='cubic')
    uy_err_grid[cylinder_mask] = np.NaN

    p_grid = sp.interpolate.griddata((x,y),p,(X_grid,Y_grid),method='cubic')
    p_grid[cylinder_mask] = np.NaN
    p_pred_grid = sp.interpolate.griddata((x,y),p_pred,(X_grid,Y_grid),method='cubic')
    p_pred_grid[cylinder_mask] = np.NaN
    p_err_grid = sp.interpolate.griddata((x,y),p_pred-p,(X_grid,Y_grid),method='cubic')
    p_err_grid[cylinder_mask] = np.NaN

    upup_grid = sp.interpolate.griddata((x,y),upup,(X_grid,Y_grid),method='cubic')
    upup_grid[cylinder_mask] = np.NaN
    upup_pred_grid = sp.interpolate.griddata((x,y),upup_pred,(X_grid,Y_grid),method='cubic')
    upup_pred_grid[cylinder_mask] = np.NaN
    upup_err_grid = sp.interpolate.griddata((x,y),upup_pred-upup,(X_grid,Y_grid),method='cubic')
    upup_err_grid[cylinder_mask] = np.NaN

    upvp_grid = sp.interpolate.griddata((x,y),upvp,(X_grid,Y_grid),method='cubic')
    upvp_grid[cylinder_mask] = np.NaN
    upvp_pred_grid = sp.interpolate.griddata((x,y),upvp_pred,(X_grid,Y_grid),method='cubic')
    upvp_pred_grid[cylinder_mask] = np.NaN
    upvp_err_grid = sp.interpolate.griddata((x,y),upvp_pred-upvp,(X_grid,Y_grid),method='cubic')
    upvp_err_grid[cylinder_mask] = np.NaN

    vpvp_grid = sp.interpolate.griddata((x,y),vpvp,(X_grid,Y_grid),method='cubic')
    vpvp_grid[cylinder_mask] = np.NaN
    vpvp_pred_grid = sp.interpolate.griddata((x,y),vpvp_pred,(X_grid,Y_grid),method='cubic')
    vpvp_pred_grid[cylinder_mask] = np.NaN
    vpvp_err_grid = sp.interpolate.griddata((x,y),vpvp_pred-vpvp,(X_grid,Y_grid),method='cubic')
    vpvp_err_grid[cylinder_mask] = np.NaN

    f1_levels = np.linspace(-MAX_ux,MAX_ux,21)
    fig = plot.figure(1)
    ax = fig.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,ux_grid,levels=f1_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,ux_pred_grid,levels=f1_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.axis('equal')
    fig.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,ux_err_grid/MAX_ux,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_ux.png',dpi=300)

    f2_levels = np.linspace(-MAX_uy,MAX_uy,21)
    fig2 = plot.figure(2)
    fig2.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,uy_grid,levels=f2_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig2.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,uy_pred_grid,levels=f2_levels)
    plot.set_cmap('bwr')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig2.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,uy_err_grid/MAX_uy,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.axis('equal')
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_uy.png',dpi=300)


    f3_levels = np.linspace(-MAX_p,MAX_p,21)
    fig3 = plot.figure(3)
    fig3.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,p_grid,f3_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    fig3.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,p_pred_grid,f3_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig3.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,p_err_grid/MAX_p,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.xlabel('x/D')
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_p.png',dpi=300)


    f4_levels = np.linspace(-MAX_upup,MAX_upup,21)
    fig4 = plot.figure(4)
    fig4.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,upup_grid,f4_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    fig4.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,upup_pred_grid,f4_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig4.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,upup_err_grid/MAX_upup,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_upup.png',dpi=300)


    f5_levels = np.linspace(-MAX_upvp,MAX_upvp,21)
    fig5 = plot.figure(5)
    fig5.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,upvp_grid,f5_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    fig5.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,upvp_pred_grid,f5_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig5.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,upvp_err_grid/MAX_upvp,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_upvp.png',dpi=300)


    f6_levels = np.linspace(-MAX_vpvp,MAX_vpvp,21)
    fig6 = plot.figure(6)
    fig6.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,vpvp_grid,f6_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    fig6.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,vpvp_pred_grid,f6_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    plot.ylabel('y/D')
    fig6.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,vpvp_err_grid/MAX_vpvp,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=-2.0,right=10.0)
    ax.set_ylim(bottom=-2.0,top=2.0)
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_vpvp.png',dpi=300)


  # check if the error file exists, if so plot
    errorfilename = output_dir+case_name+'_ep'+str(epoch_number)+'_error.mat'
    print(errorfilename)
    if os.path.isfile(errorfilename):
        errorFile =  h5py.File(errorfilename,'r')
        mx = np.array(errorFile['mxr_grid'])
        my =  np.array(errorFile['myr_grid'])
        mass = np.array(errorFile['massr_grid'])

        

        MAX_u = np.nanmax(np.power(np.power(ux.flatten(),2)+np.power(uy.flatten(),2),0.5))

        mx_grid = mx.reshape(X_grid.shape)
        mx_grid[cylinder_mask]=np.NaN
        my_grid = my.reshape(X_grid.shape)
        my_grid[cylinder_mask]=np.NaN
        mass_grid = mass.reshape(X_grid.shape)
        mass_grid[cylinder_mask]=np.NaN

        mx_b,mx_t = np.nanpercentile(mx,[1,99])
        levels_mx = np.linspace(mx_b,mx_t,21)/MAX_u
        mx_grid[mx_grid>mx_t] = mx_t
        mx_grid[mx_grid<mx_b] = mx_b
        my_b,my_t = np.nanpercentile(my,[1,99])
        my_grid[my_grid>my_t] = my_t
        my_grid[my_grid<my_b] = my_b
        levels_my = np.linspace(my_b,my_t,21)/MAX_u
        mass_b,mass_t = np.nanpercentile(mass,[1,99])
        mass_grid[mass_grid>mass_t] = mass_t
        mass_grid[mass_grid<mass_b] = mass_b
        levels_mass = np.linspace(mass_b,mass_t,21)/MAX_u
        fig7 = plot.figure(7)
        ax = fig7.add_subplot(3,1,1)
        plot.axis('equal')
        plot.contourf(X_grid,Y_grid,mx_grid/MAX_u,levels=levels_mx)
        plot.set_cmap('bwr')
        plot.colorbar()
        ax=plot.gca()
        ax.set_xlim(left=-2.0,right=10.0)
        ax.set_ylim(bottom=-2.0,top=2.0)
        plot.ylabel('y/D')
        fig7.add_subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,my_grid/MAX_u,levels=levels_my)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.ylabel('y/D')
        ax=plot.gca()
        ax.set_xlim(left=-2.0,right=10.0)
        ax.set_ylim(bottom=-2.0,top=2.0)
        plot.axis('equal')
        fig7.add_subplot(3,1,3)
        plot.contourf(X_grid,Y_grid,mass_grid/MAX_u,levels=levels_mass)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.ylabel('y/D')
        plot.xlabel('x/D')
        plot.axis('equal')
        ax=plot.gca()
        ax.set_xlim(left=-2.0,right=10.0)
        ax.set_ylim(bottom=-2.0,top=2.0)
        if SaveFig:
            plot.savefig(figure_prefix+'_error.png',dpi=300)


    if PlotFig:
        plot.show()
    
    plot.close('all')



        
