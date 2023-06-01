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

base_dir = 'C:/projects/pinns_narval/'
data_dir = base_dir+'data/mazi_fixed_grid_wake/'
case_prefix = 'mfgwS16m'
output_base_dir = base_dir+'output/'


training_cases = extract_matching_integers(output_base_dir+case_prefix,'[0-9][0-9][0-9]','_output')

for k in training_cases:
    case_name = case_prefix + "{:03d}".format(k)
    output_dir = output_base_dir+case_name + '_output/'

    meanVelocityFileF = h5py.File(data_dir+'meanVelocity.mat','r')
    meanVelocityFile = h5py.File(data_dir+'meanVelocityS16.mat','r')
    configFileF = h5py.File(data_dir+'configuration.mat','r')
    configFile = h5py.File(data_dir+'configurationS16.mat','r')
    meanPressureFileF = h5py.File(data_dir+'meanPressure.mat','r')
    meanPressureFile = h5py.File(data_dir+'meanPressureS16.mat','r')
    reynoldsStressFileF = h5py.File(data_dir+'reynoldsStress.mat','r')    
    reynoldsStressFile = h5py.File(data_dir+'reynoldsStressS16.mat','r')

    predfilename,epoch_number = find_highest_numbered_file(output_dir+case_name+'_ep','[0-9]*','_pred.mat')
    predFile =  h5py.File(predfilename,'r')

    figures_folder = output_dir+'figures/'
    create_directory_if_not_exists(figures_folder)
    figure_prefix = figures_folder + case_name+'_ep'+str(epoch_number)

    SaveFig = True
    PlotFig = False

    ux = np.array(meanVelocityFileF['meanVelocity'][0,:]).transpose()
    uy = np.array(meanVelocityFileF['meanVelocity'][1,:]).transpose()
    p = np.array(meanPressureFileF['meanPressure']).transpose()
    p = p[:,0]
    upup = np.array(reynoldsStressFileF['reynoldsStress'][0,:]).transpose()
    upvp = np.array(reynoldsStressFileF['reynoldsStress'][1,:]).transpose()
    vpvp = np.array(reynoldsStressFileF['reynoldsStress'][2,:]).transpose()

    x = np.array(configFileF['X_vec'][0,:])
    X_grid_temp = np.array(configFileF['X_grid'])
    X_grid =np.reshape(x,X_grid_temp.shape[::-1])
    y = np.array(configFileF['X_vec'][1,:])
    #Y_grid = np.array(configFile['Y_grid'])[::-1]
    Y_grid = np.reshape(y,X_grid.shape)
    d = np.array(configFile['cylinderDiameter'])[0]
    xp = np.array(configFile['X_vec'][0,:])
    yp = np.array(configFile['X_vec'][1,:])

    # note that these need to be from the downsampled file, since this is what the internal values to the NN were scaled on
    MAX_ux = np.max(np.array(meanVelocityFile['meanVelocity'][0,:]))
    MAX_uy = np.max(np.array(meanVelocityFile['meanVelocity'][1,:]))
    MAX_upup = np.max(np.array(reynoldsStressFile['reynoldsStress'][0,:]))
    MAX_upvp = np.max(np.array(reynoldsStressFile['reynoldsStress'][1,:])) # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
    MAX_vpvp = np.max(np.array(reynoldsStressFile['reynoldsStress'][2,:]))
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

    ux_grid = np.reshape(ux,X_grid.shape)
    uy_grid = np.reshape(uy,X_grid.shape)
    p_grid = np.reshape(p,X_grid.shape)
    ux_pred_grid = np.reshape(ux_pred,X_grid.shape)
    uy_pred_grid = np.reshape(uy_pred,X_grid.shape)
    p_pred_grid = np.reshape(p_pred,X_grid.shape) - np.mean(p_pred,0)+np.mean(p,0) # we adjust the mean values so they are equal
    upup_grid = np.reshape(upup,X_grid.shape)
    upup_pred_grid = np.reshape(upup_pred,X_grid.shape)
    upvp_grid = np.reshape(upvp,X_grid.shape)
    upvp_pred_grid = np.reshape(upvp_pred,X_grid.shape)
    vpvp_grid = np.reshape(vpvp,X_grid.shape)
    vpvp_pred_grid = np.reshape(vpvp_pred,X_grid.shape)


    x_lim_vec = [0.5,10.0]
    y_lim_vec = [-2.0,2.0]
    f1_levels = np.linspace(-1.2*MAX_ux,1.2*MAX_ux,21)
    fig = plot.figure(1)
    ax = fig.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,ux_grid,levels=f1_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    dots = plot.scatter(xp,yp,0.1,color='black')
    fig.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,ux_pred_grid,levels=f1_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.axis('equal')
    fig.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(ux_grid-ux_pred_grid)/MAX_ux,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_ux.png',dpi=300)

    f2_levels = np.linspace(-1.2*MAX_uy,1.2*MAX_uy,21)
    fig2 = plot.figure(2)
    fig2.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,uy_grid,levels=f2_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    dots = plot.scatter(xp,yp,0.1,color='black')
    fig2.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,uy_pred_grid,levels=f2_levels)
    plot.set_cmap('bwr')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    fig2.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(uy_grid-uy_pred_grid)/MAX_uy,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.axis('equal')
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_uy.png',dpi=300)


    f3_levels = np.linspace(-1.2*MAX_p,1.2*MAX_p,21)
    fig3 = plot.figure(3)
    fig3.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,p_grid,f3_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    dots = plot.scatter(xp,yp,0.1,color='black')
    fig3.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,p_pred_grid,f3_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    fig3.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(p_grid-p_pred_grid)/MAX_p,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.xlabel('x/D')
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_p.png',dpi=300)


    f4_levels = np.linspace(-1.2*MAX_upup,1.2*MAX_upup,21)
    fig4 = plot.figure(4)
    fig4.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,upup_grid,f4_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    dots = plot.scatter(xp,yp,0.1,color='black')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    fig4.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,upup_pred_grid,f4_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    fig4.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(upup_grid-upup_pred_grid)/MAX_upup,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_upup.png',dpi=300)


    f5_levels = np.linspace(-1.2*MAX_upvp,1.2*MAX_upvp,21)
    fig5 = plot.figure(5)
    fig5.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,upvp_grid,f5_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    dots = plot.scatter(xp,yp,0.1,color='black')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    fig5.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,upvp_pred_grid,f5_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    fig5.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(upvp_grid-upvp_pred_grid)/MAX_upvp,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_upvp.png',dpi=300)


    f6_levels = np.linspace(-1.2*MAX_vpvp,1.2*MAX_vpvp,21)
    fig6 = plot.figure(6)
    fig6.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,vpvp_grid,f6_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    dots = plot.scatter(xp,yp,0.1,color='black')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    fig6.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,vpvp_pred_grid,f6_levels)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    fig6.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,(vpvp_grid-vpvp_pred_grid)/MAX_vpvp,21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.axis('equal')
    plot.ylabel('y/D')
    plot.xlabel('x/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    if SaveFig:
        plot.savefig(figure_prefix+'_mean_vpvp.png',dpi=300)


    if PlotFig:
        plot.show()

    plot.close('all')



    
