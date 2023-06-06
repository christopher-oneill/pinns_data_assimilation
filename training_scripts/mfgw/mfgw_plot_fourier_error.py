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

base_dir = 'C:/projects/pinns_beluga/sync/'
data_dir = base_dir+'data/mazi_fixed_grid_wake/'
case_prefix = 'mfgw_fourier9_'
output_base_dir = base_dir+'output/'
mode_number=8 # equivalent to 9 in matlab notation 

training_cases = extract_matching_integers(output_base_dir+case_prefix,'[0-9][0-9][0-9]','_output')

for k in training_cases:
    case_name = case_prefix + "{:03d}".format(k)
    output_dir = output_base_dir+case_name + '_output/'

    configFile = h5py.File(data_dir+'configuration.mat','r')
    fourierModeFile = h5py.File(data_dir+'fourierDataShort.mat','r')

    errorfilename,epoch_number = find_highest_numbered_file(output_dir+case_name+'_ep','[0-9]*','_error.mat')
    if errorfilename is None:
        continue

    errorFile =  h5py.File(errorfilename,'r')
    figures_folder = output_dir+'figures/'
    create_directory_if_not_exists(figures_folder)
    figure_prefix = figures_folder + case_name+'_ep'+str(epoch_number)

    SaveFig = True
    PlotFig = False


    x = np.array(configFile['X_vec'][0,:])
    X_grid_temp = np.array(configFile['X_grid'])
    X_grid =np.reshape(x,X_grid_temp.shape[::-1])
    y = np.array(configFile['X_vec'][1,:])
    #Y_grid = np.array(configFile['Y_grid'])[::-1]
    Y_grid = np.reshape(y,X_grid.shape)
    d = np.array(configFile['cylinderDiameter'])[0]

    nu_mol = 0.0066667
    
    # velocity fourier modes
    mxr = np.array(errorFile['mxr'])
    mxi = np.array(errorFile['mxi'])
    myr = np.array(errorFile['myr'])
    myi = np.array(errorFile['myi'])
    massr = np.array(errorFile['massr'])
    massi = np.array(errorFile['massi'])


    phi_xr = np.array(fourierModeFile['velocityModesShortReal'][0,mode_number,:]).transpose()
    phi_xi = np.array(fourierModeFile['velocityModesShortImag'][0,mode_number,:]).transpose()
    phi_yr = np.array(fourierModeFile['velocityModesShortReal'][1,mode_number,:]).transpose()
    phi_yi = np.array(fourierModeFile['velocityModesShortImag'][1,mode_number,:]).transpose()
    
    print('x.shape: ',x.shape)
    print('y.shape: ',y.shape)
    print('X_grid.shape: ',X_grid.shape)
    print('Y_grid.shape: ',Y_grid.shape)
    print('d: ',d.shape)



    # fourier coefficients of the fluctuating field
    mxr_grid = np.reshape(mxr,X_grid.shape)
    mxi_grid = np.reshape(mxi,X_grid.shape)
    myr_grid = np.reshape(myr,X_grid.shape)
    myi_grid = np.reshape(myi,X_grid.shape)
    massr_grid = np.reshape(massr,X_grid.shape)
    massi_grid = np.reshape(massi,X_grid.shape)

    # given values / reference values
    phi_xr_grid = np.reshape(phi_xr,X_grid.shape)
    phi_xi_grid = np.reshape(phi_xi,X_grid.shape)
    phi_yr_grid = np.reshape(phi_yr,X_grid.shape)
    phi_yi_grid = np.reshape(phi_yi,X_grid.shape)

    print(np.mean(mxr_grid/phi_xr_grid))

    x_lim_vec = [0.5,10.0]
    y_lim_vec = [-2.0,2.0]
    fig = plot.figure(1)
    ax = fig.add_subplot(3,1,1)
    plot.axis('equal')
    plot.contourf(X_grid,Y_grid,mxr_grid-phi_xi_grid,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.ylabel('y/D')
    fig.add_subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,mxi_grid+phi_xr_grid,levels=21)
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.ylabel('y/D')
    ax=plot.gca()
    ax.set_xlim(left=x_lim_vec[0],right=x_lim_vec[1])
    ax.set_ylim(bottom=y_lim_vec[0],top=y_lim_vec[1])
    plot.axis('equal')
    fig.add_subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,myr_grid-phi_yi_grid,levels=21)
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
    plot.contourf(X_grid,Y_grid,myi_grid+phi_yr_grid,levels=21)
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



    
