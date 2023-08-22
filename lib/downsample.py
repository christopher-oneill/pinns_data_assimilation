import numpy as np

def compute_downsample_inds(S,n_x,n_y):
    # downsample the data
    n_x_d = int(n_x/S)
    n_y_d = int(n_y/S)
    downsample_inds_x = np.int64(np.linspace(0,n_x,n_x_d+2))
    downsample_inds_x = downsample_inds_x[1:downsample_inds_x.shape[0]-1]
    downsample_inds_y = np.int64(np.linspace(0,n_y,n_y_d+2))
    downsample_inds_y = downsample_inds_y[1:downsample_inds_y.shape[0]-1]
    linear_downsample_inds = np.zeros([(downsample_inds_x.size)*(downsample_inds_y.size),1],dtype=np.int64)
    #x = np.reshape(x,[n_x,n_y])
    #y = np.reshape(y,[n_x,n_y])
    for i in range(downsample_inds_x.size):
        linear_downsample_inds[(i*downsample_inds_y.size):((i+1)*downsample_inds_y.size),0] = downsample_inds_y + downsample_inds_x[i]*n_y
    return linear_downsample_inds, n_x_d, n_y_d

def compute_downsample_inds_even(S,n_x,n_y):
    # downsample the data
    n_x_d = int(n_x/S)
    n_y_d = int(n_y/S)
    downsample_inds_x = np.int64(np.linspace(-S/2,n_x-1+S/2,n_x_d+2))
    downsample_inds_x = downsample_inds_x[1:downsample_inds_x.shape[0]-1]
    downsample_inds_y = np.int64(np.linspace(-S/2,n_y-1+S/2,n_y_d+2))
    downsample_inds_y = downsample_inds_y[1:downsample_inds_y.shape[0]-1]
    linear_downsample_inds = np.zeros([(downsample_inds_x.size)*(downsample_inds_y.size),1],dtype=np.int64)
    #x = np.reshape(x,[n_x,n_y])
    #y = np.reshape(y,[n_x,n_y])
    for i in range(downsample_inds_x.size):
        linear_downsample_inds[(i*downsample_inds_y.size):((i+1)*downsample_inds_y.size),0] = downsample_inds_y + downsample_inds_x[i]*n_y
    return linear_downsample_inds, n_x_d, n_y_d