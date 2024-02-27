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



def compute_downsample_inds_center(S,Xi,Yi):
    # downsample the data
    IndMinX = np.argmin(np.abs(Xi))
    IndMinY = np.argmin(np.abs(Yi))
    lower_inds_x = np.flip(np.arange(IndMinX,-S,-S))
    lower_inds_x = lower_inds_x[lower_inds_x>=0] # if the last index is below zero, we discard it
    lower_inds_y = np.flip(np.arange(IndMinY,-S,-S))
    lower_inds_y = lower_inds_y[lower_inds_y>=0] # if the last index is below zero, we discard it
    vecX = np.concatenate((lower_inds_x,np.arange(IndMinX+S,Xi.shape[0],S)))
    vecY = np.concatenate((lower_inds_y,np.arange(IndMinY+S,Yi.shape[0],S)))
    matX,matY = np.meshgrid(vecX,vecY)
    index_vector = np.int64(np.stack((matX.ravel(),matY.ravel()),axis=1))
    linear_downsample_inds = np.ravel_multi_index(index_vector.transpose(),(Xi.shape[0],Yi.shape[0]))
    return linear_downsample_inds, vecX.shape[0], vecY.shape[0]

def compute_downsample_inds_irregular(S,X,d):
    # from riches2019 PIV data: dx,dy= 0.725mm,D=19.05; d/dx = 26.27 
    # based on the previous gridded data; dx = 0.025, D = 1, D/dx = 40
    batch_size=10000
    dx_o = 0.025
    dx = S*dx_o # compute the new spacing
    # compute target grids
    x_t = np.concatenate((-(np.arange(dx,2.0,dx)[::-1]),np.arange(0.0,10.0,dx)),0)
    y_t = np.concatenate((-(np.arange(dx,2.0,dx)[::-1]),np.arange(0.0,2.0,dx)),0)
    X_t,Y_t = np.meshgrid(x_t,y_t)
    X_t = X_t.ravel()
    Y_t = Y_t.ravel()
    # remove any that will be inside cylinder
    inds_outside = (np.power(np.power(X_t,2.0)+np.power(Y_t,2.0),0.5)>0.5*d).ravel()
    X_t =X_t[inds_outside]
    Y_t =Y_t[inds_outside]
    

    if X_t.shape[0] >batch_size:
        # batched mode if the array is too big to allocate in memory
        inds_list = []
        batches = np.int64(np.ceil(X_t.shape[0]/batch_size))

        for b in range(batches):
            batch_inds_l = b*batch_size
            batch_inds_h = np.min(np.array([(b+1)*batch_size,X_t.shape[0]]))
            distance_matrix = np.power(np.power(np.reshape(X[:,0],[X.shape[0],1])-np.reshape(X_t[batch_inds_l:batch_inds_h],[1,batch_inds_h-batch_inds_l]),2.0) + np.power(np.reshape(X[:,1],[X.shape[0],1])-np.reshape(Y_t[batch_inds_l:batch_inds_h],[1,batch_inds_h-batch_inds_l]),2.0),0.5)
            inds_batch = np.argmin(distance_matrix,axis=0)
            inds_list.append(inds_batch)
        inds = np.concatenate(inds_list,axis=0)
    else:
        # regular mode
        distance_matrix = np.power(np.power(np.reshape(X[:,0],[X.shape[0],1])-np.reshape(X_t,[1,X_t.size]),2.0) + np.power(np.reshape(X[:,1],[X.shape[0],1])-np.reshape(Y_t,[1,Y_t.size]),2.0),0.5)
        inds = np.argmin(distance_matrix,axis=0)

    return inds

def boundary_inds_irregular(Nx,Ny,X):
    # this function gets the indices of boundary points so we can extract them for testing the boundary value problem

    # top and bottom
    x_x = np.linspace(-2.0,10.0,Nx)
    y_xt = 2.0*np.ones([Nx,])
    y_xb = -2.0*np.ones([Nx,])

    # left side
    x_y = -2.0*np.ones([Ny,])
    y_y = np.linspace(-2.0,2.0,Ny)

    # concatenate them all
    X_t = np.concatenate((x_x,x_y,x_x),axis=0)
    Y_t = np.concatenate((y_xb,y_y,y_xt),axis=0)

    distance_matrix = np.power(np.power(np.reshape(X[:,0],[X.shape[0],1])-np.reshape(X_t,[1,X_t.size]),2.0) + np.power(np.reshape(X[:,1],[X.shape[0],1])-np.reshape(Y_t,[1,Y_t.size]),2.0),0.5)
    inds = np.argmin(distance_matrix,axis=0)
    unique_inds = np.unique(inds,return_index=True)[1]

    return inds[unique_inds]




