import numpy as np


def vorticity(u,v,X_grid,Y_grid):    
    dudy = np.gradient(u,Y_grid[0,:],axis=1)
    dvdx = np.gradient(v,X_grid[:,0],axis=0)
    return dvdx-dudy

def Qcriterion(u,v,X_grid,Y_grid):
    dudx = np.gradient(u,X_grid[:,0],axis=0)
    dudy = np.gradient(u,Y_grid[0,:],axis=1)
    dvdx = np.gradient(u,X_grid[:,0],axis=0)
    dvdy = np.gradient(u,Y_grid[0,:],axis=1)
    return dudx*dvdy-dudy*dvdx-0.5*np.power(dudx+dvdy,2.0)
