import numpy as np


def vorticity(u,v,X_grid,Y_grid):    
    dudy = np.gradient(u,Y_grid[0,:],axis=1)
    dvdx = np.gradient(v,X_grid[:,0],axis=0)
    return dvdx-dudy

def Qcriterion(u,v,X_grid,Y_grid):
    dudx = np.gradient(u,X_grid[:,0],axis=0)
    dudy = np.gradient(u,Y_grid[0,:],axis=1)
    dvdx = np.gradient(v,X_grid[:,0],axis=0)
    dvdy = np.gradient(v,Y_grid[0,:],axis=1)
    #return dudx*dvdy-dudy*dvdx-0.5*np.power(dudx+dvdy,2.0)
    
    #return 0.5*(np.power(dvdx-dudy
        #  0.5*( (nabla cdot u)^2- (nabla u : nabla u ^T))
        # or
        # 0.5*(tr(nabla u)^2  - tr(nabla u cdot nabla u))
        #0.5*((dudx*dudx+2*dudx*dvdy+dvdy*dvdy)-(dudx*dudx+dudy*dudy+dvdx*dvdx+dvdy*dvdy))
    return dudx*dvdy - 0.5*(dudy*dudy+dvdx*dvdx)