

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plot
import matplotlib
import time

HOMEDIR = 'C:/projects/pinns_narval/sync/'
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.dft import dft
from pinns_data_assimilation.lib.decomposition import POD
from pinns_data_assimilation.lib.decomposition import fir_SPOD

# domain
x = np.linspace(-1,1,201)
y = 1.0*x
X_grid,Y_grid=np.meshgrid(x,y)
t = np.linspace(0,10.24,256)

if False:
    fig,axs=plot.subplots(2,1)
    axs[0].contourf(X_grid,Y_grid,X_grid,levels=21,cmap= matplotlib.colormaps['bwr'])
    axs[1].contourf(X_grid,Y_grid,Y_grid,levels=21,cmap= matplotlib.colormaps['bwr'])

# create a synthetic signal
# the TCs

a1 = 0.75*np.sin(2*np.pi*t+np.pi/4)+0.25*np.sin(6*np.pi*t) # POD will give one mode here, DFT two
a2 = 0.25*np.sin(1.5*2*np.pi*t) # same for both DFT and POD

phi_1 = 2.0/(1+np.exp(np.power(X_grid,2.0)/6+np.power(Y_grid,2.0)/2.0))
phi_2 = np.sin(2*np.pi*X_grid)

if True:
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(t,a1)
    ax.plot(t,a2)

    fig,axs=plot.subplots(2,1)
    plot_phi1=axs[0].contourf(X_grid,Y_grid,phi_1,levels=21,cmap= matplotlib.colormaps['bwr'])
    axs[0].set_aspect('equal')
    plot.colorbar(plot_phi1)
    plot_phi2=axs[1].contourf(X_grid,Y_grid,phi_2,levels=21,cmap= matplotlib.colormaps['bwr'])
    axs[1].set_aspect('equal')
    plot.colorbar(plot_phi2)

# compute the signal
Tsig = np.reshape(phi_1,[phi_1.shape[0],phi_1.shape[1],1])*np.reshape(a1,[1,1,a1.shape[0]]) + np.reshape(phi_2,[phi_2.shape[0],phi_2.shape[1],1])*np.reshape(a2,[1,1,a2.shape[0]])

if False:
    # plot a movie of the signal
    fig = plot.figure()
    ax = plot.axes()
    plot.ion()
    cf = ax.contourf(X_grid,Y_grid,Tsig[:,:,0])
    plot.show()
    for i in range(1,Tsig.shape[0]):
        ax.clear()
        ax.contourf(X_grid,Y_grid,Tsig[:,:,i])
        plot.show()
        plot.pause(0.2)



# do the DFT 
Tsig_prime = Tsig - np.reshape(np.mean(Tsig,2),[Tsig.shape[0],Tsig.shape[1],1])

phi_ij,f_ij = dft(np.reshape(Tsig_prime,[phi_1.shape[0]*phi_1.shape[1],t.shape[0]]))
half_index = np.int64(t.shape[0]/2)
phi_ij=phi_ij[:,0:half_index]/t.shape[0]
f_ij = f_ij[0:half_index]*(1/((t[1]-t[0])))
phi_ij = np.reshape(phi_ij,[phi_1.shape[0],phi_1.shape[1],half_index])


if True:
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(f_ij,np.abs(phi_ij[75,75,:]))
    ax.set_yscale('log')

# do the POD


Phi_POD, Lambda_POD, Ak_POD = POD(np.reshape(Tsig_prime,[phi_1.shape[0]*phi_1.shape[1],t.shape[0]])) 
print(Phi_POD.shape)

if True:
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(np.arange(Lambda_POD.shape[0]),Lambda_POD)
    ax.set_yscale('log')

if True:
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(t,Ak_POD[:,0])
    ax.plot(t,Ak_POD[:,1])

    fig,axs=plot.subplots(2,1)
    plot_phi1=axs[0].contourf(X_grid,Y_grid,np.reshape(Phi_POD[:,0],[phi_1.shape[0],phi_1.shape[1]]),levels=21,cmap= matplotlib.colormaps['bwr'])
    axs[0].set_aspect('equal')
    plot.colorbar(plot_phi1)
    plot_phi2=axs[1].contourf(X_grid,Y_grid,np.reshape(Phi_POD[:,1],[phi_1.shape[0],phi_1.shape[1]]),levels=21,cmap= matplotlib.colormaps['bwr'])
    axs[1].set_aspect('equal')
    plot.colorbar(plot_phi2)

# do the SPOD

Phi_SPOD_DFT, Lambda_SPOD_DFT, Ak_SPOD_DFT = fir_SPOD(np.reshape(Tsig_prime,[phi_1.shape[0]*phi_1.shape[1],t.shape[0]]),t.shape[0])

if True:
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(np.arange(Lambda_SPOD_DFT.shape[0]),Lambda_SPOD_DFT)
    #ax.plot(np.arange(Lambda_SPOD_DFT.shape[0]),np.sort())
    ax.set_yscale('log')

    # mode 1,2
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(t,Ak_SPOD_DFT[:,0])
    ax.plot(t,Ak_SPOD_DFT[:,1])

    # spectra for mode 1,2
    sp1_hat,f_1 = dft(Ak_SPOD_DFT[:,0])
    sp1_hat=sp1_hat[0:half_index]/t.shape[0]
    f_1 = f_1[0:half_index]*(1/((t[1]-t[0])))
    sp2_hat,f_2 = dft(Ak_SPOD_DFT[:,1])
    sp2_hat=sp2_hat[0:half_index]/t.shape[0]
    f_2 = f_2[0:half_index]*(1/((t[1]-t[0])))


    fig = plot.figure()
    ax = plot.axes()
    ax.plot(f_1,np.abs(sp1_hat))
    ax.plot(f_2,np.abs(sp2_hat))

    # mode 2,3
    fig = plot.figure()
    ax = plot.axes()
    ax.plot(t,Ak_SPOD_DFT[:,2])
    ax.plot(t,Ak_SPOD_DFT[:,3])

    # spectra for mode 1,2
    sp1_hat,f_1 = dft(Ak_SPOD_DFT[:,2])
    sp1_hat=sp1_hat[0:half_index]/t.shape[0]
    f_1 = f_1[0:half_index]*(1/((t[1]-t[0])))
    sp2_hat,f_2 = dft(Ak_SPOD_DFT[:,3])
    sp2_hat=sp2_hat[0:half_index]/t.shape[0]
    f_2 = f_2[0:half_index]*(1/((t[1]-t[0])))


    fig = plot.figure()
    ax = plot.axes()
    ax.plot(f_1,np.abs(sp1_hat))
    ax.plot(f_2,np.abs(sp2_hat))


    fig = plot.figure()
    ax = plot.axes()
    ax.plot(t,Ak_SPOD_DFT[:,4])
    ax.plot(t,Ak_SPOD_DFT[:,5])

        # spectra for mode 1,2
    sp1_hat,f_1 = dft(Ak_SPOD_DFT[:,4])
    sp1_hat=sp1_hat[0:half_index]/t.shape[0]
    f_1 = f_1[0:half_index]*(1/((t[1]-t[0])))
    sp2_hat,f_2 = dft(Ak_SPOD_DFT[:,5])
    sp2_hat=sp2_hat[0:half_index]/t.shape[0]
    f_2 = f_2[0:half_index]*(1/((t[1]-t[0])))


    fig = plot.figure()
    ax = plot.axes()
    ax.plot(f_1,np.abs(sp1_hat))
    ax.plot(f_2,np.abs(sp2_hat))


plot.show()