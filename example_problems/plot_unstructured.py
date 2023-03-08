import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp


meanfilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/meanField.mat'
meanFieldFile = h5py.File(meanfilename,'r')
configfilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/configuration.mat'
configFile = h5py.File(configfilename,'r')
meanPressureFilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/meanPressure.mat'
meanPressureFile = h5py.File(meanPressureFilename,'r')

predfilename = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/20230303014906_tmp/dense30x10_b32_ep900_st3_pred.mat'
predFile =  h5py.File(predfilename,'r')


ux = np.array(meanFieldFile['meanField'][0,:]).transpose()
uy = np.array(meanFieldFile['meanField'][1,:]).transpose()
p = np.array(meanPressureFile['meanPressure']).transpose()

x = np.array(configFile['X'][0,:])
y = np.array(configFile['X'][1,:])
d = np.array(configFile['cylinderDiameter'])[0]


ux_pred = np.array(predFile['pred'][:,0])*np.max(ux)
uy_pred = np.array(predFile['pred'][:,1])*np.max(uy)
p_pred = np.array(predFile['pred'][:,2])
nu_pred =  np.array(predFile['pred'][:,3])

print('ux.shape: ',ux.shape)
print('uy.shape: ',uy.shape)
print('p.shape: ',p.shape)
print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)

print('ux_pred.shape: ',ux_pred.shape)
print('uy_pred.shape: ',uy_pred.shape)
print('p_pred.shape: ',p_pred.shape)

x_vec = np.arange(-2,10,0.01)
y_vec = np.arange(-2,2,0.01)
[x_grid,y_grid] = np.meshgrid(x_vec,y_vec)
ux_grid = sp.interpolate.griddata((x,y),ux,(x_grid,y_grid),method='cubic')[:,:,0]
uy_grid = sp.interpolate.griddata((x,y),uy,(x_grid,y_grid),method='cubic')[:,:,0]
p_grid = sp.interpolate.griddata((x,y),p,(x_grid,y_grid),method='cubic')[:,:,0]
ux_pred_grid = sp.interpolate.griddata((x,y),ux_pred,(x_grid,y_grid),method='cubic')
uy_pred_grid = sp.interpolate.griddata((x,y),uy_pred,(x_grid,y_grid),method='cubic')
p_pred_grid = sp.interpolate.griddata((x,y),p_pred,(x_grid,y_grid),method='cubic')
nu_pred_grid =  sp.interpolate.griddata((x,y),nu_pred,(x_grid,y_grid),method='cubic')
ux_diff_grid = sp.interpolate.griddata((x,y),ux[:,0]-ux_pred,(x_grid,y_grid),method='cubic')
uy_diff_grid = sp.interpolate.griddata((x,y),uy[:,0]-uy_pred,(x_grid,y_grid),method='cubic')
p_diff_grid = sp.interpolate.griddata((x,y),p[:,0]-p_pred,(x_grid,y_grid),method='cubic')

ux_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
ux_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
ux_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN

fig = plot.figure(1)
ax = fig.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,ux_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
fig.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,ux_pred_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
ax = fig.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,ux_diff_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')


fig2 = plot.figure(2)
fig2.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,uy_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
fig2.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,uy_pred_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
fig2.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,uy_diff_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')

fig3 = plot.figure(3)
fig3.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,p_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
fig3.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,p_pred_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
fig3.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,p_diff_grid,20)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')

fig4 = plot.figure(4)
fig4.add_subplot(1,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,nu_pred_grid,20)
plot.set_cmap('bwr')
plot.colorbar()

plot.show()