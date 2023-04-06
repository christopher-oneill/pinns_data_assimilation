import numpy as np
import h5py
import matplotlib.pyplot as plot
import scipy as sp

base_dir = 'C:/projects/pinns_galerkin_viv/data/mazi_fixed/'
meanFieldFile = h5py.File(base_dir+'meanField.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')
reynoldsStressFile = h5py.File(base_dir+'reynoldsStresses.mat','r')
meanGradientsFile = h5py.File(base_dir+'meanGradients.mat','r')

print_name = '20230313_unconstr_nu_dense30x10_b32_ep530_st3'
predfilename = base_dir+'20230313_unconstr_nu/dense30x10_b32_ep530_st3_pred.mat'
predFile =  h5py.File(predfilename,'r')


ux = np.array(meanFieldFile['meanField'][0,:]).transpose()
ux = ux[:,0]
uy = np.array(meanFieldFile['meanField'][1,:]).transpose()
uy = uy[:,0]
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]
upup = np.array(reynoldsStressFile['reynoldsStresses'][0,:]).transpose()
upvp = np.array(reynoldsStressFile['reynoldsStresses'][1,:]).transpose()
vpvp = np.array(reynoldsStressFile['reynoldsStresses'][2,:]).transpose()
dudy = (np.array(meanGradientsFile['meanGradients'][1,:]).transpose())
dvdx = np.array(meanGradientsFile['meanGradients'][2,:]).transpose()
x = np.array(configFile['X'][0,:])
y = np.array(configFile['X'][1,:])
d = np.array(configFile['cylinderDiameter'])[0]

MAX_nut= 0.3 # estimated maximum of nut # THIS VALUE is internally multiplied with 0.001 (mm and m)
MAX_p= 1 # estimated maximum pressure
nu_mol = 0.0066667
ux_pred = np.array(predFile['pred'][:,0])*np.max(ux)
uy_pred = np.array(predFile['pred'][:,1])*np.max(uy)
p_pred = np.array(predFile['pred'][:,2])*MAX_p
#nu_pred =  np.power(np.array(predFile['pred'][:,3]),2)*MAX_nut
nu_pred =  np.array(predFile['pred'][:,3])*MAX_nut
# compute the estimated reynolds stress

upvp_pred = -np.multiply(np.reshape(nu_pred+nu_mol,[nu_pred.shape[0],1]),dudy+dvdx)

print('ux.shape: ',ux.shape)
print('uy.shape: ',uy.shape)
print('p.shape: ',p.shape)
print('upvp.shape: ',upvp.shape)
print('dudy.shape: ',dudy.shape)
print('dvdx.shape: ',dvdx.shape)

print('x.shape: ',x.shape)
print('y.shape: ',y.shape)
print('d: ',d.shape)

print('ux_pred.shape: ',ux_pred.shape)
print('uy_pred.shape: ',uy_pred.shape)
print('p_pred.shape: ',p_pred.shape)
print('nu_pred.shape: ',nu_pred.shape)
print('upvp_pred.shape: ',upvp_pred.shape)

# note that the absolute value of the pressure doesnt matter, only grad p and grad2 p, so subtract the mean 
p_pred = p_pred-(1/3)*(upup+vpvp)#p_pred - (1/3)*(upup+vpvp)

print(np.max(upvp))
print(np.min(upvp))

x_vec = np.arange(-2,10,0.01)
y_vec = np.arange(-2,2,0.01)
[x_grid,y_grid] = np.meshgrid(x_vec,y_vec)
ux_grid = sp.interpolate.griddata((x,y),ux,(x_grid,y_grid),method='cubic')
uy_grid = sp.interpolate.griddata((x,y),uy,(x_grid,y_grid),method='cubic')
p_grid = sp.interpolate.griddata((x,y),p,(x_grid,y_grid),method='cubic')
upvp_grid = sp.interpolate.griddata((x,y),upvp,(x_grid,y_grid),method='cubic')
ux_pred_grid = sp.interpolate.griddata((x,y),ux_pred,(x_grid,y_grid),method='cubic')
uy_pred_grid = sp.interpolate.griddata((x,y),uy_pred,(x_grid,y_grid),method='cubic')
p_pred_grid = sp.interpolate.griddata((x,y),p_pred,(x_grid,y_grid),method='cubic')
nu_pred_grid = sp.interpolate.griddata((x,y),nu_pred,(x_grid,y_grid),method='cubic')
upvp_pred_grid =  sp.interpolate.griddata((x,y),upvp_pred,(x_grid,y_grid),method='cubic')[:,:,0]
ux_diff_grid = sp.interpolate.griddata((x,y),ux-ux_pred,(x_grid,y_grid),method='cubic')
uy_diff_grid = sp.interpolate.griddata((x,y),uy-uy_pred,(x_grid,y_grid),method='cubic')
p_diff_grid = sp.interpolate.griddata((x,y),p-p_pred,(x_grid,y_grid),method='cubic')
upvp_diff_grid = sp.interpolate.griddata((x,y),upvp-upvp_pred[:,0],(x_grid,y_grid),method='cubic')


ux_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
ux_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
nu_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
upvp_pred_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
ux_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
uy_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
p_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN
upvp_diff_grid[np.power(np.power(x_grid,2)+np.power(y_grid,2),0.5)<=0.5*d]=np.NaN

f1_max = np.nanmax(np.array([np.nanmax(ux_grid)]))
f1_min = np.nanmin(np.array([np.nanmin(ux_grid)]))
f1_lims = np.nanmax(np.abs(np.array([f1_max,f1_min])))
f1_levels = np.linspace(-f1_lims,f1_lims,21)
fig = plot.figure(1)
ax = fig.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,ux_grid,levels=f1_levels)
plot.set_cmap('bwr')
plot.colorbar()
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.ylabel('y/D')
fig.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,ux_pred_grid,levels=f1_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.axis('equal')
fig.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,ux_diff_grid,levels=f1_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
plot.xlabel('x/D')
plot.axis('equal')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.savefig(base_dir+'figures/'+print_name+'_mean_ux.png',dpi=300)



f2_max = np.nanmax(np.array([np.nanmax(uy_grid)]))
f2_min = np.nanmin(np.array([np.nanmin(uy_grid)]))
f2_lims = np.nanmax(np.abs(np.array([f2_max,f2_min])))
f2_levels = np.linspace(-f2_lims,f2_lims,21)
fig2 = plot.figure(2)
fig2.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,uy_grid,levels=f2_levels)
plot.set_cmap('bwr')
plot.colorbar()
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.ylabel('y/D')
fig2.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,uy_pred_grid,levels=f2_levels)
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
plot.contourf(x_grid,y_grid,uy_diff_grid,levels=f2_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
plot.xlabel('x/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.axis('equal')
plot.savefig(base_dir+'figures/'+print_name+'_mean_uy.png',dpi=300)

f3_max = np.nanmax(np.array([np.nanmax(p_grid)]))
f3_min = np.nanmin(np.array([np.nanmin(p_grid)]))
f3_lims = np.nanmax(np.abs(np.array([f3_max,f3_min])))
f3_levels = np.linspace(-f3_lims,f3_lims,21)
fig3 = plot.figure(3)
fig3.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,p_grid,levels=f3_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
fig3.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,p_pred_grid,21)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.ylabel('y/D')
fig3.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,p_diff_grid,21)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
plot.ylabel('y/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.xlabel('x/D')
plot.savefig(base_dir+'figures/'+print_name+'_mean_p.png',dpi=300)

f4_max = np.nanmax(np.array([np.nanmax(upvp_grid)]))
f4_min = np.nanmin(np.array([np.nanmin(upvp_grid)]))
f4_lims = np.nanmax(np.abs(np.array([f4_max,f4_min])))
f4_levels = np.linspace(-f4_lims,f4_lims,21)
fig4 = plot.figure(4)
fig4.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,upvp_grid,levels=f4_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
fig4.add_subplot(3,1,2)
plot.contourf(x_grid,y_grid,upvp_pred_grid,levels=f4_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.ylabel('y/D')
fig4.add_subplot(3,1,3)
plot.contourf(x_grid,y_grid,upvp_diff_grid,levels=f4_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.axis('equal')
plot.ylabel('y/D')
plot.xlabel('x/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.savefig(base_dir+'figures/'+print_name+'_mean_upvp.png',dpi=300)

f5_max = np.nanmax(np.array([np.nanmax(nu_pred_grid)]))
f5_min = np.nanmin(np.array([np.nanmin(nu_pred_grid)]))
f5_lims = np.nanmax(np.abs(np.array([f5_max,f5_min])))
f5_levels = np.linspace(-f5_lims,f5_lims,21)
fig5 = plot.figure(5)
fig5.add_subplot(3,1,1)
plot.axis('equal')
plot.contourf(x_grid,y_grid,nu_pred_grid,levels=f5_levels)
plot.set_cmap('bwr')
plot.colorbar()
plot.ylabel('y/D')
plot.xlabel('x/D')
ax=plot.gca()
ax.set_xlim(left=-2.0,right=10.0)
ax.set_ylim(bottom=-2.0,top=2.0)
plot.savefig(base_dir+'figures/'+print_name+'_mean_nu.png',dpi=300)
plot.show()