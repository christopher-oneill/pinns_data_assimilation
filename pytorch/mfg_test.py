

import sys
sys.path.append('C:/projects/pinns_local/code/')


import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plot
import h5py

import time

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists

# functions


def plot_err(epoch,model_RANS):
    global p_grid
    global X_grid
    global Y_grid
    global i_test
    global o_test_grid
    global saveFig
    global fig_dir
    global d

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<np.power(d/2.0,2.0)

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = o_test_grid
    o_test_grid_temp[:,:,5] = p_grid
    o_test_grid_temp[cylinder_mask] = np.NaN
    i_test_temp = torch.from_numpy(i_test[:])
    pred_test = model_RANS(i_test_temp)
    
    pred_test_grid = np.reshape(pred_test.detach().numpy(),[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask] = np.NaN

    plot.close('all')

    err_test = o_test_grid_temp-pred_test_grid

    # NS residual

    plot_save_exts = ['_ux.png','_uy.png','_uxux.png','_uxuy.png','_uyuy.png','_p.png']

    # quantities
    for i in range(6):

        o_test_max = np.nanmax(np.abs(o_test_grid_temp[:,:,i].ravel()))
        o_test_levels = np.linspace(-o_test_max,o_test_max,21)

        plot.figure(epoch)
        plot.title('Full Resolution')
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,i],levels=o_test_levels)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,i],levels=21,vmin=-o_test_max,vmax=o_test_max)
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        plot.contourf(X_grid,Y_grid,err_test[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        if saveFig:
            plot.savefig(fig_dir+'ep'+str(epoch)+plot_save_exts[i],dpi=300)





# load the training data 
HOMEDIR = 'C:/projects/pinns_narval/sync/'
# read the data
base_dir = HOMEDIR+'data/mazi_fixed_grid/'
global savedir
savedir = HOMEDIR+'output/mfg_torch_test001/'
create_directory_if_not_exists(savedir)
global fig_dir
fig_dir = savedir + 'figures/'
create_directory_if_not_exists(fig_dir)

reynoldsStressFile = h5py.File(base_dir+'reynoldsStress.mat','r')
meanVelocityFile = h5py.File(base_dir+'meanVelocity.mat','r')
configFile = h5py.File(base_dir+'configuration.mat','r')
meanPressureFile = h5py.File(base_dir+'meanPressure.mat','r')



global X_grid
global Y_grid

x = np.array(configFile['X_vec'][0,:])
x_test = x
X_grid = np.array(configFile['X_grid'])
y = np.array(configFile['X_vec'][1,:])
Y_grid = np.array(configFile['Y_grid'])
y_test = y
global d
d = np.array(configFile['cylinderDiameter'])



ux = np.array(meanVelocityFile['meanVelocity'][0,:]).transpose()
uy = np.array(meanVelocityFile['meanVelocity'][1,:]).transpose()

uxux = np.array(reynoldsStressFile['reynoldsStress'][0,:]).transpose()
uxuy = np.array(reynoldsStressFile['reynoldsStress'][1,:]).transpose()
uyuy = np.array(reynoldsStressFile['reynoldsStress'][2,:]).transpose()

MAX_ux = np.max(ux.ravel())
MAX_uy = np.max(uy.ravel())
MAX_uxux = np.max(uxux.ravel())
MAX_uxuy = np.max(uxuy.ravel())
MAX_uyuy = np.max(uyuy.ravel())
MAX_p = 1.0

# set points inside the cylinder to zero

cylinder_mask = np.reshape(np.power(x,2.0)+np.power(y,2.0)<np.power(d/2.0,2.0),[x.shape[0],])

ux[cylinder_mask] = 0.0
uy[cylinder_mask] = 0.0
uxux[cylinder_mask] = 0.0
uxuy[cylinder_mask] = 0.0
uyuy[cylinder_mask] = 0.0

# create the test data
global o_test_grid
o_test_grid = np.reshape(np.hstack((ux.reshape(-1,1)/MAX_ux,uy.reshape(-1,1)/MAX_uy,uxux.reshape(-1,1)/MAX_uxux,uxuy.reshape(-1,1)/MAX_uxuy,uyuy.reshape(-1,1)/MAX_uyuy)),[X_grid.shape[0],X_grid.shape[1],5])
ux_grid = np.reshape(ux,X_grid.shape)
ux_train = torch.from_numpy(ux/MAX_ux)

MAX_x = np.max(X_grid)
MAX_y = np.max(Y_grid)
x_test = X_grid/MAX_x
y_test = Y_grid/MAX_x
global i_test
i_test = np.hstack((x_test.reshape(-1,1),y_test.reshape(-1,1)))


from pinns_data_assimilation.lib.downsample import compute_downsample_inds_center
supersample_factor=1
# if we are downsampling and then upsampling, downsample the source data
if supersample_factor>1:
    n_x = np.array(configFile['x_grid']).size
    n_y = np.array(configFile['y_grid']).size
    downsample_inds, ndx,ndy = compute_downsample_inds_center(supersample_factor,X_grid[:,0],Y_grid[0,:].transpose())
    x = x[downsample_inds]
    y = y[downsample_inds]
    ux = ux[downsample_inds]
    uy = uy[downsample_inds]
    uxux = uxux[downsample_inds]
    uxuy = uxuy[downsample_inds]
    uyuy = uyuy[downsample_inds]


o_train = np.hstack((ux.reshape(-1,1)/MAX_ux,uy.reshape(-1,1)/MAX_uy,uxux.reshape(-1,1)/MAX_uxux,uxuy.reshape(-1,1)/MAX_uxuy,uyuy.reshape(-1,1)/MAX_uyuy))
i_train = np.hstack((x.reshape(-1,1)/MAX_x,y.reshape(-1,1)/MAX_x))

global p_grid
p = np.array(meanPressureFile['meanPressure']).transpose()
p = p[:,0]
p_grid = np.reshape(p,X_grid.shape)/MAX_p


global saveFig
saveFig = True


o_train = torch.from_numpy(o_train)
i_train = torch.from_numpy(i_train)

# scaling parameters

fs=10.0
# create a dummy object to contain all the scaling parameters
class UserScalingParameters(object):
    pass
ScalingParameters = UserScalingParameters()
ScalingParameters.fs = fs
ScalingParameters.MAX_x = np.max(x.flatten())
ScalingParameters.MAX_y = ScalingParameters.MAX_x # we scale based on the largest spatial dimension
ScalingParameters.MAX_ux = MAX_ux # we scale based on the max of the whole output array
ScalingParameters.MAX_uy = MAX_uy
ScalingParameters.MIN_x = np.min(x.flatten())
ScalingParameters.MIN_y = np.min(y.flatten())
ScalingParameters.MIN_ux = np.min(ux.flatten())
ScalingParameters.MIN_uy = np.min(uy.flatten())
ScalingParameters.MAX_uxppuxpp = MAX_uxux
ScalingParameters.MAX_uxppuypp = MAX_uxuy
ScalingParameters.MAX_uyppuypp = MAX_uyuy
ScalingParameters.nu_mol = 0.0066667
ScalingParameters.MAX_p= MAX_p # estimated maximum pressure, we should
ScalingParameters.batch_size = 32 
ScalingParameters.physics_loss_coefficient = np.float64(0.0)
ScalingParameters.boundary_loss_coefficient = np.float64(0.0)
ScalingParameters.data_loss_coefficient = np.float64(1.0)
ScalingParameters.pressure_loss_coefficient=np.float64(0.0)


class ResidualLayer(torch.nn.Module):
    def __init__(self,size_in,size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        weights = torch.zeros(size_out, size_in,dtype=torch.float64)
        self.weights = torch.nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.zeros(size_out,dtype=torch.float64)
        self.bias = torch.nn.Parameter(bias)

    def forward(self,x):
        if (self.size_in>self.size_out):
            # we must truncate the input vector for residual
            xt = torch.narrow(x,-1,0,self.size_out)
        elif(self.size_in<self.size_out):
            xt = torch.nn.functional.pad(x,(self.size_out-self.size_in))
        else:
            xt=x

        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(torch.add(w_times_x, self.bias),xt) 





# set up the model
nodes=200
model = torch.nn.Sequential(
    torch.nn.Linear(2,nodes,True,dtype=torch.float64),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    ResidualLayer(nodes,nodes),
    torch.nn.SiLU(),
    torch.nn.Linear(nodes,1,True,dtype=torch.float64),
    
)

gpu = torch.device("cuda")
cpu = torch.device("cpu")
print("Cuda is available: ",torch.cuda.is_available())
model.to(cpu)

#i_train = i_train.cuda()
#o_train = o_train.cuda()

#print(i_train.get_device())
#print(o_train.get_device())

# gradient function definitions

def compute_gradient(single):
    return torch.func.jacfwd(model)(single)

def compute_second_gradient(single):
    return torch.func.jacfwd(torch.func.jacfwd(model))(single)

compute_gradient_batch = torch.func.vmap(compute_gradient,in_dims=(0,))
compute_second_gradient_batch = torch.func.vmap(compute_second_gradient,in_dims=(0,))

def navier_stokes_RANS(x_batch):

    u = model(x_batch)

    # compute the gradients 
    u_i = compute_gradient_batch(x_batch)
    u_ii = compute_second_gradient_batch(x_batch)

    # scale the quantities
    ux = u[:,0]*ScalingParameters.MAX_ux
    uy = u[:,1]*ScalingParameters.MAX_uy

    ux_x = u_i[:,0,0]*ScalingParameters.MAX_ux/ScalingParameters.MAX_x
    ux_y = u_i[:,0,1]*ScalingParameters.MAX_ux/ScalingParameters.MAX_y
    uy_x = u_i[:,1,0]*ScalingParameters.MAX_uy/ScalingParameters.MAX_x
    uy_y = u_i[:,1,1]*ScalingParameters.MAX_uy/ScalingParameters.MAX_y

    ux_xx = u_ii[:,0,0,0]*ScalingParameters.MAX_ux/(ScalingParameters.MAX_x*ScalingParameters.MAX_x)
    ux_yy = u_ii[:,0,1,1]*ScalingParameters.MAX_ux/(ScalingParameters.MAX_y*ScalingParameters.MAX_y)
    uy_xx = u_ii[:,1,0,0]*ScalingParameters.MAX_uy/(ScalingParameters.MAX_x*ScalingParameters.MAX_x)
    uy_yy = u_ii[:,1,1,1]*ScalingParameters.MAX_uy/(ScalingParameters.MAX_y*ScalingParameters.MAX_y)

    p_x = u_i[:,5,0]*ScalingParameters.MAX_p/ScalingParameters.MAX_x
    p_y = u_i[:,5,1]*ScalingParameters.MAX_p/ScalingParameters.MAX_y

    uxux_x = u_i[:,2,0]*ScalingParameters.MAX_uxppuxpp/ScalingParameters.MAX_x
    uxuy_x = u_i[:,3,0]*ScalingParameters.MAX_uxppuypp/ScalingParameters.MAX_y
    uxuy_y = u_i[:,3,1]*ScalingParameters.MAX_uxppuypp/ScalingParameters.MAX_x
    uyuy_y = u_i[:,4,1]*ScalingParameters.MAX_uyppuypp/ScalingParameters.MAX_y

    # compute the governing equations
    loss_x = (torch.mul(ux,ux_x) + torch.mul(uy,ux_y)) + p_x + (uxux_x+uxuy_y) - ScalingParameters.nu_mol*(ux_xx+ux_yy) # x momentum
    loss_y = (torch.mul(ux,uy_x) + torch.mul(uy,uy_y)) + p_y + (uxuy_x+uyuy_y) - ScalingParameters.nu_mol*(uy_xx+uy_yy) # y momentum
    loss_mass = ux_x+uy_y # continuity equation

    return loss_x, loss_y, loss_mass



learning_rate = 1E-2
batch_size = 32
nu_mol = 0.0066667

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
nbatch = np.ceil(i_train.shape[0]/(1.0*batch_size))
def do_epoch():
    for batch in range(0,np.int64(nbatch)):    
        batch_inds = torch.arange(batch*batch_size,torch.min((torch.tensor([(batch+1)*batch_size,i_train.shape[0]]))),device=cpu)


        x_batch = i_train[batch_inds,:]
        #x_batch.requires_grad=True
        
        u = model(x_batch)                   
        optimizer.zero_grad()
        # data loss 
        data_loss =torch.mean(torch.mean(torch.square(u-ux_train[batch_inds]),0),0)
        
        data_loss.backward()
        optimizer.step()
    return data_loss


ScalingParameters.physics_loss_coefficient = 0.0
for n_epoch in range(30):
    epoch_time = time.time()
    loss = do_epoch()  
    print('Epoch ',str(n_epoch),' Epoch duration ',str(time.time()-epoch_time),' seconds. Loss = ',np.format_float_scientific(loss.cpu().detach().numpy()))

    #plot_err(n_epoch,model)

for g in optimizer.param_groups:
    g['lr'] = 1E-3

for n_epoch in range(30):
    epoch_time = time.time()
    loss = do_epoch()  
    print('Epoch ',str(n_epoch),' Epoch duration ',str(time.time()-epoch_time),' seconds. Loss = ',np.format_float_scientific(loss.cpu().detach().numpy()))
    #plot_err(n_epoch,model)

for g in optimizer.param_groups:
    g['lr'] = 1E-4

for n_epoch in range(30):
    epoch_time = time.time()
    loss = do_epoch()  
    print('Epoch ',str(n_epoch),' Epoch duration ',str(time.time()-epoch_time),' seconds. Loss = ',np.format_float_scientific(loss.cpu().detach().numpy()))
    #plot_err(n_epoch,model)


for g in optimizer.param_groups:
    g['lr'] = 1E-5

for n_epoch in range(30):
    epoch_time = time.time()
    loss = do_epoch()  
    print('Epoch ',str(n_epoch),' Epoch duration ',str(time.time()-epoch_time),' seconds. Loss = ',np.format_float_scientific(loss.cpu().detach().numpy()))
    #plot_err(n_epoch,model)


for g in optimizer.param_groups:
    g['lr'] = 1E-6

for n_epoch in range(30):
    epoch_time = time.time()
    loss = do_epoch()  
    print('Epoch ',str(n_epoch),' Epoch duration ',str(time.time()-epoch_time),' seconds. Loss = ',np.format_float_scientific(loss.cpu().detach().numpy()))
    #plot_err(n_epoch,model)


