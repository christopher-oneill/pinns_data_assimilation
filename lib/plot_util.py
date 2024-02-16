
import numpy as np
import matplotlib.pyplot as plot



def plot_gradients():
    global model_RANS
    global training_steps
    global p_grid
    global p_x_grid
    global p_y_grid
    global X_grid
    global Y_grid
    global i_test
    global i_test_large
    global o_test_grid
    global saveFig
    global fig_dir
    global d
    global ScalingParameters

    start_time = time.time()
    for i in range(20):
        mx1,my1,mass1 = RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,i_test[:])
    print('Symbolic: ',(time.time()-start_time)/float(i+1))

    start_time = time.time()
    for i in range(20):
        mx2,my2,mass2 = RANS_reynolds_stress_cartesian_GradTape(model_RANS,ScalingParameters,i_test[:])
    print('Gradtape: ',(time.time()-start_time)/float(i+1))
    
    mx1 = np.reshape(mx1,X_grid.shape)
    my1 = np.reshape(my1,X_grid.shape)
    mass1 = np.reshape(mass1,X_grid.shape)
    
    mx2 = np.reshape(mx2,X_grid.shape)
    my2 = np.reshape(my2,X_grid.shape)
    mass2 = np.reshape(mass2,X_grid.shape)

    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,mx1,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,mx2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,mx1-mx2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_mx.png',dpi=300)
    
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,my1,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,my2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,my1-my2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_my.png',dpi=300)
    
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,mass1,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,mass2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,mass1-mass2,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_mass.png',dpi=300)


def plot_large():
    global i_test_large
    global X_grid_large
    global Y_grid_large
    global training_steps
    global model_RANS
    global ScalingParameters
    plot_save_exts = ['_ux_large.png','_uy_large.png','_uxux_large.png','_uxuy_large.png','_uyuy_large.png','_p_large.png']
    cylinder_mask_large = (np.power(X_grid_large,2.0)+np.power(Y_grid_large,2.0))<=np.power(d/2.0,2.0)
    pred_test_large = model_RANS.predict(i_test_large,batch_size=1000)
    pred_test_large_grid = 1.0*np.reshape(pred_test_large,[X_grid_large.shape[0],X_grid_large.shape[1],6])
    pred_test_large_grid[cylinder_mask_large,:] = np.NaN

    for i in range(6):
        plot.figure(1)
        plot.contourf(X_grid_large,Y_grid_large,pred_test_large_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts[i],dpi=300)
        plot.close(1)

def plot_NS_large():
    global i_test_large
    global X_grid_large
    global Y_grid_large
    global training_steps
    global model_RANS
    global ScalingParameters
    cylinder_mask_large = (np.power(X_grid_large,2.0)+np.power(Y_grid_large,2.0))<=np.power(d/2.0,2.0)
    mx_large,my_large,mass_large = batch_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,i_test_large,1000)
    mx_large = 1.0*np.reshape(mx_large,X_grid_large.shape)
    mx_large[cylinder_mask_large] = np.NaN
    my_large = 1.0*np.reshape(my_large,X_grid_large.shape)
    my_large[cylinder_mask_large] = np.NaN
    mass_large = 1.0*np.reshape(mass_large,X_grid_large.shape)
    mass_large[cylinder_mask_large] = np.NaN

    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,mx_large,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_mx_large.png',dpi=300)
    plot.close(1)
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,my_large,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_my_large.png',dpi=300)
    plot.close(1)
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,mass_large,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.savefig(fig_dir+'ep'+str(training_steps)+'_mass_large.png',dpi=300)
    plot.close(1)

def plot_pressure_gradients():
    p_x_pred, p_y_pred = RANS_pressure_gradients(model_RANS,i_test[:])
    p_x_pred = 1.0*np.reshape(p_x_pred,X_grid.shape)
    #p_x_pred[cylinder_mask] = np.NaN
    p_y_pred = 1.0*np.reshape(p_y_pred,X_grid.shape)
    #p_y_pred[cylinder_mask] = np.NaN

    err_p_x = p_x_grid - p_x_pred
    plot.figure(1)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,p_x_grid,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,p_x_pred,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    e_test_min = np.nanpercentile(err_p_x.ravel(),0.1)
    e_test_max = np.nanpercentile(err_p_x.ravel(),99.9)
    e_test_level = np.max([abs(e_test_min),abs(e_test_max)])
    e_test_levels = np.linspace(-e_test_level,e_test_level,21)
    plot.contourf(X_grid,Y_grid,err_p_x,levels=e_test_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_p_x.png',dpi=300)
    plot.close(1)

    err_p_y = p_y_grid - p_y_pred
    plot.figure(1)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,p_y_grid,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,p_y_pred,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    e_test_min = np.nanpercentile(err_p_y.ravel(),0.1)
    e_test_max = np.nanpercentile(err_p_y.ravel(),99.9)
    e_test_level = np.max([abs(e_test_min),abs(e_test_max)])
    e_test_levels = np.linspace(-e_test_level,e_test_level,21)
    plot.contourf(X_grid,Y_grid,err_p_y,levels=e_test_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_p_y.png',dpi=300)
    plot.close(1)

def plot_NS_residual():
    # NS residual
    global X_grid
    global Y_grid
    global model_RANS
    global ScalingParameters
    global i_test
    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)
    mx,my,mass = batch_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,i_test,1000)
    mx = 1.0*np.reshape(mx,X_grid.shape)
    mx[cylinder_mask] = np.NaN
    my = 1.0*np.reshape(my,X_grid.shape)
    my[cylinder_mask] = np.NaN
    mass = 1.0*np.reshape(mass,X_grid.shape)
    mass[cylinder_mask] = np.NaN

    plot.figure(training_steps)
    plot.title('Full Resolution')
    plot.subplot(3,1,1)
    mx_min = np.nanpercentile(mx.ravel(),0.1)
    mx_max = np.nanpercentile(mx.ravel(),99.9)
    mx_level = np.max([abs(mx_min),abs(mx_max)])
    mx_levels = np.linspace(-mx_level,mx_level,21)
    plot.contourf(X_grid,Y_grid,mx,levels=mx_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    my_min = np.nanpercentile(my.ravel(),0.1)
    my_max = np.nanpercentile(my.ravel(),99.9)
    my_level = np.max([abs(my_min),abs(my_max)])
    my_levels = np.linspace(-my_level,my_level,21)
    plot.contourf(X_grid,Y_grid,my,levels=my_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    mass_min = np.nanpercentile(mass.ravel(),0.1)
    mass_max = np.nanpercentile(mass.ravel(),99.9)
    mass_level = np.max([abs(mass_min),abs(mass_max)])
    mass_levels = np.linspace(-mass_level,mass_level,21)
    plot.contourf(X_grid,Y_grid,mass,levels=mass_levels,extend='both')
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_NS_residual.png',dpi=300)


def plot_err():
    global p_grid
    global X_grid
    global Y_grid
    global i_test
    
    global o_test_grid
    global saveFig
    global fig_dir
    global d
    global model_RANS
    global training_steps
    global ScalingParameters

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = 1.0*o_test_grid
    o_test_grid_temp[:,:,5]=1.0*p_grid
    o_test_grid_temp[cylinder_mask,:] = np.NaN

    pred_test = model_RANS(i_test[:],training=False)
    
    pred_test_grid = 1.0*np.reshape(pred_test,[X_grid.shape[0],X_grid.shape[1],6])
    pred_test_grid[cylinder_mask,:] = np.NaN

    plot.close('all')

    i_train_plot = i_train*ScalingParameters.MAX_x

    err_test = o_test_grid_temp-pred_test_grid
    plot_save_exts = ['_ux.png','_uy.png','_uxux.png','_uxuy.png','_uyuy.png','_p.png']
    # quantities
    for i in range(6):
        plot.figure(1)
        plot.title('Full Resolution')
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        if supersample_factor>1:
            plot.scatter(i_train_plot[:,0],i_train_plot[:,1],3,'k','.')
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,pred_test_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        e_test_min = np.nanpercentile(err_test[:,:,i].ravel(),0.1)
        e_test_max = np.nanpercentile(err_test[:,:,i].ravel(),99.9)
        e_test_level = np.max([abs(e_test_min),abs(e_test_max)])
        e_test_levels = np.linspace(-e_test_level,e_test_level,21)
        plot.contourf(X_grid,Y_grid,err_test[:,:,i],levels=e_test_levels,extend='both')
        plot.set_cmap('bwr')
        plot.colorbar()
        if saveFig:
            plot.savefig(fig_dir+'ep'+str(training_steps)+plot_save_exts[i],dpi=300)
        plot.close(1)

def plot_boundary_points():
    global p_grid
    global X_grid
    global Y_grid
    
    global o_test_grid
    global saveFig
    global fig_dir
    global d
    global model_RANS
    global training_steps
    global ScalingParameters
    global boundary_tuple

    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    o_test_grid_temp = np.zeros([X_grid.shape[0],X_grid.shape[1],6])
    o_test_grid_temp[:,:,0:5] = 1.0*o_test_grid
    o_test_grid_temp[:,:,5]=1.0*p_grid
    o_test_grid_temp[cylinder_mask,:] = np.NaN

    (p_BC_vec,cyl_BC_vec,inlet_BC_vec,cylinder_inside_vec,domain_outside_vec) = boundary_tuple 

    p_BC_vec = p_BC_vec*ScalingParameters.MAX_x
    cyl_BC_vec = cyl_BC_vec*ScalingParameters.MAX_x
    inlet_BC_vec = inlet_BC_vec*ScalingParameters.MAX_x
    cylinder_inside_vec = cylinder_inside_vec*ScalingParameters.MAX_x
    domain_outside_vec = domain_outside_vec*ScalingParameters.MAX_x


    global i_test_large
    global X_grid_large
    global Y_grid_large
    cylinder_mask_large = (np.power(X_grid_large,2.0)+np.power(Y_grid_large,2.0))<=np.power(d/2.0,2.0)
    pred_test_large = model_RANS.predict(i_test_large,batch_size=1000)
    pred_test_large_grid = 1.0*np.reshape(pred_test_large,[X_grid_large.shape[0],X_grid_large.shape[1],6])
    pred_test_large_grid[cylinder_mask_large,:] = np.NaN


    plot.figure(1)
    plot.title('Pressure BC')
    plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,5],levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.scatter(p_BC_vec[:,0],p_BC_vec[:,1],color='k',marker='.')
    plot.savefig(fig_dir+'BC_pressure.png',dpi=300)
    plot.close(1)

    plot.figure(1)
    plot.title('No Slip BC')
    plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,0],levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.xlim((-2,2))
    plot.ylim((-2,2))
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.scatter(cyl_BC_vec[:,0],cyl_BC_vec[:,1],color='k',marker='.')
    plot.savefig(fig_dir+'BC_no_slip.png',dpi=300)
    plot.close(1)

    plot.figure(1)
    plot.title('Cylinder Inside')
    plot.contourf(X_grid,Y_grid,o_test_grid_temp[:,:,0],levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.xlim((-2,2))
    plot.ylim((-2,2))
    plot.colorbar()
    plot.scatter(cylinder_inside_vec[:,0],cylinder_inside_vec[:,1],color='k',marker='.')
    plot.savefig(fig_dir+'BC_cylinder_inside.png',dpi=300)
    plot.close(1)

    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,pred_test_large_grid[:,:,0],levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.scatter(inlet_BC_vec[:,0],inlet_BC_vec[:,1],color='k',marker='.')
    plot.savefig(fig_dir+'BC_inlet.png',dpi=300)
    plot.close(1)
    
    plot.figure(1)
    plot.contourf(X_grid_large,Y_grid_large,pred_test_large_grid[:,:,2],levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.scatter(domain_outside_vec[:,0],domain_outside_vec[:,1],color='k',marker='.')
    plot.savefig(fig_dir+'BC_no_stress.png',dpi=300)
    plot.close(1)



def plot_gradients():
    global o_test_grid
    global p_grid
    global X_grid
    global Y_grid


    cylinder_mask = (np.power(X_grid,2.0)+np.power(Y_grid,2.0))<=np.power(d/2.0,2.0)

    data_grads = np.zeros([o_test_grid.shape[0],o_test_grid.shape[1],14])

    labels = ['ux_x','ux_y','uy_x','uy_y','uxux_x','uxuy_x','uxuy_y','uyuy_y','p_x','p_y','ux_xx','ux_yy','uy_xx','uy_yy']

    # first derivatives of data
    data_grads[:,:,0] = np.gradient(o_test_grid[:,:,0],X_grid[:,0],axis=0) # ux_x
    data_grads[:,:,1] = np.gradient(o_test_grid[:,:,0],Y_grid[0,:],axis=1) # ux_y
    data_grads[:,:,2] = np.gradient(o_test_grid[:,:,1],X_grid[:,0],axis=0) # uy_x
    data_grads[:,:,3] = np.gradient(o_test_grid[:,:,1],Y_grid[0,:],axis=1) # uy_y
    data_grads[:,:,4] = np.gradient(o_test_grid[:,:,2],X_grid[:,0],axis=0) # uxux_x
    #data_uxux_y = np.gradient(o_test_grid[:,:,2],Y_grid[0,:],axis=1)
    data_grads[:,:,5] = np.gradient(o_test_grid[:,:,3],X_grid[:,0],axis=0) # uxuy_x
    data_grads[:,:,6] = np.gradient(o_test_grid[:,:,3],Y_grid[0,:],axis=1) # uxuy_y
    #data_uyuy_x = np.gradient(o_test_grid[:,:,4],X_grid[:,0],axis=0)
    data_grads[:,:,7] = np.gradient(o_test_grid[:,:,4],Y_grid[0,:],axis=1) # uyuy_y
    data_grads[:,:,8] = np.gradient(p_grid,X_grid[:,0],axis=0) # p_x
    data_grads[:,:,9] = np.gradient(p_grid,Y_grid[0,:],axis=1) # p_y

    # second derivatives of data
    data_grads[:,:,10] = np.gradient(data_grads[:,:,0],X_grid[:,0],axis=0) # ux_xx
    data_grads[:,:,11] = np.gradient(data_grads[:,:,0],Y_grid[0,:],axis=1) # ux_yy
    data_grads[:,:,12] = np.gradient(data_grads[:,:,2],X_grid[:,0],axis=0) # uy_xx
    data_grads[:,:,13] = np.gradient(data_grads[:,:,2],Y_grid[0,:],axis=1) # uy_yy

    data_grads[cylinder_mask,:]=np.NaN


    NN_grads = RANS_gradients(model_RANS,ScalingParameters,i_test)
    NN_grads_grid = 1.0*np.reshape(NN_grads,data_grads.shape)
    NN_grads_grid[cylinder_mask,:]=np.NaN

    for i in range(data_grads.shape[2]):
        plot.figure(1)
        plot.subplot(3,1,1)
        plot.contourf(X_grid,Y_grid,data_grads[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X_grid,Y_grid,NN_grads_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        plot.subplot(3,1,3)
        plot.contourf(X_grid,Y_grid,data_grads[:,:,i]-NN_grads_grid[:,:,i],levels=21,norm=matplotlib.colors.CenteredNorm())
        plot.set_cmap('bwr')
        plot.colorbar()
        if saveFig:
            plot.savefig(fig_dir+'ep'+str(training_steps)+'_gradients_'+labels[i]+'.png',dpi=300)
        plot.close(1)
    
    # f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy) 
    # f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)
    # f_mass = ux_x + uy_y

    f_x = (o_test_grid[:,0]*data_grads[:,0] + o_test_grid[:,1]*data_grads[:,1]) + (data_grads[:,4] + data_grads[:,5]) + data_grads[:,8] - (ScalingParameters.nu_mol)*(data_grads[:,10]+data_grads[:,11])  
    f_y = (o_test_grid[:,0]*data_grads[:,2] + o_test_grid[:,1]*data_grads[:,3]) + (data_grads[:,6] + data_grads[:,7]) + data_grads[:,9] - (ScalingParameters.nu_mol)*(data_grads[:,12]+data_grads[:,13])
    f_mass = data_grads[:,0] + data_grads[:,3]

    plot.figure(1)
    plot.subplot(3,1,1)
    plot.contourf(X_grid,Y_grid,f_x,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X_grid,Y_grid,f_y,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X_grid,Y_grid,f_mass,levels=21,norm=matplotlib.colors.CenteredNorm())
    plot.set_cmap('bwr')
    plot.colorbar()
    if saveFig:
        plot.savefig(fig_dir+'ep'+str(training_steps)+'_NS_residual_finite_difference.png',dpi=300)
    plot.close(1)
    