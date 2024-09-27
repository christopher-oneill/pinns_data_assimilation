
import platform
import sys
import glob
import numpy as np
import h5py

node_name = platform.node()

LOCAL_NODE = 'DESKTOP-L3FA8HC'
if node_name==LOCAL_NODE:
    import matplotlib
    import matplotlib.pyplot as plot
    useGPU=False    
    SLURM_TMPDIR='F:/projects/pinns_narval/sync/'
    HOMEDIR = 'F:/projects/pinns_narval/sync/'
    PROJECTDIR = HOMEDIR
    sys.path.append('F:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.file_util import create_directory_if_not_exists as crdir


job_names = ['mfg_t001_grads_nolog_101_S4','mfg_t001_grads_log_101_S4']
loss_list = []
data_list = []
physics_list = []
boundary_list = []
max_list = []
wdata_list = []
wphysics_list = []
w_boundary_list = []

data_grads_list = []
physics_grads_list = []
boundary_grads_list = []

for k in range(len(job_names)):
    global job_name 
    job_name = job_names[k]
    base_dir = HOMEDIR+'data/mazi_fixed_grid/'
    global savedir
    savedir = PROJECTDIR+'output/'+job_name+'/'
    paper_fig_dir = 'F:/projects/paper_figures/loss_weighting/'+job_name+'/'
    crdir(paper_fig_dir)

    grads_files = glob.glob(savedir+'*_grads.h5')

    grads_file = h5py.File(grads_files[0],'r')
    print(grads_file.keys())

    loss_history = np.array(grads_file['loss_history'])
    data_loss_history = np.array(grads_file['data_loss_history'])
    physics_loss_history = np.array(grads_file['physics_loss_history'])
    boundary_loss_history = np.array(grads_file['boundary_loss_history'])

    grads_data = np.abs(np.array(grads_file['grads_data']))
    grads_physics = np.abs(np.array(grads_file['grads_physics']))
    grads_boundary = np.abs(np.array(grads_file['grads_boundary']))
    grads_wdata = np.abs(np.array(grads_file['grads_wdata']))
    grads_wphysics = np.abs(np.array(grads_file['grads_wphysics']))
    grads_wboundary = np.abs(np.array(grads_file['grads_wboundary']))

    grads_data_mean_mag = np.mean(grads_data,axis=1)
    grads_physics_mean_mag = np.mean(grads_physics,axis=1)
    grads_boundary_mean_mag = np.mean(grads_boundary,axis=1)
    grads_data_max_mag = np.max(grads_data,axis=1)
    grads_physics_max_mag = np.max(grads_physics,axis=1)
    grads_boundary_max_mag = np.max(grads_boundary,axis=1)
    grads_wdata_mean_mag = np.mean(grads_wdata,axis=1)
    grads_wphysics_mean_mag = np.mean(grads_wphysics,axis=1)
    grads_wboundary_mean_mag = np.mean(grads_wboundary,axis=1)
    grads_wdata_max_mag = np.max(grads_wdata,axis=1)
    grads_wphysics_max_mag = np.max(grads_wphysics,axis=1)
    grads_wboundary_max_mag = np.max(grads_wboundary,axis=1)


    for i in range(1,len(grads_files)):
        grads_file = h5py.File(grads_files[i],'r')

        loss_history = np.concatenate((loss_history,np.array(grads_file['loss_history'])))
        data_loss_history = np.concatenate((data_loss_history,np.array(grads_file['data_loss_history'])))
        physics_loss_history = np.concatenate((physics_loss_history,np.array(grads_file['physics_loss_history'])))
        boundary_loss_history = np.concatenate((boundary_loss_history,np.array(grads_file['boundary_loss_history'])))

        grads_data = np.abs(np.array(grads_file['grads_data']))
        grads_physics = np.abs(np.array(grads_file['grads_physics']))
        grads_boundary = np.abs(np.array(grads_file['grads_boundary']))
        grads_wdata = np.abs(np.array(grads_file['grads_wdata']))
        grads_wphysics = np.abs(np.array(grads_file['grads_wphysics']))
        grads_wboundary = np.abs(np.array(grads_file['grads_wboundary']))
        

        grads_data_mean_mag = np.concatenate((grads_data_mean_mag,np.mean(grads_data,axis=1)))
        grads_physics_mean_mag = np.concatenate((grads_physics_mean_mag,np.mean(grads_physics,axis=1)))
        grads_boundary_mean_mag = np.concatenate((grads_boundary_mean_mag,np.mean(grads_boundary,axis=1)))
        grads_data_max_mag = np.concatenate((grads_data_max_mag,np.max(grads_data,axis=1)))
        grads_physics_max_mag = np.concatenate((grads_physics_max_mag,np.max(grads_physics,axis=1)))
        grads_boundary_max_mag = np.concatenate((grads_boundary_max_mag,np.max(grads_boundary,axis=1)))
        grads_wdata_mean_mag = np.concatenate((grads_wdata_mean_mag,np.mean(grads_wdata,axis=1)))
        grads_wphysics_mean_mag = np.concatenate((grads_wphysics_mean_mag,np.mean(grads_wphysics,axis=1)))
        grads_wboundary_mean_mag = np.concatenate((grads_wboundary_mean_mag,np.mean(grads_wboundary,axis=1)))
        grads_wdata_max_mag = np.concatenate((grads_wdata_max_mag,np.max(grads_wdata,axis=1)))
        grads_wphysics_max_mag = np.concatenate((grads_wphysics_max_mag,np.max(grads_wphysics,axis=1)))
        grads_wboundary_max_mag = np.concatenate((grads_wboundary_max_mag,np.max(grads_wboundary,axis=1)))

    # compute the max and weights

    max_weight = np.exp(np.ceil(np.log(1E-30+np.max(np.stack((data_loss_history,physics_loss_history,boundary_loss_history),axis=1),axis=1))))

    w_data = max_weight/np.abs(1E-30+data_loss_history)
    w_physics = max_weight/np.abs(1E-30+physics_loss_history)
    w_boundary = max_weight/np.abs(1E-30+boundary_loss_history)



    loss_list.append(loss_history)
    data_list.append(data_loss_history)
    physics_list.append(physics_loss_history)
    boundary_list.append(boundary_loss_history)
    max_list.append(max_weight)
    wdata_list.append(w_data)
    wphysics_list.append(w_physics)
    w_boundary_list.append(w_boundary_list)
    data_grads_list.append(grads_data)
    
    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(loss_history)
    plot.plot(data_loss_history)
    plot.plot(physics_loss_history)
    plot.plot(boundary_loss_history)
    plot.legend(['loss','data','physics','boundary'])
    plot.xlim([0,800])
    plot.ylabel('Loss')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.savefig(paper_fig_dir+'loss_nolog.pdf',)
    plot.close(fig)

    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(loss_history)
    plot.plot(data_loss_history*(1+w_data))
    plot.plot(physics_loss_history*(1+w_physics))
    plot.plot(boundary_loss_history*(1+w_boundary))
    plot.legend(['loss','weighted data','weighted physics','weighted boundary'])
    plot.xlim([0,800])
    plot.ylabel('Loss')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.savefig(paper_fig_dir+'loss_weighted.pdf',)
    plot.close(fig)

    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(loss_history)
    plot.plot((1+w_data))
    plot.plot((1+w_physics))
    plot.plot((1+w_boundary))
    plot.legend(['loss','wdata','wphysics','wboundary'])
    plot.xlim([0,800])
    plot.ylabel('Loss')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.savefig(paper_fig_dir+'loss_weights.pdf',)
    plot.close(fig)



    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(grads_data_mean_mag,'-r')
    plot.plot(grads_physics_mean_mag,'-b')
    plot.plot(grads_boundary_mean_mag,'-g')
    plot.plot(grads_data_max_mag,'--r')
    plot.plot(grads_physics_max_mag,'--b')
    plot.plot(grads_boundary_max_mag,'--g')
    plot.legend(['data mean','physics_mean','boundary mean','data max','physics max','boundary max'])
    plot.ylabel('Gradient Magnitude')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.xlim([0,800])
    plot.savefig(paper_fig_dir+'grads_mag_nolog.pdf',)
    plot.close(fig)

    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(grads_data_mean_mag,'-r')
    plot.plot(grads_wdata_mean_mag,'-b')

    plot.plot(grads_data_max_mag,'--r')
    plot.plot(grads_wdata_max_mag,'--b')

    plot.plot(grads_data_mean_mag/grads_wdata_mean_mag,'-k')

    plot.legend(['data mean','wdata mean','data max','wdata max','data mean / wdata mean'])
    plot.ylabel('Gradient Magnitude')
    plot.xlabel('Epochs')
    plot.xlim([0,800])
    plot.yscale('log')
    plot.savefig(paper_fig_dir+'grads_mag_wdata_nolog.pdf',)
    plot.close(fig)

    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(grads_physics_mean_mag,'-r')
    plot.plot(grads_wphysics_mean_mag,'-b')

    plot.plot(grads_physics_max_mag,'--r')
    plot.plot(grads_wphysics_max_mag,'--b')

    plot.plot(grads_physics_mean_mag/grads_wphysics_mean_mag,'-k')

    plot.legend(['physics mean','wphysics mean','physics max','wphysics max','physics mean/wphysics mean'])
    plot.ylabel('Gradient Magnitude')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.xlim([0,800])
    plot.savefig(paper_fig_dir+'grads_mag_wphysics_nolog.pdf',)
    plot.close(fig)

    fig = plot.figure(1)
    fig.set_size_inches(8.5,4)
    plot.plot(grads_boundary_mean_mag,'-r')
    plot.plot(grads_wboundary_mean_mag,'-b')

    plot.plot(grads_boundary_max_mag,'--r')
    plot.plot(grads_wboundary_max_mag,'--b')

    plot.plot(grads_boundary_mean_mag/grads_wboundary_mean_mag,'-k')

    plot.legend(['boundary mean','wboundary mean','boundary max','wboundary max','boundary mean/wboundary mean'])
    plot.ylabel('Gradient Magnitude')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.xlim([0,800])
    plot.savefig(paper_fig_dir+'grads_mag_wboundary_nolog.pdf',)
    plot.close(fig)

if False:
    summary_paper_fig_dir = paper_fig_dir = 'F:/projects/paper_figures/loss_weighting/'

    fig = plot.figure(1)
    fig.set_size_inches(8.5,8)
    plot.subplot(2,1,1)
    plot.plot(1,'-r')
    plot.plot(grads_wboundary_mean_mag,'-b')

    plot.plot(grads_boundary_max_mag,'--r')
    plot.plot(grads_wboundary_max_mag,'--b')

    plot.legend(['boundary mean','wboundary mean','boundary max','wboundary max'])
    plot.ylabel('Gradient Magnitude')
    plot.xlabel('Epochs')
    plot.yscale('log')
    plot.xlim([0,800])
    plot.savefig(summary_paper_fig_dir+'grads_mag_wboundary_nolog.pdf',)
    plot.close(fig)