
import numpy as np
import matplotlib.pyplot as plot
from scipy.signal import convolve2d

def POD(fluctuating_velocity):
    # we assume that the dimension order is (x,y,c) or (x,t)
    nx = fluctuating_velocity.shape[0]
    nt = fluctuating_velocity.shape[1]
    # assemble the snapshot matrix A
    if fluctuating_velocity.ndim ==3:
        # case of more than one component
        nc = fluctuating_velocity.shape[2]
        A = np.zeros([nx*nc,nt])
        for c in range(nc):
            A[c*nx:(c+1)*nx,:] = fluctuating_velocity[:,:,c]
    else:
        # case with one component, or where the component dimension was already concatenated along x
        nc=1
        A = 1.0*fluctuating_velocity
    
    # compute the correlation matrix C
    C = (1/nt)*np.matmul(A.transpose(),A)

    m_eps = np.finfo(np.float64).eps

    # singular value decomposition
    Ei,Lambda,Vh = np.linalg.svd(C,full_matrices=True)

    sort_inds = np.argsort(Lambda)[::-1]
    Lambda = Lambda[sort_inds]
    Lambda[Lambda<m_eps] = m_eps # set any negative eigenvalues to the machine epsilon

    # CHECK: TKE=  0.5*sum(A.^2,[1,2])./n_t = 0.5*trace(C) = 0.5*sum(Lambda,1) 
    assert(np.abs(0.5*np.sum(np.power(A,2.0),axis=(0,1))/nt - 0.5*np.trace(C))<1E-9)
    assert(np.abs(0.5*np.trace(C) - 0.5*np.sum(Lambda,axis=0))<1E-9)

    Ei = Ei[:,sort_inds]

    Phi = np.matmul(A,(Ei/(np.sqrt(nt*Lambda.transpose()))))

    # check that the leading modes are orthogonal
    modes_check = 50
    mode_dot = np.zeros([modes_check,modes_check])
    for i in range(modes_check):
        for j in range(modes_check):
            if i>=j:
                mode_dot[i,j] = np.dot(Phi[:,i],Phi[:,j])
                mode_dot[j,i] = mode_dot[i,j]
    
    #assert(np.sum(np.abs(mode_dot-np.eye(modes_check))>1E-5,axis=(0,1))==0) # (ui,uj)_omega = delta_ij


    Ak = np.multiply(Ei,np.sqrt(nt*Lambda))

    #for i in [0,1,2,4]:
    #    plot.figure(i+10)
    #    plot.plot(np.arange(Lambda.shape[0]),Ak[:,i])
    #    plot.xlim([0,200])
    #plot.show()

    #print(np.max(np.abs(np.mean(Ak,axis=0))))
   
    #assert(np.all(np.abs(np.mean(Ak,axis=0))<1E-6))

    # check that the leading temporal coefficients are orthogonal
    modes_check = 50
    mode_dot = np.zeros([modes_check,modes_check])
    for i in range(modes_check):
        for j in range(modes_check):
            if i>=j:
                mode_dot[i,j] = np.dot(Ak[:,i],Ak[:,j])/nt

    #print('Phi.shape',Phi.shape)
    #print('A.shape',A.shape)
    #print('A*Phi.shape',np.multiply(A,np.reshape(Phi[:,0],[nx*nc,1])).shape)
    ld_check = np.multiply(np.eye(modes_check),Lambda[0:modes_check])+1E-30
    assert(np.all(np.abs(mode_dot-ld_check)<1E-6)) # <ai aj> = Lambda_i delta_ij
    assert(np.all(np.abs(np.sum(np.multiply(A,np.reshape(Phi[:,0],[nx*nc,1])),axis=0).transpose()-Ak[:,0])<1E-6)) # ai = (um-u0,ui)_omega

    return Phi, Lambda, Ak


def extended_POD(fluctuating_velocity,fluctuating_pressure):
    pass

def fir_SPOD(fluctuatingVelocity,Nf,boundary=0):
    # boundary=0 -> zero pad
    # boundary=1 -> periodic  

    # we assume that the dimension order is (x,y,c) or (x,t)
    nx = fluctuatingVelocity.shape[0]
    nt = fluctuatingVelocity.shape[1]
    # assemble the snapshot matrix A
    if fluctuatingVelocity.ndim ==3:
        # case of more than one component
        nc = fluctuatingVelocity.shape[2]
        A = np.zeros([nx*nc,nt])
        for c in range(nc):
            A[c*nx:(c+1)*nx,:] = fluctuatingVelocity[:,:,c]
    else:
        # case with one component, or where the component dimension was already concatenated along x
        nc=1
        A = 1.0*fluctuatingVelocity
    
    # set the 1d filter
    if Nf==nt:
        # box filter, for equivalence with DFT
        f1d = np.ones((Nf))/Nf
        boundary=1 # we need to use 
    else:
        # gaussian filter otherwise
        f1d = np.exp(np.power(-np.linspace(-2.285,2.285,2*Nf+1),2.0))
  
    # compute the correlation matrix C
    C = (1/nt)*np.matmul(A.transpose(),A)

    f2d = np.diag(f1d)
    print(f2d.shape)

    # compute the filtered correlation matrix
    if boundary ==0:
        S=convolve2d(C,f2d,mode='same',boundary='fill',fillvalue=0.0)
    elif boundary ==1:
        S=convolve2d(C,f2d,mode='same',boundary='wrap')

    m_eps = np.finfo(np.float64).eps

    # singular value decomposition
    Ei,Lambda,Vh = np.linalg.svd(S,full_matrices=True)

    sort_inds = np.argsort(Lambda)[::-1]
    Lambda = Lambda[sort_inds]
    Lambda[Lambda<m_eps] = m_eps # set any negative eigenvalues to the machine epsilon

     # CHECK: TKE=  0.5*sum(A.^2,[1,2])./n_t = 0.5*trace(C) = 0.5*sum(Lambda,1) 
    #assert(np.abs(0.5*np.sum(np.power(A,2.0),axis=(0,1))/nt - 0.5*np.trace(C))<1E-9)
    #assert(np.abs(0.5*np.trace(C) - 0.5*np.sum(Lambda,axis=0))<1E-9)

    Ei = Ei[:,sort_inds]

    Phi = np.matmul(A,(Ei/(np.sqrt(nt*Lambda.transpose()))))

    # check that the leading modes are orthogonal
    modes_check = 50
    mode_dot = np.zeros([modes_check,modes_check])
    for i in range(modes_check):
        for j in range(modes_check):
            if i>=j:
                mode_dot[i,j] = np.dot(Phi[:,i],Phi[:,j])
                mode_dot[j,i] = mode_dot[i,j]

    #assert(np.sum(np.abs(mode_dot-np.eye(modes_check))>1E-5,axis=(0,1))==0) # (ui,uj)_omega = delta_ij


    Ak = np.multiply(Ei,np.sqrt(nt*Lambda))

    #for i in [0,1,2,4]:
    #    plot.figure(i+10)
    #    plot.plot(np.arange(Lambda.shape[0]),Ak[:,i])
    #    plot.xlim([0,200])
    #plot.show()

    print(np.max(np.abs(np.mean(Ak,axis=0))))
   
    #assert(np.all(np.abs(np.mean(Ak,axis=0))<1E-6))

    # check that the leading temporal coefficients are orthogonal
    modes_check = 50
    mode_dot = np.zeros([modes_check,modes_check])
    for i in range(modes_check):
        for j in range(modes_check):
            if i>=j:
                mode_dot[i,j] = np.dot(Ak[:,i],Ak[:,j])/nt

    #print('Phi.shape',Phi.shape)
    #print('A.shape',A.shape)
    #print('A*Phi.shape',np.multiply(A,np.reshape(Phi[:,0],[nx*nc,1])).shape)
    ld_check = np.multiply(np.eye(modes_check),Lambda[0:modes_check])+1E-30
    #assert(np.all(np.abs(mode_dot-ld_check)<1E-6)) # <ai aj> = Lambda_i delta_ij
    #assert(np.all(np.abs(np.sum(np.multiply(A,np.reshape(Phi[:,0],[nx*nc,1])),axis=0).transpose()-Ak[:,0])<1E-6)) # ai = (um-u0,ui)_omega

    return Phi, Lambda, Ak