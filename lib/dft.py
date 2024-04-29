
import numpy as np

def dft(x,f=None,fs=None):
    # the order of x should be either (n_t), (n_x,n_t) or (n_x,n_t,n_c)
    if x.ndim ==1:
        nt = x.shape[0]
    else:
        nt = x.shape[1]

    if f == None:
        # use standard fft spacing
        nf = nt # number of time points
        f = np.concatenate((np.linspace(0,0.5,np.int64(np.ceil(nf/2.0)),endpoint=True),np.linspace(-0.5,0,np.int64(nf-np.ceil(nf/2.0)),endpoint=False))) ## normalized frequency (1.0 = Fs)
    else: 
        # use the frequency vector provided
        nf = f.size
        # if f is provided and fs is not provided, we assume that it is normalzed (thus do nothing here)
        # if f is provided and fs is provided, assume that f is in the same units as fs (ie not normalized)
        if fs is not None:
            f = f/fs # normalize

    f = np.reshape(f,[1,f.size])     
    t = -2j*np.pi*np.reshape(np.linspace(0,nt,nt,endpoint=False),[nt,1])

    theta = np.exp(np.matmul(t,f)) # dft matrix

    # compute DFT of the vector
    if x.ndim ==1:
        X = np.reshape(np.matmul(np.reshape(x,[1,x.shape[0]]),theta),[nf,]) # DFT matrix = phase of eatch signal at each time
        X = np.reshape(X,[nt,]) # reduce the dimensionality back to 1
    elif x.ndim==2:
        X = np.matmul(x,theta)
    elif x.ndim==3:
        X = np.zeros(x.shape,np.complex128)
        for c in range(x.shape[2]):
            X[:,:,c] = np.matmul(x[:,:,c],theta)

    # return the correct frequency vector
    if fs is not None:
        f = f*fs # denormalize f
    
    return X, np.reshape(f,[nf,])

           
def idft(X,f=None,t=None,fs=None):
    # X should be (nf,) or (x,nf), or (x,nf,nc); f should be (nf,) t should be (nt,)

    # there are 3 cases:
    # Case 1: only X is provided
    # assume all frequencies are provided and nf=nt

    # Case 2: X, f, t are provided, fs not provided 
    # X and f should have matched lengths nf, f is normalized,
    # if t is a vector, evaluate at t normalized; if t is a scalar then nt=t]

    # Case 3: X, f, t, fs are provided 
    # X and f should have matched lengths nf, f has same units as fs, 
    # if t is a vector, evaluate at t not normalized; if t is a scalar then nt=t

    if t is None:
        # case 1
        if X.ndim==1:
            nt = X.shape[0]
            nf = X.shape[0]
        else:
            nt = X.shape[1]
            nf = X.shape[1]

        f = np.concatenate((np.linspace(0,0.5,np.int64(np.ceil(nf/2.0)),endpoint=True),np.linspace(-0.5,0,np.int64(nf-np.ceil(nf/2.0)),endpoint=False)))
        t = 2j*np.pi*nt*np.reshape(np.linspace(0,1,nt,endpoint=False),[1,nt])
    elif fs==None:
        # case 3
        if t.size==1:
            # scalar case
            nt = t
            t = 2j*np.pi*nt*np.reshape(np.linspace(0,1,nt,endpoint=False),[nt,1])
        else:
            # normalized vector t with start =0, L=1, but any t is valid
            nt = t.size
            t = 2j*np.pi*nt*np.reshape(t,[t.size,1])      
        # we assume f is correctly normalized on [-0.5,0.5] ie nyquist
    else:
        # case 2
        if t.size==1:
            # scalar t case
            nt = t
            t =  2j*np.pi*nt*np.reshape(np.linspace(0,1,nt,endpoint=False),[nt,1])
        else:
            # vector t with start t=0, L=nt/fs, but any t is valid
            nt = t.size
            t = 2j*np.pi*np.reshape(t,[1,t.size])*fs
            
        f = f/fs # normalize f

    # check for agreement of X,f,t
    if X.ndim==1:
        if X.shape[0] != f.size:
            raise ValueError('X and f should have the same number of frequencies')
    else:
        if X.shape[1] != f.size:
            raise ValueError('X and f should have the same number of frequencies')
            
    #if t.size != f.size:
    #    raise ValueError('vector f and t should have the same size or scalar t should equal f.size')

    f = np.reshape(f,[f.size,1])
    t = np.reshape(t,[1,t.size])

    Theta = np.exp(np.matmul(f,t)) # evaluate idft matrix

    # compute the idft
    if X.ndim==1:
        # (1 x nf) x (nf x nt)  = 1 x nt
        x = np.matmul(np.reshape(X,[1,X.shape[0]]),Theta)
        x = np.real(np.reshape(x,[x.shape[1],]))/X.shape[0]
    elif X.ndim==2:
        # (nx x nf) x (nf x nt) = nx x nt
        x = np.real(np.matmul(X,Theta))/X.shape[1]
    elif X.ndim==3:
        # stacked 2d case
        x = np.zeros(X.shape,dtype=np.float64)
        for c in range(X.shape[2]):
            x[:,:,c] = np.real(np.matmul(X[:,:,c],Theta))/X.shape[1]
    else:
        raise(ValueError('X should have ndim<3'))
    
    if fs is not None:
        # if needed we de normalize t on output
        t = t/(2j*np.pi*fs)
    else:
        # denormalize t for output
        t = t/(2j*np.pi*nt)

    return x, np.reshape(np.real(t),[t.size,])

