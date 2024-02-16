
import numpy as np
import sys
import matplotlib.pyplot as plot
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.dft import dft
from pinns_data_assimilation.lib.dft import idft

# (nt,)
nt = 4100
fs = 10
t = np.reshape(np.linspace(0,(nt-1)/fs,nt),[nt])

sig = np.sin(2*np.pi*t*0.183)

plot.figure(1)
plot.plot(t,sig)
Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[0]/2.0))

# test the coefficient only case
plot.figure(2)
plot.plot(fs*f[0:half_idx],np.abs(Sig[0:half_idx]))
plot.yscale('log')


iSig,iT = idft(Sig)

plot.figure(3)
plot.plot(t,sig)
plot.plot(iT*nt/fs,iSig) # manually convert from normalized time to regular time
plot.plot(t,np.log10(np.abs(sig-iSig)))


# now test with normalized f, normalized t provided

iSig,iT = idft(Sig,f,t*fs/nt)

plot.figure(4)
plot.plot(t,sig)
plot.plot(iT*nt/fs,iSig)
plot.plot(t,np.log10(np.abs(sig-iSig)))

# now test with nonormalized f,t with fs
iSig,iT = idft(Sig,f*fs,t,fs)


plot.figure(5)
plot.plot(t,sig)
plot.plot(iT,iSig)
plot.plot(t,np.log10(np.abs(sig-iSig)))


# (nx,nt)

sig = np.zeros([3,nt])
sig[0,:] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.05)
sig[1,:] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.183)
sig[2,:] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.4)

plot.figure(11)
for i in range(sig.shape[0]):
    plot.plot(t,sig[i,:])
Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[0]/2.0))

# test the coefficient only case
plot.figure(12)
for i in range(Sig.shape[0]):
    plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx]))
plot.yscale('log')


iSig,iT = idft(Sig)

plot.figure(13)
for i in range(Sig.shape[0]):
    plot.subplot(sig.shape[0],1,i+1)
    plot.plot(t,sig[i,:])
    plot.plot(iT*nt/fs,iSig[i,:]) # manually convert from normalized time to regular time
    plot.plot(t,np.log10(np.abs(sig[i,:]-iSig[i,:])))


# now test with normalized f, normalized t provided

iSig,iT = idft(Sig,f,t*fs/nt)

plot.figure(14)
for i in range(Sig.shape[0]):
    plot.subplot(sig.shape[0],1,i+1)
    plot.plot(t,sig[i,:])
    plot.plot(iT*nt/fs,iSig[i,:])
    plot.plot(t,np.log10(np.abs(sig[i,:]-iSig[i,:])))

# now test with nonormalized f,t with fs
iSig,iT = idft(Sig,f*fs,t,fs)


plot.figure(15)
for i in range(Sig.shape[0]):
    plot.subplot(sig.shape[0],1,i+1)
    plot.plot(t,sig[i,:])
    plot.plot(iT,iSig[i,:])
    plot.plot(t,np.log10(np.abs(sig[i,:]-iSig[i,:])))


# (nx,nt,nc)

sig = np.zeros([2,nt,2])
sig[0,:,0] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.05)
sig[1,:,0] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.183)
sig[0,:,1] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.4)
sig[1,:,1] = np.sin(2*np.pi*np.reshape(t,[1,nt])*0.12)

plot.figure(21)
for j in range(sig.shape[2]):
    for i in range(sig.shape[0]):
        plot.plot(t,sig[i,:,j])
Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[0]/2.0))

# test the coefficient only case
plot.figure(22)
for j in range(Sig.shape[2]):
    for i in range(Sig.shape[0]):
        plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx,j]))
plot.yscale('log')


iSig,iT = idft(Sig)

for j in range(Sig.shape[2]):
    plot.figure(23+j)
    for i in range(Sig.shape[0]):
        plot.subplot(sig.shape[0],1,i+1)
        plot.plot(t,sig[i,:,j])
        plot.plot(iT*nt/fs,iSig[i,:,j]) # manually convert from normalized time to regular time
        plot.plot(t,np.log10(np.abs(sig[i,:,j]-iSig[i,:,j])))


# now test with normalized f, normalized t provided

iSig,iT = idft(Sig,f,t*fs/nt)

for j in range(Sig.shape[2]):
    plot.figure(25+j)
    for i in range(Sig.shape[0]):
        plot.subplot(sig.shape[0],1,i+1)
        plot.plot(t,sig[i,:,j])
        plot.plot(iT*nt/fs,iSig[i,:,j])
        plot.plot(t,np.log10(np.abs(sig[i,:,j]-iSig[i,:,j])))

# now test with nonormalized f,t with fs
iSig,iT = idft(Sig,f*fs,t,fs)

for j in range(Sig.shape[2]):
    plot.figure(27+j)
    for i in range(Sig.shape[0]):
        plot.subplot(sig.shape[0],1,i+1)
        plot.plot(t,sig[i,:,j])
        plot.plot(iT,iSig[i,:,j])
        plot.plot(t,np.log10(np.abs(sig[i,:,j]-iSig[i,:,j])))




plot.show()

