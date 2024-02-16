
import numpy as np
import sys
import matplotlib.pyplot as plot
sys.path.append('C:/projects/pinns_local/code/')

from pinns_data_assimilation.lib.dft import dft, dft2

# (nt,)
nt = 4100
fs = 10
t = np.reshape(np.linspace(0,(nt-1)/fs,nt),[nt])

sig = np.sin(2*np.pi*t*0.183)

plot.figure(1)
plot.plot(t,sig)
Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[0]/2.0))

plot.figure(2)
plot.plot(fs*f[0:half_idx],np.abs(Sig[0:half_idx]))
plot.yscale('log')




# (nx,nt)
t = np.reshape(np.linspace(0,(nt-1)/fs,nt),[1,nt])

nsig = 10
sig = np.zeros([nsig,t.shape[1]])
for i in range(1,nsig+1):
    sig[i-1,:] = np.sin(i*2*np.pi*t*0.183)

plot.figure(11)
for i in range(nsig):
    plot.plot(t.transpose(),sig[i,:])


Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[1]/2.0))
plot.figure(12)
for i in range(nsig):
    plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx])/nt)
plot.yscale('log')


# (nx,nt,nc)


nsig = 3
sig = np.zeros([nsig,t.shape[1],3])
for i in range(1,nsig+1):
    sig[i-1,:,0] = np.sin(i*2*np.pi*t*0.2)
    sig[i-1,:,1] = np.sin(i*2*np.pi*t*0.1)
    sig[i-1,:,2] = np.sin(i*2*np.pi*t*0.25)

plot.figure(21)
for i in range(nsig):
    plot.plot(t.transpose(),sig[i,:,0],'-r')
    plot.plot(t.transpose(),sig[i,:,1],'-g')
    plot.plot(t.transpose(),sig[i,:,2],'-b')


Sig,f = dft(sig)

half_idx = np.int64(np.ceil(t.shape[1]/2.0))
plot.figure(22)
for i in range(nsig):
    plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx,0])/nt,'-r')
    plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx,1])/nt,'-g')
    plot.plot(fs*f[0:half_idx],np.abs(Sig[i,0:half_idx,2])/nt,'-b')
plot.yscale('log')
plot.show()