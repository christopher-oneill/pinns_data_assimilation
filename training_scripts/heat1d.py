


import numpy as np
import matplotlib.pyplot as plot


Lx = 200
Lt = 1
x = np.reshape(np.linspace(0,2,Lx),(Lx,))
dx = x[2]-x[1]

# compute the step size so r =0.2
dt = 0.2*(dx*dx)
print('dt: ',dt)
print('r: ',dt/(dx*dx))

nT = np.int64(Lt/dt)
t = np.reshape(dt*np.arange(nT),(nT))

h = np.zeros((Lx,2))
h[:,0]=np.exp(-np.power(3*x,2.0))

dt = t[2]-t[1]



print('dx: ',dx)
fig = plot.figure(1)
ims = []


for i in range(nT):
    #h_padded = np.concatenate((np.array([1.0,]),h[:,0],np.array([0.0,])))
    h_padded = np.concatenate((np.array([-h[1,0],]),h[:,0],np.array([0.0,])))
    #print('h_padded.shape: ',h_padded.shape)
    dh2dx2 =(h_padded[2:h_padded.shape[0]] -2*h_padded[1:h_padded.shape[0]-1] + h_padded[0:h_padded.shape[0]-2])/(dx*dx)
    #dh2dx2[0] = 0.0
    #print('dh2dx2.shape: ',dh2dx2.shape)

    h[:,1] = h[:,0] + dt*(dh2dx2)
    h[0,1] = 1.0
    h[100,1] = 0.0
    h[101,1] = 0.0
    h[Lx-1,1] = 1.0

    if np.mod(i,100)==0:
        im = plot.plot(x,h[:,1],'-k',animated=True,)
        ims.append(im)

    h[:,0] = h[:,1]

import matplotlib.animation as animation


    

ani = animation.ArtistAnimation(fig, ims, interval=33, blit=True,repeat=False)
ani.save('F:/projects/paper_figures/heat_equation/'+'animation.mp4')
#plot.show()