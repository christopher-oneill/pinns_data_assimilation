
import numpy as np
import matplotlib.pyplot as plot
import sys

sys.path.append('C:/projects/pinns_local/code/')

import pinns_galerkin_viv.lib.galerkin as galerkin

def galerkin_projection(nu,ux,ux_x,ux_y,ux_xx,ux_yy,uy,uy_x,uy_y,uy_xx,uy_yy,p,p_x,p_y,phi_x,phi_x_x,phi_x_y,phi_x_xx,phi_x_yy,phi_y,phi_y_x,phi_y_y,phi_y_xx,phi_y_yy,psi,psi_x,psi_y):
    nx = phi_x.shape[0]
    nk = phi_x.shape[1]

    Fk =  np.zeros([nx,nk],dtype=np.double)
    Lkl = np.zeros([nx,nk,nk],dtype=np.double)
    Qklm = np.zeros([nx,nk,nk,nk],np.double)

    for k in range(nk):
        Fk[:,k] = -phi_x[:,k]*(2*ux*ux_x+ux*uy_y+uy*ux_y)- phi_y[:,k]*(ux*uy_x+uy*ux_x+2*uy*uy_y) - 0*phi_x[:,k]*p_x - 0*phi_y[:,k]*p_y + nu*(phi_x[:,k]*(ux_xx+ux_yy)+phi_y[:,k]*(uy_xx+uy_yy))

        for l in range(nk):
            Lkl[:,k,l] = (-phi_x[:,k]*(phi_x[:,l]*ux_x+phi_y[:,l]*ux_y) - phi_y[:,k]*(phi_x[:,l]*uy_x+phi_y[:,l]*uy_y)
                          -phi_x[:,k]*(ux*phi_x_x[:,l]+uy*phi_x_y[:,l]) - phi_y[:,k]*(ux*phi_y_x[:,l]+uy*phi_y_y[:,l])
                          -0*phi_x[:,k]*psi_x[:,l]-0*phi_y[:,k]*psi_y[:,l]
                          +nu*(phi_x[:,k]*(phi_x_xx[:,l]+phi_x_yy[:,l])+phi_y[:,k]*(phi_y_xx[:,l]+phi_y_yy[:,l])))
            for m in range(nk):
                Qklm[:,k,l,m] = -phi_x[:,k]*(phi_x[:,l]*phi_x_x[:,m] + phi_y[:,l]*phi_x_y[:,m]) -phi_y[:,k]*(phi_x[:,l]*phi_y_x[:,m] + phi_y[:,l]*phi_y_y[:,m])

    return Fk, Lkl, Qklm

np.random.seed(20)
cycles = 400
L = np.int64(cycles/3.0)
x = np.linspace(-1,1,400)
y = np.linspace(-1,1,200)
t = np.linspace(0,L,100*L)

X,Y = np.meshgrid(x,y)
X = X.transpose()
Y = Y.transpose()

nu = 0.01

n_modes = 4

K_rand = 5*np.pi*np.random.rand(n_modes+1,4)-2.5*np.pi
A_rand = 2*np.random.rand(n_modes+1,2)-1
print(K_rand)
omega_rand = 2*np.pi*np.random.rand(n_modes,1)+2*np.pi
print('Omegas [Hz]:')
print(omega_rand/(2*np.pi))

u = np.zeros([x.shape[0],y.shape[0],t.shape[0]])
v = np.zeros([x.shape[0],y.shape[0],t.shape[0]])

phi_x_A = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_y_A = np.zeros([x.shape[0],y.shape[0],n_modes+1])
Ak = np.zeros([t.shape[0],n_modes+1])

# derivatives
phi_x_A_x = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_x_A_xx = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_x_A_y = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_x_A_yy = np.zeros([x.shape[0],y.shape[0],n_modes+1])

phi_y_A_x = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_y_A_xx = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_y_A_y = np.zeros([x.shape[0],y.shape[0],n_modes+1])
phi_y_A_yy = np.zeros([x.shape[0],y.shape[0],n_modes+1])

# set the mean field
phi_x_A[:,:,0] = A_rand[0,0]*np.sin(K_rand[0,0]*X)*np.sin(K_rand[0,1]*Y)
phi_x_A_x[:,:,0] = A_rand[0,0]*K_rand[0,0]*np.cos(K_rand[0,0]*X)*np.sin(K_rand[0,1]*Y)
phi_x_A_xx[:,:,0] = -A_rand[0,0]*K_rand[0,0]*K_rand[0,0]*np.sin(K_rand[0,0]*X)*np.sin(K_rand[0,1]*Y)

phi_y_A[:,:,0] = A_rand[0,1]*np.sin(K_rand[0,2]*X)*np.sin(K_rand[0,3]*Y)
phi_y_A_y[:,:,0] = A_rand[0,1]*K_rand[0,3]*np.sin(K_rand[0,2]*X)*np.cos(K_rand[0,3]*Y)
phi_y_A_yy[:,:,0] = -A_rand[0,1]*K_rand[0,3]*K_rand[0,3]*np.sin(K_rand[0,2]*X)*np.sin(K_rand[0,3]*Y)
Ak[:,0] = 1

# compute the modes
for nm in range(1,n_modes+1):
    phi_x_A[:,:,nm] = A_rand[nm,0]*np.sin(K_rand[nm,0]*X)*np.sin(K_rand[nm,1]*Y)
    phi_x_A_x[:,:,nm] = A_rand[nm,0]*K_rand[nm,0]*np.cos(K_rand[nm,0]*X)*np.sin(K_rand[nm,1]*Y)
    phi_x_A_xx[:,:,nm] = -A_rand[nm,0]*K_rand[nm,0]*K_rand[nm,0]*np.sin(K_rand[nm,0]*X)*np.sin(K_rand[nm,1]*Y)
    phi_x_A_y[:,:,nm] = A_rand[nm,0]*K_rand[nm,1]*np.sin(K_rand[nm,0]*X)*np.cos(K_rand[nm,1]*Y)
    phi_x_A_yy[:,:,nm] = -A_rand[nm,0]*K_rand[nm,1]*K_rand[nm,1]*np.sin(K_rand[nm,0]*X)*np.sin(K_rand[nm,1]*Y)
    
    phi_y_A[:,:,nm] = A_rand[nm,1]*np.sin(K_rand[nm,2]*X)*np.sin(K_rand[nm,3]*Y)
    phi_y_A_x[:,:,nm] = A_rand[nm,1]*K_rand[nm,2]*np.cos(K_rand[nm,2]*X)*np.sin(K_rand[nm,3]*Y)
    phi_y_A_xx[:,:,nm] = -A_rand[nm,1]*K_rand[nm,2]*K_rand[nm,2]*np.sin(K_rand[nm,2]*X)*np.sin(K_rand[nm,3]*Y)
    phi_y_A_y[:,:,nm] = A_rand[nm,1]*K_rand[nm,3]*np.sin(K_rand[nm,2]*X)*np.cos(K_rand[nm,3]*Y)
    phi_y_A_yy[:,:,nm] = -A_rand[nm,1]*K_rand[nm,3]*K_rand[nm,3]*np.sin(K_rand[nm,2]*X)*np.sin(K_rand[nm,3]*Y)

    Ak[:,nm] = np.sin(omega_rand[nm-1]*t)


F_k, L_kl, Q_klm = galerkin.galerkin_noack(0,phi_x_A,phi_x_A_x,phi_x_A_y,phi_x_A_xx,phi_x_A_yy,phi_y_A,phi_y_A_x,phi_y_A_y,phi_y_A_xx,phi_y_A_yy)

Fk, Lkl, Qklm = galerkin.galerkin_projection(0,phi_x_A[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_x_A_x[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_x_A_y[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_x_A_xx[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_x_A_yy[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A_x[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A_y[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A_xx[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A_yy[:,:,0].reshape([x.shape[0]*y.shape[0]]),phi_y_A[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_x[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_y[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_xx[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_yy[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_x[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_y[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_xx[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]),phi_y_A_yy[:,:,1:(n_modes+1)].reshape([x.shape[0]*y.shape[0],n_modes]))

Fk = Fk.reshape([x.shape[0],y.shape[0],n_modes])
Lkl = Lkl.reshape([x.shape[0],y.shape[0],n_modes,n_modes])
Qklm = Qklm.reshape([x.shape[0],y.shape[0],n_modes,n_modes,n_modes])


# accumulate the fluctuating field
for nm in range(n_modes+1):
    u = u + np.reshape(phi_x_A[:,:,nm],[x.shape[0],y.shape[0],1])*np.reshape(Ak[:,nm],[1,t.shape[0]])
    v = v +  np.reshape(phi_y_A[:,:,nm],[x.shape[0],y.shape[0],1])*np.reshape(Ak[:,nm],[1,t.shape[0]])

u_mean_N = np.mean(u,axis=2)
v_mean_N = np.mean(v,axis=2)


#plot.figure(1)
#plot.plot(t,Ak)

for k in range(n_modes):
    plot.figure(100+k)
    plot.subplot(3,1,1)
    plot.contourf(X,Y,F_k[:,:,k],21)
    plot.colorbar()
    plot.subplot(3,1,2)
    plot.contourf(X,Y,Fk[:,:,k],21)
    plot.colorbar()
    plot.subplot(3,1,3)
    plot.contourf(X,Y,F_k[:,:,k]-Fk[:,:,k],21)
    plot.colorbar()

for k in range(n_modes):
    for l in range(n_modes):
        plot.figure(200+10*k+l)
        plot.subplot(3,1,1)
        plot.contourf(X,Y,L_kl[:,:,k,l],21)
        plot.colorbar()
        plot.subplot(3,1,2)
        plot.contourf(X,Y,Lkl[:,:,k,l],21)
        plot.colorbar()
        plot.subplot(3,1,3)
        plot.contourf(X,Y,L_kl[:,:,k,l]-Lkl[:,:,k,l],21)
        plot.colorbar()




for nm in [1]:
    plot.figure(10+nm)
    plot.contourf(X,Y,phi_x_A[:,:,nm])
    plot.colorbar()




plot.show()






