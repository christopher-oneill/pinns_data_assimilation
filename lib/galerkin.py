import numpy as np

def galerkin_gradients(X,Y,ux,uy,p,phi_x,phi_y,psi):
    # the assumed size of ux is (nx,ny)
    ux_x = np.gradient(ux,X,axis=0)
    ux_y = np.gradient(ux,Y,axis=1)
    ux_xx = np.gradient(ux_x,X,axis=0)
    ux_yy = np.gradient(ux_y,Y,axis=1)

    uy_x = np.gradient(uy,X,axis=0)
    uy_y = np.gradient(uy,Y,axis=1)
    uy_xx = np.gradient(uy_x,X,axis=0)
    uy_yy = np.gradient(uy_y,Y,axis=1)

    # the assumed size of phi_x is (nx,ny,nk)
    phi_x_x = np.gradient(phi_x,X,axis=0)
    phi_x_y = np.gradient(phi_x,Y,axis=1)
    phi_x_xx = np.gradient(phi_x_x,X,axis=0)
    phi_x_yy = np.gradient(phi_x_y,Y,axis=1)
    # the assumed size of phi_y is (nx,ny,nk)
    phi_y_x = np.gradient(phi_y,X,axis=0)
    phi_y_y = np.gradient(phi_y,Y,axis=1)
    phi_y_xx = np.gradient(phi_y_x,X,axis=0)
    phi_y_yy = np.gradient(phi_y_y,Y,axis=1)

    # the assumed size of p is (nx,ny)
    p_x = np.gradient(p,X,axis=0)
    p_y = np.gradient(p,Y,axis=1)
    # the assumed size of psi is (nx,ny,nk)
    psi_x = np.gradient(psi,X,axis=0)
    psi_y = np.gradient(psi,Y,axis=1)

    return ux_x,ux_y,ux_xx,ux_yy,uy_x,uy_y,uy_xx,uy_yy,p_x,p_y,phi_x_x,phi_x_y,phi_x_xx,phi_x_yy,phi_y_x,phi_y_y,phi_y_xx,phi_y_yy,psi_x,psi_y


def galerkin_projection(nu,ux,ux_x,ux_y,ux_xx,ux_yy,uy,uy_x,uy_y,uy_xx,uy_yy,p,p_x,p_y,phi_x,phi_x_x,phi_x_y,phi_x_xx,phi_x_yy,phi_y,phi_y_x,phi_y_y,phi_y_xx,phi_y_yy,psi,psi_x,psi_y):
    nx = phi_x.shape[0]
    nk = phi_x.shape[1]

    Fk =  np.zeros([nx,nk],dtype=np.double)
    Lkl = np.zeros([nx,nk,nk],dtype=np.double)
    Qklm = np.zeros([nx,nk,nk,nk],np.double)

    for k in range(nk):
        Fk[:,k] = -phi_x[:,k]*(2*ux*ux_x+ux*uy_y+uy*ux_y)- phi_y[:,k]*(ux*uy_x+uy*ux_x+2*uy*uy_y) - phi_x[:,k]*p_x - phi_y[:,k]*p_y + nu*(phi_x[:,k]*(ux_xx+ux_yy)+phi_y[:,k]*(uy_xx+uy_yy))

        for l in range(nk):
            Lkl[:,k,l] = (-phi_x[:,k]*(phi_x[:,l]*ux_x+phi_y[:,l]*ux_y) - phi_y[:,k]*(phi_x[:,l]*uy_x+phi_y[:,l]*uy_y)
                          -phi_x[:,k]*(ux*phi_x_x[:,l]+uy*phi_x_y[:,l]) - phi_y[:,k]*(ux*phi_y_x[:,l]+uy*phi_y_y[:,l])
                          -phi_x[:,k]*psi_x[:,l]-phi_y[:,k]*psi_y[:,l]
                          +nu*(phi_x[:,k]*(phi_x_xx[:,l]+phi_x_yy[:,l])+phi_y[:,k]*(phi_y_xx[:,l]+phi_y_yy[:,l])))
            for m in range(nk):
                Qklm[:,k,l,m] = -phi_x[:,k]*(phi_x[:,l]*phi_x_x[:,m] + phi_y[:,l]*phi_x_y[:,m]) -phi_y[:,k]*(phi_x[:,l]*phi_y_x[:,m] + phi_y[:,l]*phi_y_y[:,m])

    return Fk, Lkl, Qklm


def galerkin_projection_complex(ux, ux_x, ux_y, ux_xx, ux_yy, uy, uy_x, uy_y, uy_xx, uy_yy, p, p_x, p_y, phi_xr, phi_xr_x, phi_xr_y, phi_xr_xx, phi_xr_yy, phi_xi, phi_xi_x, phi_xi_y, phi_xi_xx, phi_xi_yy, phi_yr, phi_yr_x, phi_yr_y, phi_yr_xx, phi_yr_yy, phi_yi, phi_yi_x, phi_yi_y, phi_yi_xx, phi_yi_yy, psi_r, psi_r_x, psi_r_y, psi_i, psi_i_x, psi_i_y):
    nx = phi_xr.shape[0]
    nk = phi_xr.shape[1]

    Fk =  np.zeros([nx,nk],dtype=np.cdouble)
    Lkl = np.zeros([nx,nk,nk],dtype=np.cdouble)
    Qklm = np.zeros([nx,nk,nk,nk],np.cdouble)

    for k in range(nk):
        # phi_k
        phi_k_x = phi_xr[:,k]+1j*phi_xi[:,k]
        phi_k_y = phi_yr[:,k]+1j*phi_yi[:,k]
        Fk[:,k] = -phi_k_x*np.cdouble(2*ux*ux_x+ux*uy_y+uy*ux_y)- phi_k_y*np.cdouble(ux*uy_x+uy*ux_x+2*uy*uy_y) - phi_k_x*np.cdouble(p_x) - phi_k_y*np.cdouble(p_y) + np.cdouble(nu_mol)*(phi_k_x*np.cdouble(ux_xx+ux_yy)+phi_k_y*np.cdouble(uy_xx+uy_yy))

        for l in range(nk):
            #phi_l
            phi_l_x = phi_xr[:,l]+1j*phi_xi[:,l]
            phi_l_y = phi_yr[:,l]+1j*phi_yi[:,l]
            # derivatives of phi_l
            phi_l_x_x = phi_xr_x[:,l] + 1j*phi_xi_x[:,l]
            phi_l_x_y = phi_xr_y[:,l] + 1j*phi_xi_y[:,l]
            phi_l_y_x = phi_yr_x[:,l] + 1j*phi_yi_x[:,l]
            phi_l_y_y = phi_yr_y[:,l] + 1j*phi_yi_y[:,l]
            phi_l_x_xx = phi_xr_xx[:,l] + 1j*phi_xi_xx[:,l]
            phi_l_x_yy = phi_xr_yy[:,l] + 1j*phi_xi_yy[:,l]
            phi_l_y_xx = phi_yr_xx[:,l] + 1j*phi_yi_xx[:,l]
            phi_l_y_yy = phi_yr_yy[:,l] + 1j*phi_yi_yy[:,l]
            # psi derivatives
            psi_l_x = psi_r_x[:,l]+1j*psi_i_x[:,l]
            psi_l_y = psi_r_y[:,l]+1j*psi_i_y[:,l]

            Lkl[:,k,l] = (-phi_k_x*(phi_l_x*np.cdouble(ux_x)+phi_l_y*np.cdouble(ux_y)) -phi_k_y*(phi_l_x*np.cdouble(uy_x)+phi_l_y*np.cdouble(uy_y))
                          -phi_k_x*(np.cdouble(ux)*phi_l_x_x+np.cdouble(uy)*phi_l_x_y)-phi_k_y*(np.cdouble(ux)*phi_l_y_x+np.cdouble(uy)*phi_l_y_y)
                          -phi_k_x*psi_l_x-phi_k_y*psi_l_y
                          +np.cdouble(nu_mol)*(phi_k_x*(phi_l_x_xx+phi_l_x_yy)+phi_k_y*(phi_l_y_xx+phi_l_y_yy)))
    
            for m in range(nk):
                # derivatives of phi_m
                phi_m_x_x = phi_xr_x[:,m] + 1j*phi_xi_x[:,m]
                phi_m_x_y = phi_xr_y[:,m] + 1j*phi_xi_y[:,m]
                phi_m_y_x = phi_yr_x[:,m] + 1j*phi_yi_x[:,m]
                phi_m_y_y = phi_yr_y[:,m] + 1j*phi_yi_y[:,m]

                Qklm[:,k,l,m] = -phi_k_x*(phi_l_x*phi_m_x_x + phi_l_y*phi_m_x_y) -phi_k_y*(phi_l_x*phi_m_y_x + phi_l_y*phi_m_y_y)

    return Fk, Lkl, Qklm
