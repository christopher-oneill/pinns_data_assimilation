import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Cartesian PINN functions and boundary conditions

# Author: 
# Christopher O'Neill
# MSc Student
# Laboratory for Turbulence Research in Aerodynamics and Flow Control
# Prof Robert Martinuzzi and Prof Chris Morton
# University of Calgary
# Feb-November 2023

# for details of the derivation see the accompanying pdf

# with inspiration and help from
# Jakob GR von Saldern
# PhD Student
# Laboratory for Flow Instabilities and Dynamics
# Prof Kilian Oberleithner
# Technical University of Berlin

# steady navier stokes functions
@tf.function
def steady_NS(model_steady,colloc_tensor):
    up = model_steady(colloc_tensor)
    # knowns
    ux = up[:,0]*model_steady.ScalingParameters.MAX_ux
    uy = up[:,1]*model_steady.ScalingParameters.MAX_uy
    # unknowns
    p = up[:,2]*model_steady.ScalingParameters.MAX_p
    
    # compute the gradients of the quantities
    # first derivatives
    #(dux,duy,dp)  =tf.gradients((ux,uy,p),(colloc_tensor,colloc_tensor,colloc_tensor)) # somehow gives wrong gradients due to averaging
    (dux)  =tf.gradients((ux),(colloc_tensor))[0]
    (duy)  =tf.gradients((uy),(colloc_tensor))[0]
    (dp)  =tf.gradients((p),(colloc_tensor))[0]

    # ux gradient
    ux_x = dux[:,0]/model_steady.ScalingParameters.MAX_x
    ux_y = dux[:,1]/model_steady.ScalingParameters.MAX_y   
    # uy gradient
    uy_x = duy[:,0]/model_steady.ScalingParameters.MAX_x
    uy_y = duy[:,1]/model_steady.ScalingParameters.MAX_y
    # pressure gradients
    p_x = dp[:,0]/model_steady.ScalingParameters.MAX_x
    p_y = dp[:,1]/model_steady.ScalingParameters.MAX_y

    # second derivatives
    #(dux_x,dux_y,duy_x,duy_y) = tf.gradients((),(colloc_tensor,colloc_tensor,colloc_tensor))
    (dux_x) = tf.gradients((ux_x),(colloc_tensor))[0]
    (dux_y) = tf.gradients((ux_y),(colloc_tensor))[0]
    (duy_x) = tf.gradients((uy_x),(colloc_tensor))[0]
    (duy_y) = tf.gradients((uy_y),(colloc_tensor))[0]
    ux_xx = dux_x[:,0]/model_steady.ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/model_steady.ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/model_steady.ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/model_steady.ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + p_x - (model_steady.ScalingParameters.nu_mol)*(ux_xx+ux_yy)  
    f_y = (ux*uy_x + uy*uy_y) + p_y - (model_steady.ScalingParameters.nu_mol)*(uy_xx+uy_yy)
    f_mass = ux_x + uy_y
    
    return f_x, f_y, f_mass

# function wrapper, combine data and physics loss
def steady_loss_wrapper(model_steady,colloc_tensor_f,BC_ns,BC_p): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
    
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_p = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss = tf.reduce_mean(data_loss_ux + data_loss_uy + data_loss_p)

        if (model_steady.ScalingParameters.physics_loss_coefficient!=0):
            # compute physics loss
            mx,my,mass = RANS_reynolds_stress_cartesian(model_steady,colloc_tensor_f)
            physical_loss1 = tf.reduce_mean(tf.square(mx))
            physical_loss2 = tf.reduce_mean(tf.square(my))
            physical_loss3 = tf.reduce_mean(tf.square(mass))

            BC_pressure_loss = BC_RANS_reynolds_stress_pressure(model_steady,BC_p)
            BC_no_slip_loss = BC_RANS_reynolds_stress_no_slip(model_steady,BC_ns)
            physics_loss = tf.reduce_mean(physical_loss1 + physical_loss2 + physical_loss3 + BC_pressure_loss + BC_no_slip_loss)
            return  data_loss+ model_steady.ScalingParameters.physics_loss_coefficient*physics_loss # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 
        else:
            return data_loss
        
    return mean_loss


# RANS functions with specified reynolds stresses
@tf.function
def RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,colloc_tensor):

    up = model_RANS(colloc_tensor)
    # knowns
    ux = up[:,0]*ScalingParameters.MAX_ux
    uy = up[:,1]*ScalingParameters.MAX_uy
    uxux = up[:,2]*ScalingParameters.MAX_uxppuxpp
    uxuy = up[:,3]*ScalingParameters.MAX_uxppuypp
    uyuy = up[:,4]*ScalingParameters.MAX_uyppuypp
    # unknowns
    p = up[:,5]*ScalingParameters.MAX_p
    
    # compute the gradients of the quantities
    
    # first gradients
    dux = tf.gradients((ux,), (colloc_tensor))[0]
    duy = tf.gradients((uy,), (colloc_tensor))[0]
    duxux = tf.gradients((uxux,), (colloc_tensor))[0]
    duxuy = tf.gradients((uxuy,), (colloc_tensor))[0]
    duyuy = tf.gradients((uyuy,), (colloc_tensor))[0]
    dp = tf.gradients((p,), (colloc_tensor))[0]
    # ux grads

    ux_x = dux[:,0]/ScalingParameters.MAX_x
    ux_y = dux[:,1]/ScalingParameters.MAX_y
    
    # uy gradient
    uy_x = duy[:,0]/ScalingParameters.MAX_x
    uy_y = duy[:,1]/ScalingParameters.MAX_y


    # gradient unmodeled reynolds stresses
    uxux_x = duxux[:,0]/ScalingParameters.MAX_x
    uxuy_x = duxuy[:,0]/ScalingParameters.MAX_x
    uxuy_y = duxuy[:,1]/ScalingParameters.MAX_y
    uyuy_y = duyuy[:,1]/ScalingParameters.MAX_y

    # pressure gradients
    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y

    # second gradients
    (dux_x) = tf.gradients((ux_x,),(colloc_tensor,))[0]
    (dux_y) = tf.gradients((ux_y,),(colloc_tensor,))[0]
    (duy_x) = tf.gradients((uy_x,),(colloc_tensor,))[0]
    (duy_y) = tf.gradients((uy_y,),(colloc_tensor,))[0]
    # and second derivative
    ux_xx = dux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = dux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = duy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = duy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    
    return f_x, f_y, f_mass


@tf.function
def RANS_reynolds_stress_cartesian_GradTape(model_RANS,ScalingParameters,colloc_points):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(model_RANS.trainable_variables)
        tape1.watch(colloc_points)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(model_RANS.trainable_variables)
            tape2.watch(colloc_points)
            up = model_RANS(colloc_points)
            
            ux = up[:,0]*ScalingParameters.MAX_ux
            uy = up[:,1]*ScalingParameters.MAX_uy
            uxux = up[:,2]*ScalingParameters.MAX_uxppuxpp
            uxuy = up[:,3]*ScalingParameters.MAX_uxppuypp
            uyuy = up[:,4]*ScalingParameters.MAX_uyppuypp
            p = up[:,5]*ScalingParameters.MAX_p
            # dont record the derivatives of up
            with tape2.stop_recording():
                # we need to record these with tape1 so that we can differentiate them again
                dux = tape2.gradient(ux,colloc_points)
                duy = tape2.gradient(uy,colloc_points)
                with tape1.stop_recording():
                    # we dont need to record these for higher derivatives
                    duxux = tape2.gradient(uxux,colloc_points)
                    duxuy = tape2.gradient(uxuy,colloc_points)
                    duyuy = tape2.gradient(uyuy,colloc_points)
                    dp = tape2.gradient(p,colloc_points)
            # exit the scope of tape 2
            ux_x = dux[:,0]/ScalingParameters.MAX_x
            ux_y = dux[:,1]/ScalingParameters.MAX_y
            uy_x = duy[:,0]/ScalingParameters.MAX_x
            uy_y = duy[:,1]/ScalingParameters.MAX_y
        with tape1.stop_recording():
            ddux_x = tape1.gradient(ux_x,colloc_points)
            ddux_y = tape1.gradient(ux_y,colloc_points) 
            dduy_x = tape1.gradient(uy_x,colloc_points) 
            dduy_y = tape1.gradient(uy_y,colloc_points) 
        
    del tape1
    del tape2

    # scale remaining gradients

    uxux_x = duxux[:,0]/ScalingParameters.MAX_x
    uxuy_x = duxuy[:,0]/ScalingParameters.MAX_x
    uxuy_y = duxuy[:,1]/ScalingParameters.MAX_y
    uyuy_y = duyuy[:,1]/ScalingParameters.MAX_y

    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y

    ux_xx = ddux_x[:,0]/ScalingParameters.MAX_x
    ux_yy = ddux_y[:,1]/ScalingParameters.MAX_y
    uy_xx = dduy_x[:,0]/ScalingParameters.MAX_x
    uy_yy = dduy_y[:,1]/ScalingParameters.MAX_y

    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxux_x + uxuy_y) + p_x - (ScalingParameters.nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxuy_x + uyuy_y) + p_y - (ScalingParameters.nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    
    return f_x, f_y, f_mass


def example_RANS_reynolds_stress_loss_wrapper(model_RANS,ScalingParameters,colloc_points,BC_p,BC_ns,BC_inlet,BC_inlet2): # def custom_loss_wrapper(colloc_tensor_f,BCs,BCs_p,BCs_t):
     
    def mean_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
        data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
        data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
        data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
        data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''
        
        data_loss = tf.reduce_mean(data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp)

        if (ScalingParameters.physics_loss_coefficient==0):
            physics_loss = 0.0
        else:
            if (ScalingParameters.colloc_batch_size==colloc_points.shape[0]):
                # all colloc points
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,colloc_points)
            else:
                # random selection of collocation points with batch size
                rand_colloc_points = np.random.choice(colloc_points.shape[0],ScalingParameters.batch_size)
                mx,my,mass = RANS_reynolds_stress_cartesian(model_RANS,tf.gather(colloc_points,rand_colloc_points))
            physical_loss1 = tf.reduce_mean(tf.square(mx))
            physical_loss2 = tf.reduce_mean(tf.square(my))
            physical_loss3 = tf.reduce_mean(tf.square(mass))
            physics_loss = tf.reduce_mean(physical_loss1 + physical_loss2 + physical_loss3)
        if (ScalingParameters.boundary_loss_coefficient==0):
            boundary_loss = 0.0
        else:
            BC_pressure_loss = 500.0*BC_RANS_reynolds_stress_pressure(model_RANS,BC_p) # scaled to compensate the reduce sum on other BCs
            BC_no_slip_loss = BC_RANS_reynolds_stress_no_slip(model_RANS,BC_ns)
            BC_inlet_loss = BC_RANS_inlet(model_RANS,BC_inlet)
            BC_inlet_loss2 = BC_RANS_inlet2(model_RANS,BC_inlet2)
                       
            boundary_loss = (BC_pressure_loss + BC_no_slip_loss + BC_inlet_loss+BC_inlet_loss2)

        combined_phyisics_loss = ScalingParameters.physics_loss_coefficient*physics_loss + ScalingParameters.boundary_loss_coefficient*boundary_loss
        dynamic_data_weight = tf.math.exp(tf.math.ceil(tf.math.log(combined_phyisics_loss+1E-30)))
        return  tf.math.reduce_max((dynamic_data_weight,1))*ScalingParameters.data_loss_coefficient*data_loss + combined_phyisics_loss
        
    return mean_loss

@tf.function
def RANS_poisson_equation(model_RANS,ScalingParameters,colloc_tensor):
    # reynolds averaged poisson equation, from: Turbulent Flows, Stephen B. Pope. Page 85, Eqn 4.13
    up = model_RANS(colloc_tensor)
    # knowns
    ux = up[:,0]*ScalingParameters.MAX_ux
    uy = up[:,1]*ScalingParameters.MAX_uy
    uxux = up[:,2]*ScalingParameters.MAX_uxppuxpp
    uxuy = up[:,3]*ScalingParameters.MAX_uxppuypp
    uyuy = up[:,4]*ScalingParameters.MAX_uyppuypp
    # unknowns
    p = up[:,5]*ScalingParameters.MAX_p
    
    # compute the gradients of the quantities
    
    # ux gradient
    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/ScalingParameters.MAX_x
    ux_y = dux[:,1]/ScalingParameters.MAX_y

    # uy gradient
    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/ScalingParameters.MAX_x
    uy_y = duy[:,1]/ScalingParameters.MAX_y

    # gradient unmodeled reynolds stresses
    uxux_x = tf.gradients(uxux, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    uxux_xx = tf.gradients(uxux_x, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    uxuy_x = tf.gradients(uxuy, colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    uxuy_xy =  tf.gradients(uxuy_x, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    uyuy_y = tf.gradients(uyuy, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y
    uyuy_yy = tf.gradients(uyuy_y, colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor)[0]
    p_x = dp[:,0]/ScalingParameters.MAX_x
    p_y = dp[:,1]/ScalingParameters.MAX_y
    p_xx = tf.gradients(p_x,colloc_tensor)[0][:,0]/ScalingParameters.MAX_x
    p_yy = tf.gradients(p_y,colloc_tensor)[0][:,1]/ScalingParameters.MAX_y

    f_p = (p_xx+p_yy) + (ux_x*ux_x+2.0*ux_y*uy_x+uy_y*uy_y) + (uxux_xx+2.0*uxuy_xy+uyuy_yy)
    return f_p


@tf.function
def predict_RANS_reynolds_stress_cartesian(model_RANS,ScalingParameters,colloc_tensor):
    u_mean = model_RANS(colloc_tensor)
    ux = u_mean[:,0]*ScalingParameters.MAX_ux
    uy = u_mean[:,1]*ScalingParameters.MAX_uy

    dux = tf.gradients(ux, colloc_tensor)[0]
    ux_x = dux[:,0]/ScalingParameters.MAX_x
    ux_y = dux[:,1]/ScalingParameters.MAX_y

    duy = tf.gradients(uy, colloc_tensor)[0]
    uy_x = duy[:,0]/ScalingParameters.MAX_x
    uy_y = duy[:,1]/ScalingParameters.MAX_y

    return tf.stack([ux,uy,ux_x,ux_y,uy_x,uy_y],axis=1)

@tf.function
def BC_RANS_reynolds_stress_outlet_pressure_sign(model_RANS,BC_points):
    # pressure outlet should be slightly negative
    up = model_RANS(BC_points)
    # knowns
    # unknowns
    p = up[:,5]
    return tf.cast((p>0.0),tf.float64)

@tf.function
def BC_RANS_reynolds_stress_no_slip(model_RANS,BC_points):
    up = model_RANS(BC_points)
    # knowns
    ux = up[:,0]
    uy = up[:,1]
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_sum(tf.square(ux)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))

@tf.function
def BC_RANS_inlet2(model_RANS,BC_points):
    up = model_RANS(BC_points)
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_mean(tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp))
 # note there is no point where the pressure is close to zero, so we neglect it in the mean field model

@tf.function
def BC_RANS_reynolds_stress_pressure(model_RANS,BC_points):
    up = model_RANS(BC_points)
    # knowns
    # unknowns
    p = up[:,5]
    return tf.reduce_mean(tf.square(p))

@tf.function
def BC_RANS_inlet(model_RANS,ScalingParameters,BC_points):
    up = model_RANS(BC_points)
    ux = up[:,0]*ScalingParameters.MAX_ux
    uy = up[:,1] # no need to scale since they should go to zero
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    return tf.reduce_mean(tf.square(ux-1.0))+tf.reduce_mean(tf.square(uy))+tf.reduce_mean(tf.square(uxppuxpp))+tf.reduce_mean(tf.square(uxppuypp))+tf.reduce_mean(tf.square(uyppuypp))
 # note there is no point where the pressure is close to zero, so we neglect it in the mean field model

@tf.function
def BC_cylinder_inside(model_RANS,ScalingParameters,BC_points):
    up = model_RANS(BC_points)
    return tf.reduce_mean(tf.square(up[:,0]))+tf.reduce_mean(tf.square(up[:,1]))+tf.reduce_mean(tf.square(up[:,2]))+tf.reduce_mean(tf.square(up[:,3]))+tf.reduce_mean(tf.square(up[:,4]))

@tf.function
def BC_RANS_wall(model_RANS,ScalingParameters,BC_points):
    wall_coord = BC_points[:,0:2]
    wall_angle = BC_points[:,2]
    up = model_RANS(wall_coord)
    # knowns
    ux = up[:,0]
    uy = up[:,1]
    uxppuxpp = up[:,2]
    uxppuypp = up[:,3]
    uyppuypp = up[:,4]
    #p = up[:,5]
    #(dp,) = tf.gradients((p,),(wall_coord)) 
    #p_x = dp[:,0]/ScalingParameters.MAX_x
    #p_y = dp[:,1]/ScalingParameters.MAX_y
    #grad_p_norm = p_x*tf.cos(wall_angle)+p_y*tf.sin(wall_angle)

    return tf.reduce_sum(tf.square(ux)+tf.square(uy)+tf.square(uxppuxpp)+tf.square(uxppuypp)+tf.square(uyppuypp)) #+tf.square(grad_p_norm)





# fans functions, with specified fourier stresses
def example_FANS_loss_wrapper(model_FANS,colloc_tensor_f,colloc_grads,ns_BC_points,p_BC_points,inlet_BC_points): 
    
    def custom_loss(y_true, y_pred):
        # this needs to match the order that they are concatinated in the array when setting up the network
        # additionally, the known quantities must be first, unknown quantites second
        data_loss_phi_xr = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0])
        data_loss_phi_xi = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) 
        data_loss_phi_yr = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) 
        data_loss_phi_yi = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) 
        data_loss_tau_xx_r = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) 
        data_loss_tau_xx_i = keras.losses.mean_squared_error(y_true[:,5], y_pred[:,5]) 
        data_loss_tau_xy_r = keras.losses.mean_squared_error(y_true[:,6], y_pred[:,6]) 
        data_loss_tau_xy_i = keras.losses.mean_squared_error(y_true[:,7], y_pred[:,7])
        data_loss_tau_yy_r = keras.losses.mean_squared_error(y_true[:,8], y_pred[:,8]) 
        data_loss_tau_yy_i = keras.losses.mean_squared_error(y_true[:,9], y_pred[:,9]) 
        data_loss = data_loss_phi_xr + data_loss_phi_xi + data_loss_phi_yr + data_loss_phi_yi +data_loss_tau_xx_r+data_loss_tau_xx_i+data_loss_tau_xy_r+data_loss_tau_xy_i+data_loss_tau_yy_r+data_loss_tau_yy_i


        if model_FANS.ScalingParameters.physics_loss_coefficient!=0:
            rand_colloc_points = np.random.choice(colloc_tensor_f.shape[0],model_FANS.ScalingParameters.batch_size)
            mxr,mxi,myr,myi,massr,massi = FANS_cartesian(model_FANS,tf.gather(colloc_tensor_f,rand_colloc_points),tf.gather(colloc_grads,rand_colloc_points))
            loss_mxr = tf.reduce_mean(tf.square(mxr))
            loss_mxi = tf.reduce_mean(tf.square(mxi))
            loss_myr = tf.reduce_mean(tf.square(myr))
            loss_myi = tf.reduce_mean(tf.square(myi))
            loss_massr = tf.reduce_mean(tf.square(massr))
            loss_massi = tf.reduce_mean(tf.square(massi))

            BC_pressure_loss = BC_FANS_pressure_outlet(model_FANS,p_BC_points)
            BC_no_slip_loss = BC_FANS_no_slip(model_FANS,ns_BC_points)
            BC_inlet_loss = BC_FANS_inlet(model_FANS,inlet_BC_points)
            return data_loss + model_FANS.ScalingParameters.physics_loss_coefficient*(loss_mxr + loss_mxi + loss_myr + loss_myi + loss_massr + loss_massi + BC_pressure_loss + BC_no_slip_loss + 5.0*BC_inlet_loss) 
        else:
            return data_loss

    return custom_loss

@tf.function
def FANS_cartesian(model_FANS,colloc_tensor, mean_grads):
    up = model_FANS(colloc_tensor)
    # velocity fourier coefficients
    phi_xr = up[:,0]*model_FANS.ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*model_FANS.ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*model_FANS.ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*model_FANS.ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*model_FANS.ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*model_FANS.ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*model_FANS.ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*model_FANS.ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*model_FANS.ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*model_FANS.ScalingParameters.MAX_tau_yy_i
    # unknowns, pressure fourier modes
    psi_r = up[:,10]*model_FANS.ScalingParameters.MAX_psi
    psi_i = up[:,11]*model_FANS.ScalingParameters.MAX_psi
    
    ux = mean_grads[:,0]
    uy = mean_grads[:,1]
    ux_x = mean_grads[:,2]
    ux_y = mean_grads[:,3]
    uy_x = mean_grads[:,4]
    uy_y = mean_grads[:,5]


    # first derivatives
    (dphi_xr,dphi_xi,dphi_yr,dphi_yi,dtau_xx_r,dtau_xx_i,dtau_xy_r,dtau_xy_i,dtau_yy_r,dtau_yy_i,dpsi_r,dpsi_i) = tf.gradients((phi_xr,phi_xi,phi_yr,phi_yi,tau_xx_r,tau_xx_i,tau_xy_r,tau_xy_i,tau_yy_r,tau_yy_i,psi_r,psi_i), (colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))
    
    # velocity mode gradients    
    phi_xr_x = dphi_xr[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_xr_y = dphi_xr[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_xi_x = dphi_xi[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_xi_y = dphi_xi[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_yr_x = dphi_yr[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_yr_y = dphi_yr[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_yi_x = dphi_yi[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_yi_y = dphi_yi[:,1]/model_FANS.ScalingParameters.MAX_y

    # gradient reynolds stress fourier component, real
    tau_xx_r_x = dtau_xx_r[:,0]/model_FANS.ScalingParameters.MAX_x
    tau_xy_r_x = dtau_xy_r[:,0]/model_FANS.ScalingParameters.MAX_x
    tau_xy_r_y = dtau_xy_r[:,1]/model_FANS.ScalingParameters.MAX_y
    tau_yy_r_y = dtau_yy_r[:,1]/model_FANS.ScalingParameters.MAX_y
    # gradient reynolds stress fourier component, complex
    tau_xx_i_x = dtau_xx_i[:,0]/model_FANS.ScalingParameters.MAX_x
    tau_xy_i_x = dtau_xy_i[:,0]/model_FANS.ScalingParameters.MAX_x
    tau_xy_i_y = dtau_xy_i[:,1]/model_FANS.ScalingParameters.MAX_y
    tau_yy_i_y = dtau_yy_i[:,1]/model_FANS.ScalingParameters.MAX_y

    # pressure gradients
    psi_r_x = dpsi_r[:,0]/model_FANS.ScalingParameters.MAX_x
    psi_r_y = dpsi_r[:,1]/model_FANS.ScalingParameters.MAX_y
    psi_i_x = dpsi_i[:,0]/model_FANS.ScalingParameters.MAX_x
    psi_i_y = dpsi_i[:,1]/model_FANS.ScalingParameters.MAX_y

    # second derivatives

    (dphi_xr_x,dphi_xr_y,dphi_xi_x,dphi_xi_y,dphi_yr_x,dphi_yr_y,dphi_yi_x,dphi_yi_y) = tf.gradients((phi_xr_x,phi_xr_y,phi_xi_x,phi_xi_y,phi_yr_x,phi_yr_y,phi_yi_x,phi_yi_y),(colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor,colloc_tensor))

    phi_xr_xx = dphi_xr_x[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_xr_yy = dphi_xr_y[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_xi_xx = dphi_xi_x[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_xi_yy = dphi_xi_y[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_yr_xx = dphi_yr_x[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_yr_yy = dphi_yr_y[:,1]/model_FANS.ScalingParameters.MAX_y
    phi_yi_xx = dphi_yi_x[:,0]/model_FANS.ScalingParameters.MAX_x
    phi_yi_yy = dphi_yi_y[:,1]/model_FANS.ScalingParameters.MAX_y


    # governing equations
    f_xr = -model_FANS.ScalingParameters.omega*phi_xi+(phi_xr*ux_x + phi_yr*ux_y+ ux*phi_xr_x +uy*phi_xr_y ) + (tau_xx_r_x + tau_xy_r_y) + psi_r_x - (model_FANS.ScalingParameters.nu_mol)*(phi_xr_xx+phi_xr_yy)  
    f_xi =  model_FANS.ScalingParameters.omega*phi_xr+(phi_xi*ux_x + phi_yi*ux_y+ ux*phi_xi_x +uy*phi_xi_y ) + (tau_xx_i_x + tau_xy_i_y) + psi_i_x - (model_FANS.ScalingParameters.nu_mol)*(phi_xi_xx+phi_xi_yy)  
    f_yr = -model_FANS.ScalingParameters.omega*phi_yi+(phi_xr*uy_x + phi_yr*uy_y+ ux*phi_yr_x +uy*phi_yr_y ) + (tau_xy_r_x + tau_yy_r_y) + psi_r_y - (model_FANS.ScalingParameters.nu_mol)*(phi_yr_xx+phi_yr_yy) 
    f_yi =  model_FANS.ScalingParameters.omega*phi_yr+(phi_xi*uy_x + phi_yi*uy_y+ ux*phi_yi_x +uy*phi_yi_y ) + (tau_xy_i_x + tau_yy_i_y) + psi_i_y - (model_FANS.ScalingParameters.nu_mol)*(phi_yi_xx+phi_yi_yy)  
    f_mr = phi_xr_x + phi_yr_y
    f_mi = phi_xi_x + phi_yi_y

    return f_xr, f_xi, f_yr, f_yi, f_mr, f_mi

@tf.function
def BC_FANS_pressure_outlet(model_FANS,BC_points):
    up = model_FANS(BC_points)
    # unknowns, pressure fourier modes
    psi_r = tf.square(up[:,10]*model_FANS.ScalingParameters.MAX_psi)
    psi_i = tf.square(up[:,11]*model_FANS.ScalingParameters.MAX_psi)
    return tf.square(psi_r[1]-psi_r[0])+tf.square(psi_i[1]-psi_i[0]) # square and then subtract 

@tf.function
def BC_FANS_no_slip(model_FANS,BC_points):
    up = model_FANS(BC_points)
    # velocity fourier coefficients
    phi_xr = up[:,0]*model_FANS.ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*model_FANS.ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*model_FANS.ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*model_FANS.ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*model_FANS.ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*model_FANS.ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*model_FANS.ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*model_FANS.ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*model_FANS.ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*model_FANS.ScalingParameters.MAX_tau_yy_i
    return tf.reduce_mean(tf.square(phi_xr)+tf.square(phi_xi)+tf.square(phi_yr)+tf.square(phi_yi)+tf.square(tau_xx_r)+tf.square(tau_xx_i)+tf.square(tau_xy_r)+tf.square(tau_xy_i)+tf.square(tau_yy_r)+tf.square(tau_yy_i))

@tf.function
def BC_FANS_inlet(model_FANS,BC_points):
    up = model_FANS(BC_points)
    # velocity fourier coefficients
    phi_xr = up[:,0]*model_FANS.ScalingParameters.MAX_phi_xr
    phi_xi = up[:,1]*model_FANS.ScalingParameters.MAX_phi_xi
    phi_yr = up[:,2]*model_FANS.ScalingParameters.MAX_phi_yr
    phi_yi = up[:,3]*model_FANS.ScalingParameters.MAX_phi_yi

    # fourier coefficients of the fluctuating field
    tau_xx_r = up[:,4]*model_FANS.ScalingParameters.MAX_tau_xx_r
    tau_xx_i = up[:,5]*model_FANS.ScalingParameters.MAX_tau_xx_i
    tau_xy_r = up[:,6]*model_FANS.ScalingParameters.MAX_tau_xy_r
    tau_xy_i = up[:,7]*model_FANS.ScalingParameters.MAX_tau_xy_i
    tau_yy_r = up[:,8]*model_FANS.ScalingParameters.MAX_tau_yy_r
    tau_yy_i = up[:,9]*model_FANS.ScalingParameters.MAX_tau_yy_i

    psi_r = up[:,10]*model_FANS.ScalingParameters.MAX_psi
    psi_i = up[:,11]*model_FANS.ScalingParameters.MAX_psi
    return tf.reduce_sum(tf.square(phi_xr)+tf.square(phi_xi)+tf.square(phi_yr)+tf.square(phi_yi)+tf.square(tau_xx_r)+tf.square(tau_xx_i)+tf.square(tau_xy_r)+tf.square(tau_xy_i)+tf.square(tau_yy_r)+tf.square(tau_yy_i)+tf.square(psi_r)+tf.square(psi_i)) # note there is no point where the pressure is close to zero, so we neglect it in the mean field model

def FANS_cartesian_batch(model_FANS,colloc_tensor,mean_grads,batch_size=32):
    # useful if the memory allocation of the whole collocation points is too large
    n_x = colloc_tensor.shape[0]
    n_batch = np.int64(np.ceil(n_x/batch_size))

    list_xr = []
    list_xi = []
    list_yr = []
    list_yi = []
    list_mr = []
    list_mi = []

    progbar = keras.utils.Progbar(n_batch)
    # loop over the colloc points in batches
    for i in range(0,n_batch):
        progbar.update(i+1)
        batch_inds = np.arange(i*batch_size,np.min([(i+1)*batch_size,n_x]))
        t_xr, t_xi, t_yr, t_yi, t_mr, t_mi = FANS_cartesian(model_FANS,tf.gather(colloc_tensor,batch_inds),tf.gather(mean_grads,batch_inds))
        list_xr.append(t_xr)
        list_xi.append(t_xi)
        list_yr.append(t_yr)
        list_yi.append(t_yi)
        list_mr.append(t_mr)
        list_mi.append(t_mi)
    
    # combine the batches together
    f_xr = tf.concat(list_xr,axis=0)
    f_xi = tf.concat(list_xi,axis=0)
    f_yr = tf.concat(list_yr,axis=0)
    f_yi = tf.concat(list_yi,axis=0)
    f_mr = tf.concat(list_mr,axis=0)
    f_mi = tf.concat(list_mi,axis=0)

    return f_xr, f_xi, f_yr, f_yi, f_mr, f_mi

