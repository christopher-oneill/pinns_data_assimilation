import tensorflow as tf
import tensorflow.keras as keras



# mean field functions, mean + u'u', u'v', v'v'
@tf.function
def net_f_mean_cartesian(): #,model_mean,nu_mol,MAX_x,MAX_y,MAX_ux,MAX_uy,MAX_p,MAX_uxppuxpp,MAX_uxppuypp,MAX_uyppuypp
    up = model_mean(colloc_mean)
    # knowns
    ux = up[:,0]*MAX_ux
    uy = up[:,1]*MAX_uy
    uxppuxpp = up[:,2]*MAX_uxppuxpp
    uxppuypp = up[:,3]*MAX_uxppuypp
    uyppuypp = up[:,4]*MAX_uyppuypp
    # unknowns
    p = up[:,5]*MAX_p
    
    # compute the gradients of the quantities
    
    # ux gradient
    dux = tf.gradients(ux, colloc_mean)[0]
    ux_x = dux[:,0]/MAX_x
    ux_y = dux[:,1]/MAX_y
    # and second derivative
    ux_xx = tf.gradients(ux_x, colloc_mean)[0][:,0]/MAX_x
    ux_yy = tf.gradients(ux_y, colloc_mean)[0][:,1]/MAX_y
    
    # uy gradient
    duy = tf.gradients(uy, colloc_mean)[0]
    uy_x = duy[:,0]/MAX_x
    uy_y = duy[:,1]/MAX_y
    # and second derivative
    uy_xx = tf.gradients(uy_x, colloc_mean)[0][:,0]/MAX_x
    uy_yy = tf.gradients(uy_y, colloc_mean)[0][:,1]/MAX_y

    # gradient unmodeled reynolds stresses
    uxppuxpp_x = tf.gradients(uxppuxpp, colloc_mean)[0][:,0]/MAX_x
    duxppuypp = tf.gradients(uxppuypp, colloc_mean)[0]
    uxppuypp_x = duxppuypp[:,0]/MAX_x
    uxppuypp_y = duxppuypp[:,1]/MAX_y
    uyppuypp_y = tf.gradients(uyppuypp, colloc_mean)[0][:,1]/MAX_y

    # pressure gradients
    dp = tf.gradients(p, colloc_tensor_f)[0]
    p_x = dp[:,0]/MAX_x
    p_y = dp[:,1]/MAX_y


    # governing equations
    f_x = (ux*ux_x + uy*ux_y) + (uxppuxpp_x + uxppuypp_y) + p_x - (nu_mol)*(ux_xx+ux_yy)  #+ uxux_x + uxuy_y    #- nu*(ur_rr+ux_rx + ur_r/r - ur/pow(r,2))
    f_y = (ux*uy_x + uy*uy_y) + (uxppuypp_x + uyppuypp_y) + p_y - (nu_mol)*(uy_xx+uy_yy)#+ uxuy_x + uyuy_y    #- nu*(ux_xx+ur_xr+ur_x/r)
    f_mass = ux_x + uy_y
    

    return f_x, f_y, f_mass


def mean_loss(y_true, y_pred):
    nonlocal colloc_mean
    # this needs to match the order that they are concatinated in the array when setting up the network
    # additionally, the known quantities must be first, unknown quantites second
    data_loss_ux = keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0]) # u 
    data_loss_uy = keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) # v 
    data_loss_uxppuxpp = keras.losses.mean_squared_error(y_true[:,2], y_pred[:,2]) # u''u''   
    data_loss_uxppuypp = keras.losses.mean_squared_error(y_true[:,3], y_pred[:,3]) # u''v''
    data_loss_uyppuypp = keras.losses.mean_squared_error(y_true[:,4], y_pred[:,4]) # v''v''

    mx,my,mass = net_f_mean_cartesian(colloc_mean)
    physical_loss1 = tf.reduce_mean(tf.square(mx))
    physical_loss2 = tf.reduce_mean(tf.square(my))
    physical_loss3 = tf.reduce_mean(tf.square(mass))
                      
    return data_loss_ux + data_loss_uy + data_loss_uxppuxpp + data_loss_uxppuypp +data_loss_uyppuypp + physics_loss_coefficient*(physical_loss1 + physical_loss2 + physical_loss3) # 0*f_boundary_p + f_boundary_t1+ f_boundary_t2 

