#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:05:57 2022

@author: saldern

Definition of additional function for the keras package

L-BFGS optimizer example

"""

import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp


def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            #loss_value = loss(model(train_x, training=True), train_y) # jvs
            loss_value = loss(train_y, model(train_x, training=True))

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

def function_factory_diff_evo(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        #with tf.GradientTape() as tape:
            # update the parameters in the model
        assign_new_model_parameters(params_1d)
            # calculate the loss
            #loss_value = loss(model(train_x, training=True), train_y) # jvs
        loss_value = loss(train_y, model(train_x, training=True))

        # calculate gradients and convert to 1D tf.Tensor
        #grads = tape.gradient(loss_value, model.trainable_variables)
        #grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f



def plot_helper(inputs, outputs, title, fname):
    """Plot helper"""
    pyplot.figure()
    pyplot.tricontourf(inputs[:, 0], inputs[:, 1], outputs.flatten(), 100)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)

# if __name__ == "__main__":

#     # use float64 by default
#     tf.keras.backend.set_floatx("float64")

#     # prepare training data
#     x_1d = np.linspace(-1., 1., 11)
#     x1, x2 = np.meshgrid(x_1d, x_1d)
#     inps = np.stack((x1.flatten(), x2.flatten()), 1)
#     outs = np.reshape(inps[:, 0]**2+inps[:, 1]**2, (x_1d.size**2, 1))

#     # prepare prediction model, loss function, and the function passed to L-BFGS solver
#     pred_model = tf.keras.Sequential(
#         [tf.keras.Input(shape=[2,]),
#          tf.keras.layers.Dense(64, "tanh"),
#          tf.keras.layers.Dense(64, "tanh"),
#          tf.keras.layers.Dense(1, None)])

#     loss_fun = tf.keras.losses.MeanSquaredError()
#     func = function_factory(pred_model, loss_fun, inps, outs)

#     # convert initial model parameters to a 1D tf.Tensor
#     init_params = tf.dynamic_stitch(func.idx, pred_model.trainable_variables)

#     # train the model with L-BFGS solver
#     results = tfp.optimizer.lbfgs_minimize(
#         value_and_gradients_function=func, initial_position=init_params, max_iterations=500)

#     # after training, the final optimized parameters are still in results.position
#     # so we have to manually put them back to the model
#     func.assign_new_model_parameters(results.position)

#     # do some prediction
#     pred_outs = pred_model.predict(inps)
#     err = np.abs(pred_outs-outs)
#     print("L2-error norm: {}".format(np.linalg.norm(err)/np.sqrt(11)))

#     # plot figures
#     plot_helper(inps, outs, "Exact solution", "ext_soln.png")
#     plot_helper(inps, pred_outs, "Predicted solution", "pred_soln.png")
#     plot_helper(inps, err, "Absolute error", "abs_err.png")
#     pyplot.show()

#     # print out history
#     print("\n"+"="*80)
#     print("History")
#     print("="*80)
#     print(*func.history, sep='\n')












