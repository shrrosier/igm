#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
import os

from igm.processes.iceflow.utils import fieldin_to_X, Y_to_UV, update_2d_iceflow_variables, compute_PAD, print_info
from igm.processes.iceflow.energy_iceflow.energy_iceflow import iceflow_energy_XY
from igm.processes.iceflow.emulate.neural_network import *
from igm.processes.iceflow.emulate import emulators
import importlib_resources 
import igm  
import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.append('/home/srosier/work/tgregov')
from Optimizer_NN import Optimizer_NN
from Optimizer_NN_LBFGS import Optimizer_NN_LBFGS

def initialize_iceflow_emulator(cfg, state):

    if (cfg.processes.iceflow.emulator.optimizer == "Adam"):

        if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
            state.opti_retrain = getattr(tf.keras.optimizers,cfg.processes.iceflow.emulator.optimizer)(
                learning_rate=cfg.processes.iceflow.emulator.lr,
                epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
                clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm
            )
        else:
            state.opti_retrain = getattr(tf.keras.optimizers.legacy,cfg.processes.iceflow.emulator.optimizer)( 
                learning_rate=cfg.processes.iceflow.emulator.lr,
                epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
                clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm
            )

    direct_name = (
        "pinnbp"
        + "_"
        + str(cfg.processes.iceflow.numerics.Nz)
        + "_"
        + str(int(cfg.processes.iceflow.numerics.vert_spacing))
        + "_"
    )
    direct_name += (
        cfg.processes.iceflow.emulator.network.architecture
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_layers)
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_out_filter)
        + "_"
    )
    direct_name += (
        str(cfg.processes.iceflow.physics.dim_arrhenius)
        + "_"
        + str(int(cfg.processes.iceflow.physics.new_friction_param))
    )

    if cfg.processes.iceflow.emulator.pretrained:
        if cfg.processes.iceflow.emulator.name == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print(
                    "Found pretrained emulator in the igm package: " + direct_name
                )
            else:
                print("No pretrained emulator found in the igm package")
        else:
            dirpath = os.path.join(state.original_cwd, cfg.processes.iceflow.emulator.name)
            if os.path.exists(dirpath):
                print("----------------------------------> Found pretrained emulator: " + cfg.processes.iceflow.emulator.name)
            else:
                print("----------------------------------> No pretrained emulator found ")

        fieldin = []
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg.processes.iceflow.emulator.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile() 
    else:
        print("----------------------------------> No pretrained emulator, start from scratch.") 
        nb_inputs = len(cfg.processes.iceflow.emulator.fieldin) + (cfg.processes.iceflow.physics.dim_arrhenius == 3) * (
            cfg.processes.iceflow.numerics.Nz - 1
        )
        nb_outputs = 2 * cfg.processes.iceflow.numerics.Nz
        state.iceflow_model = getattr(igm.processes.iceflow.emulate.emulate, cfg.processes.iceflow.emulator.network.architecture)(
            cfg, nb_inputs, nb_outputs
        )

    print(state.iceflow_model.summary())

    # direct_name = 'pinnbp_10_4_cnn_16_32_2_1'        
    # dirpath = importlib_resources.files(emulators).joinpath(direct_name)
    # iceflow_model_pretrained = tf.keras.models.load_model(
    #     os.path.join(dirpath, "model.h5"), compile=False
    # )
    # N=16
    # pretrained_weights = [layer.get_weights() for layer in iceflow_model_pretrained.layers[:N]]
    # for i in range(N):
    #     state.iceflow_model.layers[i].set_weights(pretrained_weights[i])

def update_iceflow_emulated(cfg, state):
    # Define the input of the NN, include scaling

    Ny, Nx = state.thk.shape
    N = cfg.processes.iceflow.numerics.Nz

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]

    X = fieldin_to_X(cfg, fieldin)

    if cfg.processes.iceflow.emulator.exclude_borders>0:
        iz = cfg.processes.iceflow.emulator.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        
    if cfg.processes.iceflow.emulator.network.multiple_window_size==0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if cfg.processes.iceflow.emulator.exclude_borders>0:
        iz = cfg.processes.iceflow.emulator.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(cfg, Y)
    U = U[0]
    V = V[0]

    state.U = tf.where(state.thk > 0, U, 0)
    state.V = tf.where(state.thk > 0, V, 0)

    # If requested, the speeds are artifically upper-bounded
    if cfg.processes.iceflow.force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        state.V = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.V / velbar_mag),
                state.V,
            ) 

    update_2d_iceflow_variables(cfg, state)

def update_iceflow_emulator(cfg, state, it, pertubate=False):

    if cfg.processes.iceflow.emulator.optimizer == "LBFGS":
        update_iceflow_emulator_LBFGS(cfg, state, it, pertubate)
    elif cfg.processes.iceflow.emulator.optimizer == "Adam":
        update_iceflow_emulator_ADAM(cfg, state, it, pertubate)
    else:
        raise ValueError("Unknown optimizer: {}".format(cfg.processes.iceflow.emulator.optimizer))


def update_iceflow_emulator_LBFGS(cfg, state, it, pertubate=False):

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]

    XX = fieldin_to_X(cfg, fieldin) 

    XXX = pertubate_SR(cfg,XX)

    patches = split_into_patches_with_overlap(XXX, cfg.processes.iceflow.emulator.framesizemax, overlap=0.25)
        
    Ny = patches.shape[-3]
    Nx = patches.shape[-2]

    # combine perturbation and patch axes into single batch axis at index 0
    X = tf.reshape(patches, (-1, patches.shape[2], patches.shape[3], patches.shape[4]))
    
    # PAD = compute_PAD(cfg,Nx,Ny)

    # Xin = tf.pad(X[0, :, :, :, :], PAD, "CONSTANT")


    cost_fn = lambda Y: calculate_cost(cfg, X, Y, Nx, Ny)

    # Y = state.iceflow_model(X)  # compute the output of the NN
    # cost = cost_fn(Y)
    # print(cost.numpy())

    optimizer = Optimizer_NN_LBFGS(
        cost_fn, 
        state.iceflow_model, 
        X, 
        scale     = 1, 
        iter_max  = 100000, 
        tol       = 1e-20,
        time_max  = 500, 
        alpha_min = 1e-20,
    )

    w,optim = optimizer.minimize()

    times = optim.times
    costs = optim.costs
    grads = optim.grads_norm

    # save the costs, gradients and times
    if len(cfg.processes.iceflow.emulator.save_cost)>0:
        np.savetxt('LBFGS-'+str(it)+'.dat',
                   np.array(list(zip(costs, grads, times))), fmt="%5.10f")

    

def calculate_cost(cfg, X, Y, Nx, Ny):

    C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(cfg, X, Y[:,:Ny,:Nx,:])
 
    C_shear_cost = tf.reduce_mean(C_shear)
    C_slid_cost  = tf.reduce_mean(C_slid)
    C_grav_cost  = tf.reduce_mean(C_grav)
    C_float_cost = tf.reduce_mean(C_float)

    COST = C_shear_cost + C_slid_cost + C_grav_cost + C_float_cost

    return COST

def update_iceflow_emulator_ADAM(cfg, state, it, pertubate=False):
 
    run_it = False
    if cfg.processes.iceflow.emulator.retrain_freq > 0:
       run_it = (it % cfg.processes.iceflow.emulator.retrain_freq == 0)
 
    warm_up = int(it <= cfg.processes.iceflow.emulator.warm_up_it)

    if (warm_up | run_it):

        state.COST_EMULATOR = []
        state.GRAD_EMULATOR = []
        state.OPTIMIZER_TIMES = []

        # initialize time t0
        t0 = tf.timestamp()
     
        fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]

        XX = fieldin_to_X(cfg, fieldin) 

        if pertubate:
            XX = pertubate_X(cfg, XX)  

        X = split_into_patches(XX, cfg.processes.iceflow.emulator.framesizemax,
                                   cfg.processes.iceflow.emulator.split_patch_method)
 
        Ny = X.shape[-3]
        Nx = X.shape[-2]
        
        PAD = compute_PAD(cfg,Nx,Ny)

        if warm_up:
            nbit = cfg.processes.iceflow.emulator.nbit_init
            lr = cfg.processes.iceflow.emulator.lr_init
        else:
            nbit = cfg.processes.iceflow.emulator.nbit
            lr = cfg.processes.iceflow.emulator.lr

        state.opti_retrain.lr = lr 

        iz = cfg.processes.iceflow.emulator.exclude_borders 

        if cfg.processes.iceflow.emulator.plot_sol:
            plt.ion()  # enable interactive mode
            state.fig = plt.figure(dpi=200)
            state.ax = state.fig.add_subplot(1, 1, 1)
            state.ax.axis("off")
            state.ax.set_aspect("equal")

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape() as t:

                    if cfg.processes.iceflow.emulator.lr_decay < 1:
                        state.opti_retrain.lr = lr * (cfg.processes.iceflow.emulator.lr_decay ** (i / 1000))

                    Y = state.iceflow_model(tf.pad(X[i, :, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
                    
                    if iz>0:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(cfg, X[i, :, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
                    else:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(cfg, X[i, :, :, :, :], Y[:, :, :, :])
 
                    C_shear_cost = tf.reduce_mean(C_shear)
                    C_slid_cost  = tf.reduce_mean(C_slid)
                    C_grav_cost  = tf.reduce_mean(C_grav)
                    C_float_cost = tf.reduce_mean(C_float)

                    COST = C_shear_cost + C_slid_cost + C_grav_cost + C_float_cost

                    cost_emulator = cost_emulator + COST

                    U, V = Y_to_UV(cfg, Y) ; velsurf_mag = tf.sqrt(U[0][-1] ** 2 + V[0][-1] ** 2)

                    if warm_up:
                        print_info(state, epoch, C_shear_cost.numpy(), C_slid_cost.numpy(), \
                                          C_grav_cost.numpy(), COST.numpy(), tf.reduce_max(velsurf_mag).numpy())

                    if (epoch + 1) % 100 == 0:
                         
                        if cfg.processes.iceflow.emulator.plot_sol:
                            im = state.ax.imshow(
                                np.where(state.thk > 0, velsurf_mag, np.nan),
                                origin="lower",
                                cmap="turbo",
                                norm=matplotlib.colors.LogNorm(vmin=1,vmax=300)
                            )
                            if not hasattr(state, "already_set_cbar"):
                                state.cbar = plt.colorbar(im, label='velocity')
                                state.already_set_cbar = True
                            state.fig.canvas.draw()  # re-drawing the figure
                            state.fig.canvas.flush_events()  # to flush the GUI events
                            state.ax.set_title("epoch : " + str(epoch), size=15)


                grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                # if (epoch + 1) % 100 == 0:
                #     values = [tf.norm(g) for g in grads]
                #     normalized = values / tf.reduce_sum(values) 
                #     percentages = [100 * v.numpy() for v in normalized] 
                #     print("Percentages:", " | ".join(f"{p:.1f}%" for p in percentages[::2]))

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

                grad_emulator = tf.linalg.global_norm(grads)
 
            state.COST_EMULATOR.append(cost_emulator)
            state.GRAD_EMULATOR.append(grad_emulator)
            state.OPTIMIZER_TIMES.append(tf.timestamp() - t0)

    
        if len(cfg.processes.iceflow.emulator.save_cost)>0:
            np.savetxt(cfg.processes.iceflow.emulator.save_cost+'-'+str(it)+'.dat',
                    np.array(list(zip(state.COST_EMULATOR,state.GRAD_EMULATOR,state.OPTIMIZER_TIMES))), fmt="%5.10f")

def split_into_patches(X, nbmax, split_patch_method):
    """
    This function splits the input tensor into patches of size nbmax x nbmax.
    The patches are then stacked together to form a new tensor.
    If stack along axis 0, the adata will be streammed in a sequential way
    If stack along axis 1, the adata will be streammed in a parallel way by baches
    """
    XX = []
    ny = X.shape[1]
    nx = X.shape[2]
    sy = ny // nbmax + 1
    sx = nx // nbmax + 1
    ly = int(ny / sy)
    lx = int(nx / sx)

    for i in range(sx):
        for j in range(sy):
#            if tf.reduce_max(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :]) > 0:
            XX.append(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    if split_patch_method == "sequential":
        XXX = tf.stack(XX, axis=0)
    elif split_patch_method == "parrallel":
        XXX = tf.expand_dims(tf.concat(XX, axis=0), axis=0)

    return XXX

def pertubate_X(cfg, X):

    XX = [X]

    for i,f in enumerate(cfg.processes.iceflow.emulator.fieldin):

        vec = [tf.ones_like(X[:,:,:,i])*(i==j) for j in range(X.shape[3])]
        vec = tf.stack(vec, axis=-1)
 
        if hasattr(cfg.processes, "data_assimilation"):
            if f in cfg.processes.data_assimilation.control_list:
                XX.append(X + X*vec*0.2)
                XX.append(X - X*vec*0.2)
        else:
            if f in ["thk","usurf"]: 
                XX.append(X + X*vec*0.2)
                XX.append(X - X*vec*0.2)
 
    return tf.concat(XX, axis=0)


def save_iceflow_model(cfg, state):
    directory = "iceflow-model"
    
    import shutil

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(cfg.processes.iceflow.physics.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(cfg.processes.iceflow.emulator.fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in cfg.processes.iceflow.emulator.fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (cfg.processes.iceflow.numerics.Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (cfg.processes.iceflow.numerics.vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()

def split_into_patches_with_overlap(X, nbmax, overlap=0.25):
    """
    Split the input tensor into patches of size nbmax x nbmax, with at least the specified minimum overlap.
    The stride is chosen so that all patches are nbmax x nbmax and the entire input is covered,
    and the difference is split evenly across the input (so the actual overlap may be slightly larger).
    Args:
        X: Input tensor of shape (batch_size, height, width, channels).
        nbmax: Patch size (height and width).
        overlap: Minimum fractional overlap between patches (e.g., 0.25 for 25% overlap).
    Returns:
        A tensor containing the patches.
    """
    print(nbmax)
    XX = []
    ny, nx = X.shape[1], X.shape[2]
    if nbmax > nx and nbmax > ny:
        return tf.expand_dims(X, axis=0)

    # Calculate the number of steps needed to cover the input with the minimum overlap
    min_stride = int(nbmax * (1 - overlap))
    n_patches_y = int(np.ceil((ny - nbmax) / min_stride)) + 1
    n_patches_x = int(np.ceil((nx - nbmax) / min_stride)) + 1

    # Now recalculate the stride so that the last patch lands exactly at the end
    if n_patches_y > 1:
        stride_y = (ny - nbmax) / (n_patches_y - 1)
    else:
        stride_y = 0
    if n_patches_x > 1:
        stride_x = (nx - nbmax) / (n_patches_x - 1)
    else:
        stride_x = 0

    # Generate patch start indices
    y_starts = [int(round(i * stride_y)) for i in range(n_patches_y)]
    x_starts = [int(round(i * stride_x)) for i in range(n_patches_x)]

    for i in y_starts:
        for j in x_starts:
            XX.append(X[:, i:i + nbmax, j:j + nbmax, :])

    return tf.stack(XX, axis=0)

def pertubate_SR(cfg, X):
    """
    Expand X along the batch dimension by adding Perlin-noise-perturbed copies
    for each channel except 'dX'.
    """
    Ny, Nx = X.shape[1:3]
    scale = cfg.processes.iceflow.emulator.perturbation_scale

    # Find the smallest power-of-two dimensions larger than the field dimensions
    largest_dim = max(Ny, Nx)
    smallest_squared = 2 ** (int(np.log2(largest_dim)) + 1)

    XX = [X]  # Start with the original

    for _ in range(cfg.processes.iceflow.emulator.num_perturbations - 1):
        # Start with a copy of X
        perturbed_X = tf.identity(X)
        noise_channels = []

        for i, f in enumerate(cfg.processes.iceflow.emulator.fieldin):
            if f == "dX":
                # No noise for 'dX'
                noise = tf.zeros((1, Ny, Nx, 1), dtype=X.dtype)
            else:
                # Generate Perlin noise for this channel
                noise_np = generate_perlin_noise_2d(
                    shape=(smallest_squared, smallest_squared),
                    res=(4, 4),
                    tileable=(False, False)
                )
                noise_np = noise_np[:Ny, :Nx]
                noise = tf.convert_to_tensor(noise_np, dtype=X.dtype)
                noise = tf.expand_dims(noise, axis=0)   # batch
                noise = tf.expand_dims(noise, axis=-1)  # channel
            noise_channels.append(noise)

        # Stack all noise channels to shape (1, Ny, Nx, num_fields)
        noise_tensor = tf.concat(noise_channels, axis=-1)
        # Apply noise to all channels at once
        perturbed_X = perturbed_X + perturbed_X * noise_tensor * scale
        XX.append(perturbed_X)

    # Concatenate along batch dimension
    return tf.concat(XX, axis=0)

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)