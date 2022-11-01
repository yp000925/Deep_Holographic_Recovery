import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from network import deep_decoder
import tensorflow as tf
from tensorflow.keras import layers as ls, activations as acts
import tensorflow_addons as tfa
from skimage.restoration import unwrap_phase
from fringe.utils.io import import_image, export_image
from fringe.utils.modifiers import ImageToArray, PreprocessHologram, ConvertToTensor
from fringe.process.gpu import AngularSpectrumSolver as AsSolver

hologram_path = 'Dataset/Cheek cells/hologram.tif'
background_path = 'Dataset/Cheek cells/background.tif'

p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = import_image(background_path, modifiers=p1)
p2 = PreprocessHologram(background=bg)
p3 = ConvertToTensor()
hologram = import_image(hologram_path, modifiers=[p1, p2, p3])
hologram_amp = tf.math.abs(hologram)


solver = AsSolver(shape=hologram_amp.shape,dx = 1.12,dy=1.12,wavelength=532e-3)
z = 238
rec = solver.solve(hologram,z)
amp = np.abs(rec)
dtype_f = tf.float32
num_epochs = 30000
lr = tf.Variable(0.01, dtype=dtype_f)
weight_decay = tf.Variable(0.002, dtype=dtype_f)
def get_lr():
    return lr.numpy()

def get_wd():
    return weight_decay.numpy()

input_t_ref = tf.random.normal([1, 16, 16, 256], mean=0, stddev=0.1, dtype=dtype_f)
input_t = tf.Variable(input_t_ref)
net = deep_decoder(input_shape=input_t[0].shape,
                   layers_channels=[256, 256, 256, 256, 256],
                   kernel_sizes=[1]*5,
                   out_channels=2,
                   upsample_mode='bilinear',
                   activation_func=ls.ReLU(),
                   out_activation=acts.sigmoid,
                   bn_affine=True)
optimizer = tfa.optimizers.AdamW(learning_rate=get_lr, weight_decay=get_wd)
mse = tf.keras.losses.MeanSquaredError()

net.summary()

logs_path = 'logs'
log_folder = 'exp1'

log_root = os.path.join(logs_path, log_folder)
if not os.path.exists(log_root):
    os.mkdir(log_root)

if not os.path.exists(os.path.join(log_root, 'exports')):
    os.mkdir(os.path.join(log_root, 'exports'))
checkpoint_folder = 'ckpts'

amp_coefs = [1.3, 1.4]
amp_coef = tf.Variable(amp_coefs[0], dtype=dtype_f)
amp_rand_std = tf.Variable(0.02, dtype=dtype_f)

checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 optimizer=optimizer,
                                 model=net,
                                 input_t=input_t,
                                 amp_coef=amp_coef,
                                 amp_rand_std=amp_rand_std,
                                 lr=lr,
                                 wd=weight_decay)
manager = tf.train.CheckpointManager(checkpoint, os.path.join(log_root, checkpoint_folder), max_to_keep=20)
checkpoint.restore(manager.latest_checkpoint)


out = net(input_t, training=False)
out = tf.squeeze(out)

out_ph = out[...,0]
out_ph = tf.scalar_mul(2 * np.pi, out_ph)
out_ph = tf.complex(real=tf.zeros_like(out_ph), imag=out_ph)
out_amp = out[...,1]
out_amp = tf.scalar_mul(amp_coef, out_amp)

out_amp = tf.complex(real=out_amp, imag=tf.zeros_like(out_amp))
out_func = tf.multiply(out_amp, tf.math.exp(out_ph))

out_hol = solver.solve(out_func, z)
out_hol_amp = tf.math.pow(tf.math.abs(out_hol), 2)

cmap = matplotlib.cm.get_cmap('viridis')
export_image(out[...,0].numpy(), path=os.path.join(log_root, 'exports', 'out_phase_{:d}.png'.format(int(checkpoint.step))),dtype='uint8')
export_image(cmap(out[...,0].numpy()), path=os.path.join(log_root, 'exports', 'out_phase_c_{:d}.png'.format(int(checkpoint.step))), dtype='uint8')
export_image(out[...,1].numpy(), path=os.path.join(log_root, 'exports', 'out_amp_{:d}.png'.format(int(checkpoint.step))), dtype='uint8')


fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(out_hol_amp.numpy(), cmap='gray')
ax.set_title("hologram output")
ax2 = fig.add_subplot(122)
ax2.imshow(hologram_amp.numpy(), cmap='gray')
ax2.set_title("hologram gt")
plt.show()