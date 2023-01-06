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

from skimage.filters import gaussian
from misc_functions import Scale

dtype_c  = tf.complex64
amp_path = 'Dataset/Simulation source images/baboon.png'
ph_path = 'Dataset/Simulation source images/peppers.png'

p1 = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
amplitude = import_image(amp_path, modifiers=p1)
phase = import_image(ph_path, modifiers=p1)

# Adjusting contrast
amplitude = Scale(amplitude, perc=1, max_val=1)

# Blurriness
sigma = 0 #np.exp(3)
amplitude = gaussian(amplitude, sigma, mode='reflect', truncate=np.round(10 * sigma) + 1)

phase /= np.max(phase)
phase *= 2 * np.pi - 0.2 * np.pi
phase -= np.pi

solver = AsSolver(shape=amplitude.shape, dx=1.12, dy=1.12, wavelength=532e-3)
z = 300
obj_func = tf.convert_to_tensor(amplitude * np.exp(1j * phase), dtype_c)
hologram = solver.solve(obj_func, z)
hologram_amp = tf.math.pow(tf.math.abs(hologram), 2)


# plt.figure()
# plt.imshow(amplitude, cmap='gray', vmin=0, vmax=1)
# plt.show()
# plt.figure()
# plt.imshow(phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
# plt.show()
# plt.figure()
# plt.imshow(hologram_amp.numpy(), cmap='gray', vmin=0, vmax=1)
# plt.show()
# plt.figure()
# plt.hist((hologram_amp.numpy()).flatten(), 256)
# plt.show()
