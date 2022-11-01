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
dtype_c  = tf.complex64
dtype_f = tf.float32
hologram_path = 'Dataset/air_2/hologram.png'
background_path = 'Dataset/air_2/bg.png'

p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = import_image(background_path, modifiers=p1)
p2 = PreprocessHologram(background=bg)
p3 = ConvertToTensor()
hologram = import_image(hologram_path, modifiers=[p1, p2, p3])
hologram_amp = tf.math.abs(hologram)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(hologram_amp.numpy(), cmap='gray')
ax.set_title('hologram')

solver = AsSolver(shape=hologram_amp.shape,dx = 3.45,dy=3.45,wavelength=532e-3)
# z = 238
z = 17526
rec = solver.solve(hologram,z)
amp = np.abs(rec)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(amp, cmap='gray')
ax.set_title("reconstruction amplitude")
plt.show()