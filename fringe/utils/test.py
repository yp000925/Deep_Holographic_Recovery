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

device = 'gpu'

if device == "gpu":
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print('GPU is up and running')
        device = "/gpu:0"
    else:
        print('No GPUs found. The process will run on CPU.')
        device = "/cpu:0"
elif device == "tpu":
    if len(tf.config.experimental.list_physical_devices('TPU')) > 0:
        print('TPU is up and running')
        device = "/tpu:0"
    else:
        print('No TPUs found. The process will run on CPU.')
        device = "/cpu:0"
else:
    device = "/cpu:0"

dtype_f = tf.float32
dtype_c = tf.complex64


hologram_path = 'Dataset/Cheek cells/hologram.tif'
background_path = 'Dataset/Cheek cells/background.tif'

p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = import_image(background_path, preprocessor=p1)
p2 = PreprocessHologram(background=bg)
p3 = ConvertToTensor(dtype=dtype_c)
hologram = import_image(hologram_path, preprocessor=[p1, p2, p3])
hologram_amp = tf.math.abs(hologram)

solver = AsSolver(shape=hologram_amp.shape, dx=1.12, dy=1.12, wavelength=532e-3)
z = 238
rec = solver.solve(hologram, z)
amp = np.abs(rec)
#phase = unwrap_phase(np.angle(rec))

plt.imshow(hologram_amp.numpy(), cmap='gray')
plt.show()
plt.imshow(amp, cmap='gray')
plt.show()
plt.hist((hologram_amp.numpy()).flatten(), 256)
plt.show()