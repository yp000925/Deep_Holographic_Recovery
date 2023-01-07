import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
import PIL.Image as Image
dtype_c  = tf.complex64
dtype_f = tf.float32
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

hologram_path = 'Dataset/air_2/hologram.png'
background_path = 'Dataset/air_2/bg.png'

Nx = 512
Ny = 512
z = 16000
wavelength = 532e-3
deltaX = 3.45*2*2
deltaY = 3.45*2*2
# p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=[600,244,512,512], dtype='float32')
# bg = import_image(background_path, modifiers=p1)
# p2 = PreprocessHologram(background=bg)
# p3 = ConvertToTensor()
# hologram = import_image(hologram_path, modifiers=[p1, p2, p3])
# hologram_amp = tf.math.abs(hologram)
hologram = Image.open(hologram_path)
hologram = hologram.crop((0,0,1024,1024))
hologram = hologram.resize((512,512))
bg = Image.open(background_path)
bg = bg.crop((0,0,1024,1024))
bg = bg.resize((512,512))
hologram = np.array(hologram)
bg = np.array(bg)
img = hologram/bg
minh = np.min(img)
img -= minh
img /= 1 - minh
hologram =  tf.convert_to_tensor(img)
hologram_amp = tf.math.abs(hologram)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(hologram_amp.numpy(), cmap='gray')
ax.set_title('hologram')
plt.show()
solver = AsSolver(shape=hologram_amp.shape,dx = deltaX,dy=deltaY,wavelength=wavelength)
bu = tf.cast(hologram, dtype_c)
rec = solver.solve(bu,z)
amp = np.abs(rec)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(amp, cmap='gray')
ax.set_title("reconstruction amplitude")
plt.show()
plt.hist((hologram_amp.numpy()).flatten(), 256)
plt.show()
rec = solver.solve(hologram.numpy(),z)
amp = np.abs(rec)
fig = plt.figure()