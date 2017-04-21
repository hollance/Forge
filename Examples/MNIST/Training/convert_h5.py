# Converts a trained Keras model from an HDF5 file to Metal format.
#
# Keras stores the weights for each layer in this shape:
#    (kernelHeight, kernelWidth, inputChannels, outputChannels)
#
# Metal expects weights in the following shape:
#    (outputChannels, kernelHeight, kernelWidth, inputChannels)

import os
import sys
import numpy as np
import h5py

args = sys.argv[1:]
if len(args) != 2:
	print("Usage: %s model.h5 output-folder" % os.path.basename(__file__))
	exit(-1)
data_path, dst_path = args

f = h5py.File(data_path, "r")

#f.visititems(lambda name, obj: print(name, obj))

conv_weights = { "model_weights/conv2d_1/conv2d_1/kernel:0": "conv1_W",
                 "model_weights/conv2d_2/conv2d_2/kernel:0": "conv2_W" }

conv_biases = { "model_weights/conv2d_1/conv2d_1/bias:0": "conv1_b",
                "model_weights/conv2d_2/conv2d_2/bias:0": "conv2_b" }

fc_weights = { "model_weights/dense_1/dense_1/kernel:0": "fc1_W",
               "model_weights/dense_2/dense_2/kernel:0": "fc2_W" }

fc_shapes = { "fc1_W": (7, 7, 50, 320),
              "fc2_W": (1, 1, 320, 10) }

fc_biases = { "model_weights/dense_1/dense_1/bias:0": "fc1_b",
              "model_weights/dense_2/dense_2/bias:0": "fc2_b" }

out_dict = {}

print("Convolutional layer weights:")
for k, v in conv_weights.items():
	data = f[k].value
	k_h = data.shape[0]
	k_w = data.shape[1]
	c_i = data.shape[2]
	c_o = data.shape[3]
	print("%s: %d x %d x %d x %d" % (v, k_h, k_w, c_i, c_o))
	out_dict[v] = data.transpose(3, 0, 1, 2)

print("Convolutional layer biases:")
for k, v in conv_biases.items():
	data = f[k].value
	print(v, data.shape)
	out_dict[v] = data

print("Fully-connected layer weights:")
for k, v in fc_weights.items():
	data = f[k].value
	c_i  = data.shape[0]
	c_o  = data.shape[1]
	print("%s: %d x %d" % (v, c_i, c_o))
	out_dict[v] = data.reshape(fc_shapes[v]).transpose(3, 0, 1, 2).reshape(c_i, c_o)
		
print("Fully-connected layer biases:")
for k, v in fc_biases.items():
	data = f[k].value
	print(v, data.shape)
	out_dict[v] = data

for k, v in out_dict.items():
	v.tofile(os.path.join(dst_path, k + ".bin"))
