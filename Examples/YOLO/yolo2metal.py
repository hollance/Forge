# Conversion script for tiny-yolo-voc to Metal.
#
# The pretrained YOLOv2 model was made with the Darknet framework. You first
# need to convert it to a Keras model using YAD2K, and then yolo2metal.py can 
# convert the Keras model to Metal.
# 
# Required packages: python, numpy, h5py, pillow, tensorflow, keras.
#
# Download the tiny-yolo-voc.weights and tiny-yolo-voc.cfg files:
# wget https://pjreddie.com/media/files/tiny-yolo-voc.weights
# wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/tiny-yolo-voc.cfg
#
# Install YAD2K:
# https://github.com/allanzelener/YAD2K/
#
# Run the yad2k.py script to convert the Darknet model to Keras:
# ./yad2k.py -p tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.h5
#
# Edit the model_path variable to point to where tiny-yolo-voc.h5 was saved.
#
# Finally, run yolo2metal.py. It will convert the weights to Metal format
# and save them to the "Parameters" directory.

import os
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

model_path = "tiny-yolo-voc.h5"

# Load the model that was exported by YAD2K.
model = load_model(model_path)
# model.summary()

# The original model has batch normalization layers. We will now create
# a new model without batch norm. We will fold the parameters for each
# batch norm layer into the conv layer before it, so that we don't have
# to perform the batch normalization at inference time.
#
# All conv layers (except the last) have 3x3 kernel, stride 1, and "same"
# padding. Note that these conv layers did not have a bias in the original 
# model, but here they do get a bias (from the batch normalization).
#
# The last conv layer has a 1x1 kernel and identity activation.
#
# All max pool layers (except the last) have 2x2 kernel, stride 2, "valid" 
# padding. The last max pool layer has stride 1 and "same" padding.
#
# We still need to add the LeakyReLU activation as a separate layer, but 
# in Metal we can combine the LeakyReLU with the conv layer.
model_nobn = Sequential()
model_nobn.add(Conv2D(16, (3, 3), padding="same", input_shape=(416, 416, 3)))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(32, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(64, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(128, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(256, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(512, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D(strides=(1, 1), padding="same"))
model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(Conv2D(125, (1, 1), padding="same", activation='linear'))
#model_nobn.summary()

def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    conv_weights = conv_layer.get_weights()[0]

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]
    
    epsilon = 1e-3
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    return new_weights, new_bias

W_nobn = []
W_nobn.extend(fold_batch_norm(model.layers[1], model.layers[2]))
W_nobn.extend(fold_batch_norm(model.layers[5], model.layers[6]))
W_nobn.extend(fold_batch_norm(model.layers[9], model.layers[10]))
W_nobn.extend(fold_batch_norm(model.layers[13], model.layers[14]))
W_nobn.extend(fold_batch_norm(model.layers[17], model.layers[18]))
W_nobn.extend(fold_batch_norm(model.layers[21], model.layers[22]))
W_nobn.extend(fold_batch_norm(model.layers[25], model.layers[26]))
W_nobn.extend(fold_batch_norm(model.layers[28], model.layers[29]))
W_nobn.extend(model.layers[31].get_weights())
model_nobn.set_weights(W_nobn)

# Make a prediction using the original model and also using the model that
# has batch normalization removed, and check that the differences between
# the two predictions are small enough. They seem to be smaller than 1e-4,
# which is good enough for us, since we'll be using 16-bit floats anyway.

print("Comparing models...")

image_data = np.random.random((1, 416, 416, 3)).astype('float32')
features = model.predict(image_data)
features_nobn = model_nobn.predict(image_data)

max_error = 0
for i in range(features.shape[1]):
    for j in range(features.shape[2]):
        for k in range(features.shape[3]):
            diff = np.abs(features[0, i, j, k] - features_nobn[0, i, j, k])
            max_error = max(max_error, diff)
            if diff > 1e-4:
                print(i, j, k, ":", features[0, i, j, k], features_nobn[0, i, j, k], diff)

print("Largest error:", max_error)

# Convert the weights and biases to Metal format.

print("\nConverting parameters...")

dst_path = "Parameters"
W = model_nobn.get_weights()
for i, w in enumerate(W):
    j = i // 2 + 1
    print(w.shape)
    if i % 2 == 0:
        w.transpose(3, 0, 1, 2).tofile(os.path.join(dst_path, "conv%d_W.bin" % j))
    else:
        w.tofile(os.path.join(dst_path, "conv%d_b.bin" % j))

print("Done!")
