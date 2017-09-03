# Importing a model from Keras, TensorFlow, Caffe, or other tools

Chances are that you trained your model using one of the popular deep learning packages. To run this model on iOS using MPSCNN, you need to be aware of the following.

If the iOS version of your model does not appear to work correctly, then double check these items. It's easy to make a small mistake somewhere and then everything falls to pieces.

## Make sure the same settings are used

Use `print(model.summary())` to print a description of your model to the Xcode debug pane. Verify that the sizes of the tensors correspond to the original model.

If not, then double check that you used the same settings to create your layers, notably padding.

#### Padding

Some tools use `valid` and `same` padding.

- `valid` corresponds to `padding: false` in Forge. The layer does not apply zero-padding and will shrink the output image.

- `same` corresponds to `padding: true` in Forge. With this setting, the output width and height are the same as the input width and height.

In Keras the default padding for convolutional layers and pooling layers is `valid`. In Forge, however, the default padding is `true` (i.e. `same`) for convolutional layers (but `false` for pooling layers).

## You need to convert the weights

For MPSCNN, the weights must be arranged like this in memory:

```
[outputChannels][kernelHeight][kernelWidth][inputChannels]
```

For a depthwise convolution layer the memory layout is as follows:

```
[channels][channelMultiplier][kernelHeight][kernelWidth]
```

Most training packages will use a different format, so you'll need to convert the weights by transposing some of their dimensions.

#### TensorFlow / Keras

TensorFlow stores the weights for each layer in this order:

```
[kernelHeight][kernelWidth][inputChannels][outputChannels]
```

For a convolutional later, you will need to transpose the weights as follows (using numpy):

```python
metal_weights = trained_weights.transpose(3, 0, 1, 2)
```

For a fully-connected layer, it works best if you do this:

```python
channels_in = 7*7*50
channels_out = 320
fc_shape = (7, 7, 50, 320)

metal_weights = trained_weights.reshape(fc_shape)
                               .transpose(3, 0, 1, 2)
                               .reshape(channels_in, channels_out)
```

In this example the fully-connected layer has 320 neurons. It is connected to a convolutional layer (or pooling layer) that outputs a 7x7-pixel image with 50 channels. You first need to reshape the weights array to make it 4-dimensional, then transpose it, and finally reshape the array back to 2 dimensions.

For a practical example, [see the conversion script in the MNIST Training folder](../Examples/MNIST/Training/convert_h5.py).

#### Caffe

Caffe stores the weights for each layer in this order:

```
[outputChannels][inputChannels][kernelHeight][kernelWidth]
```

For a convolutional later, you will need to transpose the weights as follows (using numpy):

```python
metal_weights = trained_weights.transpose(0, 2, 3, 1)
```

For a fully-connected layer, it works best if you do this:

```python
channels_in = 7*7*50
channels_out = 320
fc_shape = (320, 50, 7, 7)

metal_weights = trained_weights.reshape(fc_shape)
                               .transpose(0, 2, 3, 1)
                               .reshape(channels_out, channels_in)
```

In this example the fully-connected layer has 320 neurons. It is connected to a convolutional layer (or pooling layer) that outputs a 7x7-pixel image with 50 channels. You first need to reshape the weights array to make it 4-dimensional, then transpose it, and finally reshape the array back to 2 dimensions.

For a practical example, [see the conversion script in the VGGNet-Metal project](https://github.com/hollance/VGGNet-Metal/blob/master/convert/convert_vggnet.py).

## Don't forget to do any preprocessing

Forge uses the `.float16` channel format. No matter what the pixel format of your input texture is (usually 8-bit unorm BGRA), after the `Resize` layer all image data will be 16-bit floats, RGB order, pixel values between 0.0 and 1.0.

If your model expects the data in a different format or order, you may have to write a custom compute kernel to do preprocessing.

For example:

- Models trained with Caffe expect image data to be in BGR order instead of RGB. 
- A model trained on ImageNet expects you to subtract the mean RGB values from each pixel. 
- You may need to divide -- or multiply -- the pixel values by 255.
- And so on...

This totally depends on how your model was trained and which tool you used to train it.

## Make sure the output from the two models is the same

In Forge you can look at the output of the neural network using the following line of code:

```
let image = model.outputImage(inflightIndex: inflightIndex)
let probabilities = image.toFloatArray()
```

This gets the `MPSImage` for the output tensor and converts it to an array of 32-bit floating point numbers.

To make sure the neural network functions properly, you should compare this to the output of the original model in Python (or whatever you used) *for the exact same input*.

Ideally, you'd use a number of different test images from different classes. If the Metal version of the model gives the same outputs as the Python model for a handful of test images, the conversion probably went OK.

#### Use the exact same input file for making comparisons

Always use the exact same input image file to compute the output form the original model and the Metal model. Obviously, if you're using different inputs, you can expect different outputs. Less obvious is that the input to the model may be different even though you think you're using the same image file.

- If your network includes a `Resize` layer to scale down the input, and in Python you do something like `skimage.transform.resize()`, then you are **NOT** using the same input!!! These are two different ways of scaling down the image and they will not produce the same data. If you resize all images to 224x224, for example, then use a 224x224 image for testing so that no resizing actually happens. (Or temporarily remove the resizing layer.)

- To make 100% sure that you're using the exact same input for both models, print out the first 20 or so pixel values in Python and in the iOS app. They should correspond exactly. With Forge you can set a tensor to `imageIsTemporary = false` and then you can get its `MPSImage` with `model.image(for: tensor)` and use `toFloatArray()` to see what is inside the image.
 
#### 16-bit floats are not very precise

MPSCNN uses 16-bit floats for its calculations. You can expect to see errors in the `0.001` range. If the largest difference between the Metal output and the original model is no larger than `0.001` but all errors are `1e-3` or smaller, then you're probably OK.

#### Compare the output layer-by-layer

If you have determined that the input is exactly the same for both models, but the output of the network is different, then the question is: where in the model does it go wrong? 

A divide-and-conquer strategy is useful here. I suggest you first check that the output of the last convolutional/pooling layer before any fully-connected layers is correct. If it is correct up to that point, then something probably went wrong with the weights of the fully-connected layers.

To look at the output of a layer, you must keep a reference to the tensor that holds this output and set `tensor.imageIsTemporary = false`. After the forward pass is complete, do the following:

```swift
let outputImage = model.image(for: tensor, inflightIndex: inflightIndex)
printChannelsForPixel(x: 0, y: 0, image: outputImage)
```

The `printChannelsForPixel()` function takes an `MPSImage` and prints out the values from the feature maps for a single pixel from the image. You need to compare this to the output of this same pixel in the same layer in the original model, for the exact same input image -- how to do that depends on the training tool you used. 

Writing `printChannelsForPixel(x: 10, y: 10, ...)` is the same as doing `print(layer_output[0, 10, 10, :])` in Python with layer output from Keras.

The reason I suggest using `printChannelsForPixel()`, rather than dumping the entire image with `outputImage.toFloatArray()` is that `MPSImage` stores its data in a very confusing manner. It's easier to look at a few select pixels in the image rather than at the whole thing.

#### MPSImage is weird

If you're looking directly at the data inside an `MPSImage` using `someImage.toFloatArray()`, then make sure you understand what you're looking at!

Because Metal is a graphics API, `MPSImage` stores the data in `MTLTexture` objects. Each pixel from the texture stores 4 channels: R contains the first channel, G is the second channel, B is the third, A is the fourth. 

If there are > 4 channels in the MPSImage, then the channels are organized in the output as follows:

```
[ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,7,8,5,6,7,8,...,5,6,7,8 ]
```
  
and not as you'd expect:

```
[ 1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8 ]
```

First are channels 1 - 4 for the entire image, followed by channels 5 - 8 for the entire image, and so on. That happens because we copy the data out of the texture by slice, and we can't interleave slices.

If the number of channels is not a multiple of 4, then the output will have padding bytes in it:

```
[ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,-,-,5,6,-,-,...,5,6,-,- ]
```

The size of the array is therefore always a multiple of 4! So if you have a classifier for 10 classes, the output vector is 12 elements and the last two elements are zero.

The only case where you get the kind of array you'd actually expect is when the number of channels is 1, 2, or 4 (i.e. there is only one slice):

```
[ 1,1,1,...,1 ] or [ 1,2,1,2,1,2,...,1,2 ] or [ 1,2,3,4,...,1,2,3,4 ]
```

Since most intermediate layers in a neural network produce images with more than 4 channels, you can't just print out `image.toFloatArray()` and compare it to the output from your original model. The order of the channels will be all mixed up.
