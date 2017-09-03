# Forge: a neural network toolkit for Metal

**Forge** is a collection of helper code that makes it a little easier to construct deep neural networks using Apple's MPSCNN framework.

[Read the blog post](http://machinethink.net/blog/forge-neural-network-toolkit-for-metal/)

![Geordi likes it!](Geordi.png)

## What does this do?

Features of Forge:

**Conversion functions.** MPSCNN uses `MPSImage`s and `MTLTexture`s for everything, often using 16-bit floats. But you probably want to work with Swift `[Float]` arrays. Forge's conversion functions make it easy to work with Metal images and textures.

**Easy layer creation.** Reduce the boilerplate when building the layers for your neural network. Forge's domain-specific language makes defining a neural net as simple as:

```swift
let input = Input()

let output = input
        --> Resize(width: 28, height: 28)
        --> Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (5, 5), channels: 50, activation: relu, name: "conv2")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Dense(neurons: 320, activation: relu, name: "fc1")
        --> Dense(neurons: 10, name: "fc2")
        --> Softmax()

let model = Model(input: input, output: output)
```

**Custom layers.** MPSCNN only supports a limited number of layers, so we've added a few of our own:

- Depth-wise convolution
- Transpose channels
- Deconvolution (coming soon!)

**Preprocessing kernels.** Often you need to preprocess data before it goes into the neural network. Forge comes with a few handy kernels for this:

- SubtractMeanColor
- RGB2Gray
- RGB2BGR

**Custom compute kernels.** Many neural networks require custom compute kernels, so Forge provides helpers that make it easy to write and launch your own kernels.

**Debugging tools.** When you implement a neural network in Metal you want to make sure it actually computes the correct thing. Due to the way Metal encodes the data, inspecting the contents of the `MTLTexture` objects is not always straightforward. Forge can help with this.

**Example projects.** Forge comes with a number of pretrained neural networks, such as LeNet-5 on MNIST, Inception3 on ImageNet, and MobileNets.

> **Note:** A lot of the code in this library is still *experimental* and subject to change. Use at your own risk!

## iOS 10 and iOS 11 compatibility

Forge supports both iOS 10 and iOS 11.

Forge must be compiled with **Xcode 9** and the iOS 11 SDK. (An older version is available under the tag `xcode8`, but is no longer supported.)

**Important changes:**

The order of the weights for `DepthwiseConvolution` layers has changed. It used to be:

	[kernelHeight][kernelWidth][channels]
	
now it is:

	[channels][kernelHeight][kernelWidth]

This was done to make this layer compatible with MPS's new depthwise convolution. On iOS 10, Forge will use its own `DepthwiseConvolutionKernel`, on iOS 11 and later is uses the MPS version (`MPSCNNDepthWiseConvolutionDescriptor`).

Note: Forge does not yet take advantage of all the MPS improvements in iOS 11, such as the ability to load batch normalization parameters or loading weights via data sources. This functionality will be added in a future version.

## Run the examples!

To see a demo of Forge in action, open **Forge.xcworkspace** in Xcode and run one of the example apps on your device.

You need at least Xcode 9 and a device with an A8 processor (iPhone 6 or better) running iOS 10 or later. You cannot build for the simulator as it does not support Metal.

The included examples are:

### MNIST

This example implements a very basic LeNet5-type neural network, trained on the MNIST dataset for handwritten digit recognition.

Run the app and point the camera at a handwritten digit (there are some images in the `Test Images` folder you can use for this) and the app will tell you what digit it is, and how sure it is of this prediction.

![MNIST example](Examples/MNIST/MNIST.jpg)

The small image in the top-left corner shows what the network sees (this is the output of the preprocessing shader that attempts to increase the contrast between black and white).

There are two targets in this project: 

- MNIST
- MNIST-DSL

They do the exact same thing, except the first one is written with straight MPSCNN code and the second one uses the Forge DSL and is therefore much easier to read.

### Inception-v3

Google's famous [Inception network](https://arxiv.org/pdf/1512.00567v3.pdf) for image classification. Point your phone at some object and the app will give you its top-5 predictions:

![Inception example](Examples/Inception/Inception.jpg)

The Inception example app is based on [Apple's sample code](https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html) but completely rewritten using the DSL. We use their learned parameters. Thanks, Apple!

### YOLO

YOLO is an object detection network. It can detect multiple objects in an image and will even tell you where they are!

![YOLO example](Examples/YOLO/YOLO.jpg)

The example app implements the Tiny YOLO network, which is not as accurate as the full version of [YOLO9000](https://pjreddie.com/darknet/yolo/) and can detect only 20 different kinds of objects.

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmon and Ali Farhadi (2016).

### MobileNets

The **MobileNets** example app is an implementation of the network architecture from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1).

It works like Inception-v3 but is much faster. On the iPhone 6s it runs at 20 FPS with only moderate-to-high energy usage.

Forge uses the pretrained weights from [shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe).

## How to add Forge to your own project

Use Xcode 9 or better.

1. Copy the **Forge** folder into your project.
2. Use **File > Add Files to "YourProject" > Forge.xcodeproj** to add the Forge project inside your own project.
3. Drag **Products/Forge.framework** into the **Embedded Binaries** section of your project settings.
4. `import Forge` in your code.

NOTE: You cannot build for the simulator, only for "Generic iOS Device" or an actual device with arm64 architecture.

## How to use Forge

- [Creating a model with Forge](Docs/DSL.markdown)
- [Importing a model from Keras, TensorFlow, Caffe, etc](Docs/Importing.markdown)

## Where are the unit tests?

Run the **ForgeTests** app on a device.

The reason the tests are in a separate app is that Metal does not work on the simulator and Xcode can't run logic tests on the device. Catch-22.

## TODO

Forge is under active development. Here is the [list of bugs and upcoming features](Docs/TODO.markdown).

## License and credits

Forge is copyright 2016-2017 Matthijs Hollemans and is licensed under the terms of the [MIT license](LICENSE.txt).
