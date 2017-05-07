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

## Run the demo!

To see a demo of Forge in action, open **Forge.xcworkspace** in Xcode and run the **Inception** app on your device.

You need at least Xcode 8.3 and a device with an A8 processor (iPhone 6 or better) running iOS 10 or later. You cannot build for the simulator as it does not support Metal.

Also try out the other demo apps:

- MNIST
- MNIST-DSL (same as MNIST but using a lot less code)
- MobileNets

## How to install Forge

Use Xcode 8.3 or better.

1. Copy the **Forge** folder into your project.
2. Use **File > Add Files to "YourProject" > Forge.xcodeproj** to add the Forge project inside your own project.
3. Drag **Products/Forge.framework** into the **Embedded Binaries** section of your project settings.
4. `import Forge` in your code.

NOTE: You cannot build for the simulator, only for "Generic iOS Device" or an actual device with arm64 architecture.

## How to use Forge

- [Creating a model with Forge](Docs/DSL.markdown)
- [Importing a model from Keras, TensorFlow, Caffe, etc](Docs/Importing.markdown)

## TODO

Forge is under active development. Here is the [list of bugs and upcoming features](Docs/TODO.markdown).

## License and credits

Forge is copyright 2016-2017 Matthijs Hollemans and is licensed under the terms of the [MIT license](LICENSE.txt).

The **Inception** example app is based on [Apple's Inception-v3 sample code](https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html) but completely rewritten using the DSL. We use their learned parameters. Thanks, Apple!

The **MobileNets** example app is an implementation of the network architecture from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1).

