# Creating a model with Forge

Forge's domain-specific language (DSL) allows you to specify your neural network in just a few lines of code.

The steps to define your model are:

1. Define an `Input` tensor.
2. Create layers and connect them.
3. Instantiate a `Model` and compile it.

### The Input tensor

A *tensor* describes data. It's really just a fancy word for multidimensional array. Because we're using MPSCNN, our tensors use an `MPSImage` or `MPSTemporaryImage` to store their data (there are also tensors that write into the `MPSImage` belonging to another tensor).

You always start by declaring an `Input` tensor:

```swift
let input = Input()
```

Or like so:

```swift
let input = Input(width: 100, height: 100, channels: 3)
```

Tensors have a *shape*, which describes their width, height, and number of feature channels (also called depth).

When you declare the `Input` tensor without any parameters, as in the first example, it has an unknown shape. That means it will accept an image of any width, height, and depth. This is usually how you want to set up things.

In the second example, the `Input` has a fixed shape and will only accept images with those dimensions.

Most layers require their inputs to have a known size. As the data flows through your neural network, at some point it will encounter a layer that requires the tensor to have a specific shape. If any parts of the tensor's shape are still undefined at that point, Forge will give an error.

This is why the `Input` from the first example should be followed by a `Resize` layer (or a `Custom` layer) that scales the image to a specific width and height (and depth).

### Layers

A layer takes a tensor, performs some kind of computation on the data, and outputs a new tensor.

You can declare a layer like this:

```swift
let sigmoid = MPSCNNNeuronSigmoid(device: device)
let layer = Dense(neurons: 100, activation: sigmoid, name: "dense1")
```

This creates a new fully-connected -- or "dense" -- layer with 100 neurons and sigmoid activation (provided by an `MPSCNNNeuron` object). The name `"dense1"` is used to load the parameters for this layer.

To get the output of the layer, you apply the layer to the input tensor:

```swift
let output = input --> layer
```

The `-->` is a custom operator that makes it easy to connect tensors and layers. Here, `output` is a new tensor.

One way to create the model is in separate steps:

```swift
let input = Input()

let layer1 = Resize(width: 28, height: 28)
let layer2 = Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")
let layer3 = Dense(neurons: 10, name: "dense1")
let layer4 = Softmax()

var x: Tensor
x = input --> layer1
x = x --> layer2
x = x --> layer3
x = x --> layer4
``` 

But more typically, you'd create your model by connecting all the tensors and layers in one go:

```swift
let input = Input()

let output = input
         --> Resize(width: 28, height: 28)
         --> Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")
         --> Dense(neurons: 10, name: "dense1")
         --> Softmax()
```

The nice thing about this mini-language is that it automatically infers the size of the data as it flows through the network. With the possible exception of the `Input` tensor, you never have to specify how large the tensors are. (You can see these inferred sizes with `print(model.summary())` after you've compiled the model.)

Sometimes it's useful to keep track of a specific layer or tensor. In that case you'd store a reference to these objects somewhere:

```swift
let resizeLayer: Resize
let conv1Output: Tensor
. . .

let input = Input()

resizeLayer = Resize(width: 28, height: 28)

conv1Output = input 
         --> resizeLayer 
         --> Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")

let output = conv1Output 
         --> Dense(neurons: 10, name: "dense1")
         --> Softmax()
```

Now we can use `resizeLayer` to change the properties of this layer later on (for example, to crop the input image based on face detection).

We can use `conv1Output` to access the tensor. Normally you don't need to keep track of individual tensors, but it's handy for debugging purposes. A tensor writes its data into an `MPSTemporaryImage`, which only exists for a very short while. By setting the tensor's `imageIsTemporary` property to false, it will use a permanent `MPSImage` instead. After the forward pass through the network completes, you can ask the model for this tensor's image and look at its contents. This is useful for making sure your neural network actually computes the right thing.

```swift
// before compiling:
conv1Output.imageIsTemporary = false

// after inference:
let image = model.image(for: conv1Output, inflightIndex: i)
print(image.toFloatArray())
```

This method of connecting tensors to layers to tensors to layers etc is quite powerful. For example, here's how you can make an Inception module:

```swift
let avgPool = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: true)

let mixed0 = Concatenate([
  initial --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_conv"),
  initial --> Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "mixed_tower_conv")
          --> Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "mixed_tower_conv_1"),
  initial --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_tower_1_conv")
          --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_tower_1_conv_1")
          --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_tower_1_conv_2"),
  initial --> avgPool
          --> Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "mixed_tower_2_conv")
])
```

The `initial` tensor contains the output from the first part of the network. Here that same tensor is passed into four different layers. The outputs of these layers are concatenated into a new tensor, `mixed0`. The entire Inception-v3 network is built up of such modules. Also note that the `avgPool` layer is stored in a separate variable. That's because this particular layer will be reused throughout the network. You're allowed to use layers on multiple tensors.

### Compiling the model

The `Model` contains the graph you have just defined and is the main interface the rest of your app will be interacting with.

To create the model, you supply both the input and the output tensor:

```swift
model = Model(input: input, output: output)
```

Once you've created the model you can compile it:

```swift
let success = model.compile(device: device, inflightBuffers: 3) {
  name, count, type in 
  return ParameterLoaderBundle(name: name, count: count,
                               suffix: type == .weights ? "_W" : "_b",
                               ext: "bin")
}

if success {
  print(model.summary())
}
```

Compiling calculates the sizes of all the tensors, builds all the MPSCNN kernels, loads the parameters, and prepares the neural network for use.

An `MPSCNNConvolution` or `MPSCNNFullyConnected` object needs to know the weights and biases the network has learned during training, so you have to provide these somehow. To this end, `model.compile()` takes a closure that should return a `ParameterData` object.

This closure is called for every layer that takes parameters, once for the weights and once for the biases. The above example returns a `ParameterLoaderBundle` instance, an implementation of `ParameterData` that reads the weights and biases from files stored in the app bundle. For the layer named `"conv1"`, this would load the files **conv1_W.bin** (weights) and **conv1_b.bin** (biases).

Because Forge uses this indirect mechanism for loading weights and biases, you can store them anywhere you want: in multiple smaller files, in one big file, in the asset catalog, in files you downloaded from a server, encrypted, etc. Forge does not force a particular storage type upon you.

After the model successfully compiles, `print(model.summary())` will output a list of all the layers that are in the model. This is useful for double-checking that you specified everything correctly (and that Forge didn't make any mistakes!).

### Running the model

Once you have a compiled model you call `model.encode()` to encode the GPU commands into a Metal `MTLCommandBuffer` object:

```swift
model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: i)
```

This performs a single forward pass of the neural network.

After the command buffer completes executing on the GPU, you can read the neural network's output as follows:

```swift
let probabilities = model.outputImage(inflightIndex: i).toFloatArray()
let top5 = probabilities.top(k: 5)
let top5Labels = top5.map { x -> (String, Float) in (labels[x.0], x.1) }
```

The `inflightIndex` parameter is used for *triple-buffering*, a mechanism that prevents the CPU and GPU from waiting on each other. This improves throughput so that the CPU can already be encoding the commands for the next video frame while GPU is still working on the previous frame. Forge provides a class `Runner` that takes care of all the CPU-GPU synchronization stuff for you. You can read more about it [in the blog post](http://machinethink.net/blog/forge-neural-network-toolkit-for-metal/).
