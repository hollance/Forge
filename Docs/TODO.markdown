# Forge TODO

### General

- Improve API documentation
- Add package manager support of some kind
- Add more unit tests
- Add more layer types and compute kernels
- Add more examples
- Make a nice logo (use as icon for example apps)

### DSL

#### Slow compilation?

Sometimes compiling Inception takes 0.5 seconds and other times only about 0.12 seconds (which is similar to the setup time Apple's original Inception code). I wonder what's causing this... (Loading from disk? Creating the MPSCNN objects? Gremlins?)

#### Allow nested concatenation

```swift
let m1 = Concatenate([a, b])
let m2 = Concatenate([m1, c])
```

Here, tensors `a`, `b`, and `c` should all write into `m2`'s image (`m1` does not get an image). I haven't actually tried to do this yet, but I suspect the `destinationChannelOffset` for `c` would be wrong.

#### Allow "residual" or bypass connections

```swift
let a = ... 
let b = a --> Convolution()
let c = Concatenate(a, b)
```

Here, tensor `a` writes into `c`'s image (no problem). But `b` will read from `c`'s image and also write to `c`'s image. I haven't tried this yet. It might work, or not...

#### Make compilation more robust / multiple outputs

Currently you can write this:

```swift
let x = input --> Layer() --> ...
let y = x --> Layer()
let model = Model(input: input, output: x)
```

Here, `y` is an orphaned tensor. It is connected to the graph (and will show up below the last tensor from `x` in the summary) but its `MPSTemporaryImage` is not read by anything, and MPSCNN will throw an error.

One way to fix this is to not pass an explicit output to `Model()` but only the input. Any tensors in the array that don't have a next tensor will be treated as outputs and are given a real `MPSImage` (not a temporary one). The summary should list how many outputs it has found. `model.outputImage()` will take an index (no index means the first output).

Alternatively, just treat this as an error. If a tensor acts as an output but is not explicitly specified as being an output, then something's not right.

#### Reduce encoding overhead

It's not particularly slow as-is, but the less work we do during runtime, the better!

#### Add a compilation mode that outputs source code

The Forge DSL is nice for quickly putting together a neural network. But it can't do *everything* (yet) so for production code you may want to revert to writing "pure" MPSCNN code.

`Model.compile()` could take a flag so that it writes the Swift code for you and dumps it to stdout. Then you just have to copy-paste the code into your project and tweak it to fit your needs. That would save a lot of time!

### VideoCapture

This should properly handle interruptions from phone calls, FaceTime, going to the background, etc. It is not yet robust enough for production code.

### Automatic conversion of trained model weights

It might be possible to write a Python script that takes a trained model (e.g. a Keras .h5 file) and automatically converts the weights as an Xcode build phase. That way you don't have to convert the weights manually when your trained model changes.

### Examples

Refactor the examples to share more code. Currently there is a bit of code duplication going on.
