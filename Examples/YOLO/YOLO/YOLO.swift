import MetalPerformanceShaders
import Forge

let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/*
  The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/

  This implementation is cobbled together from the following sources:

  - https://github.com/pjreddie/darknet
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
  - https://github.com/allanzelener/YAD2K
*/
class YOLO: NeuralNetwork {
  typealias PredictionType = YOLO.Prediction

  public static let inputWidth = 416
  public static let inputHeight = 416
  public static let maxBoundingBoxes = 10

  // Tweak these values to get more or fewer predictions.
  let confidenceThreshold: Float = 0.3
  let iouThreshold: Float = 0.5

  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  let model: Model

  public init(device: MTLDevice, inflightBuffers: Int) {
    // Note: YOLO expects the input pixels to be in the range 0-1. Our input
    // texture most likely has pixels with values 0-255. However, since Forge
    // uses .float16 as the channel format the Resize layer will automatically
    // convert the pixels to be between 0 and 1.

    let leaky = MPSCNNNeuronReLU(device: device, a: 0.1)

    let input = Input()

    let output = input
             --> Resize(width: YOLO.inputWidth, height: YOLO.inputHeight)
             --> Convolution(kernel: (3, 3), channels: 16, activation: leaky, name: "conv1")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (3, 3), channels: 32, activation: leaky, name: "conv2")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (3, 3), channels: 64, activation: leaky, name: "conv3")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (3, 3), channels: 128, activation: leaky, name: "conv4")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (3, 3), channels: 256, activation: leaky, name: "conv5")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (3, 3), channels: 512, activation: leaky, name: "conv6")
             --> MaxPooling(kernel: (2, 2), stride: (1, 1), padding: .same)
             --> Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv7")
             --> Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv8")
             --> Convolution(kernel: (1, 1), channels: 125, activation: nil, name: "conv9")

    model = Model(input: input, output: output)

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 suffix: type == .weights ? "_W" : "_b",
                                                 ext: "bin")
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture sourceTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: sourceTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let featuresImage = model.outputImage(inflightIndex: inflightIndex)
    let features = featuresImage.toFloatArray()
    assert(features.count == 13*13*128)

    // We only run the convolutional part of YOLO on the GPU. The last part of
    // the process is done on the CPU. It should be possible to do this on the
    // GPU too, but it might not be worth the effort.

    var predictions = [Prediction]()

    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 20

    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
      let slice = channel / 4
      let indexInSlice = channel - slice*4
      let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
      return offset
    }

    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of 
    // five data items: x, y, width, height, and a confidence score. Each grid 
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell 
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 13x13x125 elements (actually x128 instead of x125 because in
    // Metal the number of channels must be a multiple of 4).

    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
        for b in 0..<boxesPerCell {

          // The 13x13x125 image is arranged in planes of 4 channels. First are
          // channels 0-3 for the entire image, then channels 4-7 for the whole
          // image, then channels 8-11, and so on. Since we have 128 channels,
          // there are 128/4 = 32 of these planes (a.k.a. texture slices).
          //
          //    0123 0123 0123 ... 0123    ^
          //    0123 0123 0123 ... 0123    |
          //    0123 0123 0123 ... 0123    13 rows
          //    ...                        |
          //    0123 0123 0123 ... 0123    v
          //    4567 4557 4567 ... 4567
          //    etc
          //    <----- 13 columns ---->
          //
          // For the first bounding box (b=0) we have to read channels 0-24, 
          // for b=1 we have to read channels 25-49, and so on. Unfortunately,
          // these 25 channels are spread out over multiple slices. We use a
          // helper function to find the correct place in the features array.
          // (Note: It might be quicker / more convenient to transpose this
          // array so that all 125 channels are stored consecutively instead
          // of being scattered over multiple texture slices.)
          let channel = b*(numClasses + 5)
          let tx = features[offset(channel, cx, cy)]
          let ty = features[offset(channel + 1, cx, cy)]
          let tw = features[offset(channel + 2, cx, cy)]
          let th = features[offset(channel + 3, cx, cy)]
          let tc = features[offset(channel + 4, cx, cy)]

          // The predicted tx and ty coordinates are relative to the location 
          // of the grid cell; we use the logistic sigmoid to constrain these 
          // coordinates to the range 0 - 1. Then we add the cell coordinates 
          // (0-12) and multiply by the number of pixels per grid cell (32).
          // Now x and y represent center of the bounding box in the original
          // 416x416 image space.
          let x = (Float(cx) + Math.sigmoid(tx)) * blockSize
          let y = (Float(cy) + Math.sigmoid(ty)) * blockSize

          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          let w = exp(tw) * anchors[2*b    ] * blockSize
          let h = exp(th) * anchors[2*b + 1] * blockSize

          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = Math.sigmoid(tc)

          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
          var classes = [Float](repeating: 0, count: numClasses)
          for c in 0..<numClasses {
            classes[c] = features[offset(channel + 5 + c, cx, cy)]
          }
          classes = Math.softmax(classes)

          // Find the index of the class with the largest score.
          let (detectedClass, bestClassScore) = classes.argmax()

          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = bestClassScore * confidence

          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > confidenceThreshold {
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))

            let prediction = Prediction(classIndex: detectedClass,
                                        score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
          }
        }
      }
    }

    // We already filtered out any bounding boxes that have very low scores, 
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    var result = NeuralNetworkResult<Prediction>()
    result.predictions = nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    //result.debugTexture = model.image(for: resized, inflightIndex: inflightIndex).texture
    return result
  }
}

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.
  
  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score. 
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain 
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}
