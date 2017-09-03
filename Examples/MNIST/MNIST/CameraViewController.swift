/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import UIKit
import Metal
import MetalPerformanceShaders
import CoreMedia
import Forge

let MaxBuffersInFlight = 3   // use triple buffering

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var predictionLabel: UILabel!
  @IBOutlet weak var extraLabel: UILabel!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!

  var videoCapture: VideoCapture!
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var runner: Runner!
  var network: MNIST!

  var startupGroup = DispatchGroup()
  let fpsCounter = FPSCounter()

  override func viewDidLoad() {
    super.viewDidLoad()

    predictionLabel.text = ""
    extraLabel.text = ""
    timeLabel.text = ""

    device = MTLCreateSystemDefaultDevice()
    if device == nil {
      print("Error: this device does not support Metal")
      return
    }

    commandQueue = device.makeCommandQueue()

    // NOTE: At this point you'd disable the UI and show a spinner.

    videoCapture = VideoCapture(device: device)
    videoCapture.delegate = self

    // Initialize the camera.
    startupGroup.enter()
    videoCapture.setUp { success in
      // Add the video preview into the UI.
      if let previewLayer = self.videoCapture.previewLayer {
        self.videoPreview.layer.addSublayer(previewLayer)
        self.resizePreviewLayer()
      }
      self.startupGroup.leave()
    }

    // Initialize the neural network.
    startupGroup.enter()
    createNeuralNetwork {
      self.startupGroup.leave()
    }

    // Once the NN is set up, we can start capturing live video.
    startupGroup.notify(queue: .main) {
      // NOTE: At this point you'd remove the spinner and enable the UI.

      self.fpsCounter.start()
      self.videoCapture.start()
    }
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

  // MARK: - Neural network

  func createNeuralNetwork(completion: @escaping () -> Void) {
    // Make sure the current device supports MetalPerformanceShaders.
    guard MPSSupportsMTLDevice(device) else {
      print("Error: this device does not support Metal Performance Shaders")
      return
    }

    runner = Runner(commandQueue: commandQueue, inflightBuffers: MaxBuffersInFlight)

    // Because it may take a few seconds to load the network's parameters,
    // perform the construction of the neural network in the background.
    DispatchQueue.global().async {

      timeIt("Setting up neural network") {
        self.network = MNIST(device: self.device, inflightBuffers: MaxBuffersInFlight)
      }
      DispatchQueue.main.async(execute: completion)
    }
  }

  func predict(texture: MTLTexture) {
    // Since we want to run in "realtime", every call to predict() results in
    // a UI update on the main thread. It would be a waste to make the neural
    // network do work and then immediately throw those results away, so the 
    // network should not be called more often than the UI thread can handle.
    // It is up to VideoCapture to throttle how often the neural network runs.

    runner.predict(network: network, texture: texture, queue: .main) { result in
      self.show(predictions: result.predictions)

      if let texture = result.debugTexture {
        self.debugImageView.image = UIImage.image(texture: texture, scale: result.debugScale, offset: result.debugOffset)
      }

      self.fpsCounter.frameCompleted()
      self.timeLabel.text = String(format: "%.1f FPS (latency: %.5f sec)", self.fpsCounter.fps, result.latency)
    }
  }

  private func show(predictions: [MNIST.Prediction]) {
    let p = predictions[0]
    if p.probability > 0.9 {
      predictionLabel.text = p.label
      extraLabel.text = String(format: "%2.1f%%", p.probability * 100)
    } else {
      predictionLabel.text = "???"
      extraLabel.text = String(format: "%@ (%2.1f%%)", p.label, p.probability * 100)
    }
  }
}

extension CameraViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
    // To test with a fixed image (useful for debugging), do this:
    //predict(texture: loadTexture(named: "three.png")!)

    // Call the predict() method, which encodes the neural net's GPU commands,
    // on our own thread. Since NeuralNetwork.predict() can block, so can our
    // thread. That is OK, since any new frames will be automatically dropped
    // while the serial dispatch queue is blocked.
    if let texture = texture {
      //timeIt("Encoding") {
        predict(texture: texture)
      //}
    }
  }

  func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
    // not implemented
  }
}
