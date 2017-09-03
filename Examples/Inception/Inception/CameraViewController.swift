import UIKit
import Metal
import MetalPerformanceShaders
import CoreMedia
import Forge

let MaxBuffersInFlight = 3   // use triple buffering

/*
  Using Apple's implementation of Inception v3.

  Runs at 6-7 FPS on my iPhone 6s, but energy usage is very high.

  Changes I made:
    - Getting the top-5 predictions uses helper code.
    - Now uses triple-buffering so the CPU does not wait for the GPU.
*/

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var predictionLabel: UILabel!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!

  var videoCapture: VideoCapture!
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var runner: Runner!
  var network: Inception3!

  var startupGroup = DispatchGroup()
  let fpsCounter = FPSCounter()

  override func viewDidLoad() {
    super.viewDidLoad()

    predictionLabel.text = ""
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
        self.network = Inception3(device: self.device, inflightBuffers: MaxBuffersInFlight)
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
        self.debugImageView.image = UIImage.image(texture: texture)
      }

      self.fpsCounter.frameCompleted()
      self.timeLabel.text = String(format: "%.1f FPS (latency: %.5f sec)", self.fpsCounter.fps, result.latency)
    }
  }

  private func show(predictions: [Inception3.Prediction]) {
    var s: [String] = []
    for pred in predictions {
      s.append(String(format: "%@ %2.1f%%", pred.label, pred.probability * 100))
    }
    predictionLabel.text = s.joined(separator: "\n\n")
  }
}

extension CameraViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
    // To test with a fixed image (useful for debugging), do this:
    //predict(texture: loadTexture(named: "final3.jpg")!)

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
