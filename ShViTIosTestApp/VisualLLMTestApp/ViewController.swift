import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func pickImageTapped(_ sender: UIButton) {
        presentImagePicker()
    }

    func presentImagePicker() {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        guard let image = info[.originalImage] as? UIImage else { return }
        imageView.image = image
        runModel(on: image)
    }

    func runModel(on image: UIImage) {
        guard let buffer = image.toCVPixelBuffer(size: CGSize(width: 256, height: 256)) else {
            resultLabel.text = "Failed to convert image"
            return
        }

//        guard let model = try? shvit_s4_256_256(configuration: MLModelConfiguration()) else {
//            resultLabel.text = "Failed to load model"
//            return
//        }
        let config = MLModelConfiguration()

        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #endif

        guard let model = try? shvit_s4_256_256(configuration: config) else {
            resultLabel.text = "Failed to load model"
            return
        }


        do {
            let inputMultiArray = try buffer.toMultiArray()
            let output = try model.prediction(images: inputMultiArray)

            let result = output.var_1461
            let topIndex = result.topIndex()
//            resultLabel.text = "Predicted Class Index: \(topIndex)"
            let label = ImageNetLabels.shared.label(for: topIndex)
            resultLabel.text = "Predicted Class: \(label)"

        } catch {
            resultLabel.text = "Prediction failed: \(error.localizedDescription)"
        }
    }
}

extension UIImage {
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(size.width),
                                         Int(size.height),
                                         kCVPixelFormatType_32BGRA,
                                         attrs,
                                         &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                width: Int(size.width),
                                height: Int(size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        guard let cgImage = self.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))

        return buffer
    }
}

extension CVPixelBuffer {
    func toMultiArray() throws -> MLMultiArray {
        CVPixelBufferLockBaseAddress(self, .readOnly)

        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let baseAddress = CVPixelBufferGetBaseAddress(self)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        let array = try MLMultiArray(shape: [1, 3, height as NSNumber, width as NSNumber], dataType: .float32)
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
        let bytePerRow = CVPixelBufferGetBytesPerRow(self)

        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytePerRow + x * 4
                let r = Float32(buffer[offset + 2]) / 255.0
                let g = Float32(buffer[offset + 1]) / 255.0
                let b = Float32(buffer[offset + 0]) / 255.0

                let idx = y * width + x
                ptr[idx] = r
                ptr[height * width + idx] = g
                ptr[2 * height * width + idx] = b
            }
        }

        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        return array
    }
}


extension MLMultiArray {
    func topIndex() -> Int {
        var maxIndex = 0
        var maxValue = self[0].floatValue

        for i in 1..<self.count {
            let val = self[i].floatValue
            if val > maxValue {
                maxValue = val
                maxIndex = i
            }
        }
        return maxIndex
    }
}
