import Foundation

class ImageNetLabels {
    static let shared = ImageNetLabels()
    private var labels: [String] = []

    private init() {
        if let url = Bundle.main.url(forResource: "imagenet_labels", withExtension: "json") {
            do {
                let data = try Data(contentsOf: url)
                labels = try JSONDecoder().decode([String].self, from: data)
            } catch {
                print("Failed to load labels: \(error)")
            }
        }
    }

    func label(for index: Int) -> String {
        if index >= 0 && index < labels.count {
            return labels[index]
        } else {
            return "Unknown"
        }
    }
}
