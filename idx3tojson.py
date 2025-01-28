import struct
import json
import numpy as np

def read_idx3_ubyte(file_path):
    """Read MNIST image data from idx3-ubyte file."""
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images

def read_idx1_ubyte(file_path):
    """Read MNIST label data from idx1-ubyte file."""
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def convert_to_json(image_file, label_file, output_json):
    """Convert MNIST idx3-ubyte and idx1-ubyte files to dataset.json."""
    images = read_idx3_ubyte(image_file)
    labels = read_idx1_ubyte(label_file)
    
    dataset = {}
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert image array to a comma-separated string
        image_str = ','.join(map(str, image))
        # One-hot encode the label
        label_one_hot = [0] * 10
        label_one_hot[label] = 1
        dataset[image_str] = label_one_hot
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {output_json}")

# Example usage
if __name__ == "__main__":
    train_images_file = 'train-images-idx3-ubyte'
    train_labels_file = 'train-labels-idx1-ubyte'
    output_json_file = 'dataset.json'
    convert_to_json(train_images_file, train_labels_file, output_json_file)