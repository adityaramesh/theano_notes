#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Converts MNIST data from original format to HDF5.
"""

import os.path as path

# To read the original MNIST data.
import gzip
import struct
import numpy as np

# To convert to HDF5.
import h5py

def read_mnist(image_fn, label_fn):
    file_names = [image_fn, label_fn]
    image_file, label_file = [gzip.open(fn, 'rb') for fn in file_names]

    def read_int(f):
        return struct.unpack('>I', f.read(4))[0]
    
    # Verify the contents of the headers.
    files            = [image_file, label_file]
    magic_1, magic_2 = [read_int(f) for f in files]
    count, count_2   = [read_int(f) for f in files]
    rows, cols       = [read_int(image_file) for _ in range(2)]

    assert magic_1 == 0x00000803
    assert magic_2 == 0x00000801
    assert count == count_2
    assert rows == 28 and cols == 28

    # Parse the data.
    labels = np.fromstring(label_file.read(count), dtype=np.uint8)
    images = np.fromstring(image_file.read(count * rows * cols), dtype=np.uint8)
    images.resize(count, rows, cols)
    return images, labels

def main():
    mnist_dir = "data/mnist"
    file_names = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
    paths = [path.join(mnist_dir, fn) for fn in file_names]

    print("Parsing test images.")
    test_images, test_labels = read_mnist(paths[0], paths[1])
    print("Parsing train images.")
    train_images, train_labels = read_mnist(paths[2], paths[3])

    print("Creating HDF5 file.")
    with h5py.File(path.join(mnist_dir, "mnist.hdf5"), 'w') as f:
        f.create_dataset("/train/inputs", data=train_images)
        f.create_dataset("/train/targets", data=train_labels)
        f.create_dataset("/test/inputs", data=test_images)
        f.create_dataset("/test/targets", data=test_labels)

if __name__ == "__main__":
    main()
