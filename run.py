import mnist
import numpy as np
from model import SVMTree
from PIL import Image

# Download and get mnist
training_images = mnist.train_images()
test_images = mnist.test_images()
training_labels = mnist.train_labels()
test_labels = mnist.test_labels()

# Set images to between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Flatten to 2 dimensions
training_images_flat = training_images.reshape((training_images.shape[0], training_images.shape[1] * training_images.shape[2]))
test_images_flat = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))

# Create svm tree
svm = SVMTree(training_images_flat.shape[1], 0.003, list(range(10)), 0.0001)

# Train svm
svm.train(training_images_flat, training_labels, 2)

print(svm.inference(test_images_flat[4,:]))
print(test_labels[4])
