import mnist
import numpy as np
from scipy.linalg import eig


def calculate_fisher_discriminant(x_list, y_class):
    count = {}
    x_dic = {}
    score = {}
    for index in range(x_list.shape[0]):
        y = y_class[index]
        if y in count.keys():
            count[y] += 1
            x_dic[y].append(x_list[index])
        else:
            count[y] = 1
            x_dic[y] = [x_list[index]]

    for y in count.keys():
        x = np.stack(x_dic[y], axis=1)



# Download and get mnist
training_images = mnist.train_images()
test_images = mnist.test_images()
training_labels = mnist.train_labels()
test_labels = mnist.test_labels()
# Set images to between 0 and 1
training_images = training_images / 127.5 - 1.0
test_images = test_images / 127.5 - 1.0
# Flatten to 2 dimensions
training_images_flat = training_images.reshape((training_images.shape[0],
                                            training_images.shape[1] * training_images.shape[2]))
test_images_flat = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))

calculate_fisher_discriminant(training_images_flat, training_labels)