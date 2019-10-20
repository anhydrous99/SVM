import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


def batch_np_to_image(arr, save_path, name):
    save_path = Path(save_path)
    for index in range(len(arr)):
        np_to_image(arr[index], save_path / f'{name}_{index}.png')


def np_to_image(arr, save_path=None, show=False):
    img = Image.fromarray(((arr[0] + 1) * 127.5).astype(np.uint8), 'L')
    if save_path is not None:
        img.save(save_path)
    if show:
        img.show()
    return img


def fig_creator(start_image, gradient_image, end_image, y_target, y_predicted):
    start_image = ((start_image + 1) * 127.5).astype(np.uint8)
    gradient_image = ((gradient_image + 1) * 127.5).astype(np.uint8)
    end_image = ((end_image + 1) * 127.5).astype(np.uint8)
    f = plt.figure()
    ax1 = f.add_subplot(1, 3, 1, title=r"$x$" + f"-pred:{y_target}")
    ax1.axis('off')
    plt.imshow(start_image)
    ax2 = f.add_subplot(1, 3, 2, title=r"sign$(\nabla_x J(\theta,x,y))$")
    ax2.axis('off')
    plt.imshow(gradient_image)
    ax3 = f.add_subplot(1, 3, 3, title=r"$\epsilon$sign$(\nabla_x J(\theta,x,y))$" + f"-pred:{y_predicted}")
    ax3.axis('off')
    plt.imshow(end_image)
    plt.show(block=True)
