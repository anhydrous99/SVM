import torch
import random
import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, input_size, epsilon):
        self.input_size = input_size
        self.epsilon = epsilon

        self.w = torch.rand(input_size, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)

    def forward(self, x):
        return torch.dot(self.w, x) - self.b

    def loss(self, x, y):
        return max(0, 1 - y * self.forward(x))**2.0

    def get_parameters(self):
        return self.w, self.b


class SVM_Tree:
    def __init__(self, input_size, epsilon, classes):
        input_size = np.prod(input_size)
        self.svms = {}
        self.optimizers = {}
        for cls in classes:
            self.svms[cls] = SVM(input_size, epsilon)
            self.optimizers[cls] = torch.optim.SGD(self.svms[cls].get_parameters(), 0.01)
        self.classes = classes
        self.epsilon = epsilon

    def step(self, x, y):
        for cls in self.classes:
            self.optimizers[cls].zero_grad()
            t = torch.tensor(1.0 if y == cls else 0.0)
            loss = self.svms[cls].loss(x, t)
            if loss != 0:
                loss.backward()
                self.optimizers[cls].step()
                return loss
            return None

    def epoch(self, x_list, y_list):
        losses = []
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps', dynamic_ncols=True):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

    def train(self, x_list, y_list, n_epochs, shuffle=True):
        indexes = range(x_list)
        losses = []
        pbar = tqdm(range(n_epochs), unit='epochs', dynamic_ncols=True)
        for i in pbar:
            last_loss = None
            if shuffle:
                random.shuffle(indexes)
            for index in tqdm(indexes, unit='steps', dynamic_ncols=True):
                last_loss = self.step(x_list[index], y_list[index])
            if last_loss is not None:
                pbar.set_description(f'loss: {last_loss}')
                losses.append(last_loss)
        return losses
