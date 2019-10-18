import torch
import pickle
import random
import operator
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm

class SVM:
    def __init__(self, input_size, epsilon, train_mode=True):
        self.input_size = input_size
        self.epsilon = torch.tensor(epsilon)

        self.w = torch.rand(input_size, requires_grad=train_mode)
        self.b = torch.rand(1, requires_grad=train_mode)

    def forward(self, x):
        return torch.dot(self.w, x) + self.b

    def loss(self, x, y):
        return F.mse_loss(self.forward(x), y.reshape(1), reduction='mean')

    def get_parameters(self, numpy_tensor=False):
        if not numpy_tensor:
            return self.w, self.b
        else:
            return self.w.data.numpy(), self.b.data.numpy()

    def train_mode(self, mode):
        self.w.requires_grad_(mode)
        self.b.requires_grad_(mode)


class SVMTree:
    def __init__(self, input_size, epsilon, classes, learning_rate, train_mode=True):
        input_size = np.prod(input_size)
        self.svms = {}
        self.optimizers = {}
        for cls in classes:
            self.svms[cls] = SVM(input_size, epsilon, train_mode)
            self.optimizers[cls] = torch.optim.SGD(self.svms[cls].get_parameters(), learning_rate)
        self.classes = classes
        self.epsilon = epsilon

    def step(self, x, y):
        l = 0
        for cls in self.classes:
            self.optimizers[cls].zero_grad()
            x =  torch.tensor(x, dtype=torch.float32) if type(x) != torch.Tensor else x.clone().detach()
            t = torch.tensor(1.0 if y.astype(int) == cls else -1.0)
            loss = self.svms[cls].loss(x, t)
            if loss != 0:
                loss.backward()
                self.optimizers[cls].step()
                l += loss
        return l

    def inference(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.train_mode(False)
        inferenced = {}
        for cls in self.classes:
            inferenced[cls] = self.svms[cls].forward(x)
        return max(inferenced.items(), key=operator.itemgetter(1))[0]

    def epoch(self, x_list, y_list):
        losses = []
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps', dynamic_ncols=True):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

    def train(self, x_list, y_list, n_epochs, shuffle=True):
        indexes = list(range(len(x_list[:, 0])))
        pbar = tqdm(range(n_epochs), unit='epochs', dynamic_ncols=True)
        losses = []
        for i in pbar:
            total_loss = 0
            if shuffle:
                random.shuffle(indexes)
            for index in tqdm(indexes, unit='steps', dynamic_ncols=True):
                last_loss = self.step(x_list[index, :], y_list[index])
                if last_loss is not None:
                    total_loss += last_loss.data
            loss_avg = total_loss.data.item() / len(indexes)
            losses.append(loss_avg)
            pbar.set_description(f'loss: {round(loss_avg, 4)}')

    def train_mode(self, mode):
        for cls in self.classes:
            self.svms[cls].train_mode(mode)

    def save_tree(self, path, as_numpy=True):
        to_save = {}
        for cls in self.classes:
            to_save[cls] = self.svms[cls].get_parameters(as_numpy)
        pickle.dump(to_save, open(path, 'wb'))
