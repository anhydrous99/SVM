import torch
import pickle
import random
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm


class SVMTree:
    def __init__(self, input_size, classes, learning_rate, train_mode=True):
        # Save both the classes and the input size
        self.classes = classes
        self.input_size = input_size
        # Save a RBF fourier kernel approximation parameters
        self.gamma = 0.004
        self.n_components = 940
        self.random_weights_ = np.sqrt(2 * self.gamma) * torch.randn(self.input_size, self.n_components)
        self.random_offset_ = 2 * np.pi * torch.rand(self.n_components)
        # These are the SVM's parameters
        self.w = np.sqrt(2 * self.gamma) * torch.randn(len(classes), self.n_components, dtype=torch.float32)
        self.b = 2 * np.pi * torch.rand(len(classes), dtype=torch.float32)
        self.w.requires_grad_(train_mode)
        self.b.requires_grad_(train_mode)
        # The SVM's optimizer uses Stochastic Gradient Descent
        self.optim = torch.optim.SGD((self.w, self.b), lr=learning_rate)

    # Pass the data through the SVMs
    def forward(self, x):
        return torch.matmul(self.w, self.kernel(x)) + self.b

    # Calculate the loss/error of the SVM
    def loss(self, x, y):
        return F.cross_entropy(self.forward(x).view(1, -1), y.view((1)))

    # Perform a step of SGD to minimize the loss function
    def step(self, x, y):
        self.optim.zero_grad() # Zero out the gradient
        # Get x to a torch tensor type
        x =  torch.tensor(x, dtype=torch.float32) if type(x) != torch.Tensor else x.clone().detach()
        # Get y to a torch tensor type
        t = torch.tensor(y, dtype=torch.long)
        # Calculate loss
        loss = self.loss(x, t)
        if loss != 0:
            loss.backward() # Use back-propagation to calculate gradients
            self.optim.step() # Optimize the weights and biases according to the gradients
        return loss # Return the loss

    # Pass data through the SVMs without remembering operations
    def inference(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.train_mode(False)
        inferenced = self.forward(x)
        return torch.argmax(inferenced)

    # Train to an epoch
    def epoch(self, x_list, y_list):
        losses = []
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps', ascii=True):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

    # The main training function
    def train(self, x_list, y_list, n_epochs, shuffle=True):
        indexes = list(range(len(x_list[:, 0])))
        pbar = tqdm(range(n_epochs), unit='epochs', ascii=True)
        losses = []
        for _ in pbar:
            total_loss = torch.zeros(1)
            if shuffle:
                random.shuffle(indexes)
            for index in tqdm(indexes, unit='steps', ascii=True):
                last_loss = self.step(x_list[index, :], y_list[index])
                if last_loss is not None:
                    total_loss += last_loss.data
            loss_avg = total_loss.data.item() / len(indexes)
            losses.append(loss_avg)
            pbar.set_description(f'loss: {round(loss_avg, 4)}')

    # The training mode, whether to have the weights and biases remember operations for back-propagation
    def train_mode(self, mode):
        self.w.requires_grad_(mode)
        self.b.requires_grad_(mode)

    # Save the SVMs to a file
    def save_tree(self, path):
        self.train_mode(False)
        pickle.dump((self.w, self.b), open(path, 'wb'))

    # Evaluate the SVMs performance
    def evaluate(self, x, y):
        correct = 0.0
        count = x.shape[0]
        for index in tqdm(range(count), desc='inference', ascii=True):
            y_inf = self.inference(x[index])
            if y_inf == y[index]:
                correct += 1.0
        return correct / count

    def kernel(self, x):
        projection = torch.matmul(x, self.random_weights_)
        projection += self.random_offset_
        projection = torch.cos(projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection
