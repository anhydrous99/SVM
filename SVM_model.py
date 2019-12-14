import torch
import pickle
import random
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm


class SVMTree:
    def __init__(self, input_size, classes, learning_rate, train_mode=True, dimensions=940, gamma=0.004, device='cpu'):
        if input_size is None or classes is None or learning_rate is None:
            return
        # Save both the classes and the input size
        self.classes = classes
        self.input_size = input_size
        # Save a RBF fourier kernel approximation parameters
        self.gamma = gamma
        self.n_components = dimensions
        self.device = device
        self.random_weights_ = np.sqrt(2 * self.gamma) * torch.randn(self.input_size, self.n_components, device=device)
        self.random_offset_ = 2 * np.pi * torch.rand(self.n_components, device=device)
        # These are the SVM's parameters
        self.w = np.sqrt(2 * self.gamma) * torch.randn(len(classes), self.n_components, dtype=torch.float32, device=device)
        self.b = 2 * np.pi * torch.rand(len(classes), dtype=torch.float32, device=device)
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
        self.optim.zero_grad()  # Zero out the gradient
        # Get x to a torch tensor type
        x = torch.tensor(x, dtype=torch.float32, device=self.device) if type(x) != torch.Tensor else x
        # Get y to a torch tensor type
        t = torch.tensor(y, dtype=torch.long, device=self.device) if type(x) != torch.Tensor else y
        # Calculate loss
        loss = self.loss(x, t)
        if loss != 0:
            loss.backward()  # Use back-propagation to calculate gradients
            self.optim.step()  # Optimize the weights and biases according to the gradients
        return loss  # Return the loss

    # Pass data through the SVMs without remembering operations
    def inference(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.train_mode(False)
        inferenced = self.forward(x)
        return torch.argmax(inferenced)

    # Train to an epoch
    def epoch(self, x_list, y_list):
        losses = []
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps'):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

    # The main training function
    def train(self, x_list, y_list, n_epochs, shuffle=True):
        x_list_device = torch.tensor(np.stack(x_list), dtype=torch.float32, device=self.device)
        y_list_device = torch.tensor(np.stack(y_list), dtype=torch.long, device=self.device)
        indexes = list(range(len(x_list[:, 0])))
        pbar = tqdm(range(n_epochs), unit='epochs')
        losses = []
        total_total_loss = 0
        for _ in pbar:
            total_loss = 0
            if shuffle:
                random.shuffle(indexes)
            for index in tqdm(indexes, unit='steps'):
                last_loss = self.step(x_list_device[index, :], y_list_device[index])
                if last_loss is not None:
                    total_loss += last_loss.to(device='cpu').detach().item()
            loss_avg = total_loss / len(indexes)
            losses.append(loss_avg)
            pbar.set_description(f'loss: {round(loss_avg, 4)}')
            total_total_loss += loss_avg if loss_avg is not None else 0
        return float(total_total_loss / n_epochs)

    # The training mode, whether to have the weights and biases remember operations for back-propagation
    def train_mode(self, mode):
        self.w.requires_grad_(mode)
        self.b.requires_grad_(mode)

    # Save the SVMs to a file
    def save(self, path):
        self.train_mode(False)
        pickle.dump(self.__dict__, open(path, 'wb'))

    def load(self, path):
        svm = pickle.load(open(path, 'rb'))
        self.__dict__.update(svm)
        return self

    # Evaluate the SVMs performance
    def evaluate(self, x, y):
        correct = 0.0
        count = x.shape[0]
        for index in tqdm(range(count), desc='inference'):
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
