import torch
import pickle
import random
from torch.nn import functional as F
from tqdm import tqdm


class SVMTree:
    def __init__(self, input_size, classes, learning_rate, train_mode=True):
        # These are the SVM's parameters
        self.w = torch.randn(len(classes), input_size, dtype=torch.float32, requires_grad=train_mode)
        self.b = torch.randn(len(classes), dtype=torch.float32, requires_grad=train_mode)
        # The SVM's optimizer uses Stochastic Gradient Descent
        self.optim = torch.optim.SGD((self.w, self.b), lr=learning_rate)
        # Save both the classes and the input size
        self.classes = classes
        self.input_size = input_size

    # Pass the data through the SVMs
    def forward(self, x):
        return self.kernel_product(self.w, x) + self.b

    # Calculate the loss/error of the SVM
    def loss(self, x, y):
        f = self.forward(x)
        return F.cross_entropy(f.view(1, -1), y.view((1)))

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
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps', dynamic_ncols=True, ascii=True):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

    # The main training function
    def train(self, x_list, y_list, n_epochs, shuffle=True):
        indexes = list(range(len(x_list[:, 0])))
        pbar = tqdm(range(n_epochs), unit='epochs', dynamic_ncols=True, ascii=True)
        losses = []
        for _ in pbar:
            total_loss = torch.zeros(1)
            if shuffle:
                random.shuffle(indexes)
            for index in tqdm(indexes, unit='steps', dynamic_ncols=True, ascii=True):
                last_loss = self.step(x_list[index, :], y_list[index])
                if last_loss is not None:
                    total_loss += last_loss.data
            loss_avg = total_loss.data.item() / len(indexes)
            losses.append(loss_avg)
            pbar.set_description(f'loss: {round(loss_avg, 2)}')

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

    def kernel_product(self, w, x, mode='energy', s=0.1):
        w_i = torch.t(w)
        xmy = ((w_i - x[:, None]) ** 2).sum(0)

        if mode == "gaussian":
            K = torch.exp(- (torch.t(xmy) ** 2) / (s ** 2))
        elif mode == "laplace":
            K = torch.exp(- torch.sqrt(torch.t(xmy) + (s ** 2)))
        elif mode == "energy":
            K = torch.pow(torch.t(xmy) + (s ** 2), -.25)

        return K
