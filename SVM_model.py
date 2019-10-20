import torch
import pickle
import random
from torch.nn import functional as F
from tqdm import tqdm


class SVMTree:
    def __init__(self, input_size, classes, learning_rate, train_mode=True):
        self.svms = {}
        self.optimizers = {}
        self.w = torch.randn(len(classes), input_size, dtype=torch.float32, requires_grad=train_mode)
        self.b = torch.randn(len(classes), dtype=torch.float32, requires_grad=train_mode)
        self.optim = torch.optim.SGD((self.w, self.b), lr=learning_rate)
        self.classes = classes
        self.input_size = input_size

    def forward(self, x):
        return torch.matmul(self.w, x) + self.b

    def loss(self, x, y):
        f = self.forward(x)
        return F.cross_entropy(f.view(1, -1), y.view((1)))

    def step(self, x, y):
        self.optim.zero_grad()
        x =  torch.tensor(x, dtype=torch.float32) if type(x) != torch.Tensor else x.clone().detach()
        t = torch.tensor(y, dtype=torch.long)
        loss = self.loss(x, t)
        if loss != 0:
            loss.backward()
            self.optim.step()
        return loss

    def inference(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.train_mode(False)
        inferenced = self.forward(x)
        return torch.argmax(inferenced)

    def epoch(self, x_list, y_list):
        losses = []
        for index, x in tqdm(enumerate(x_list), total=len(x_list), unit='steps', dynamic_ncols=True, ascii=True):
            y = y_list[index]
            ls = self.step(x, y)
            if ls is not None:
                losses.append(ls)
        return losses

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

    def train_mode(self, mode):
        self.w.requires_grad_(mode)
        self.b.requires_grad_(mode)

    def save_tree(self, path):
        self.train_mode(False)
        pickle.dump((self.w, self.b), open(path, 'wb'))

    def evaluate(self, x, y):
        correct = 0.0
        count = x.shape[0]
        for index in tqdm(range(count), desc='inference', ascii=True):
            y_inf = self.inference(x[index])
            if y_inf == y[index]:
                correct += 1.0
        return correct / count
