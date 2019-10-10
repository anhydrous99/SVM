from model import SVM_Tree
import torch
import random

classes = list(range(10))
svm_tree = SVM_Tree(2, 0.01, classes)

j = random.randint(0, 9)
tj = torch.tensor([0.8, 0.5])

svm_tree.epoch([tj, tj, tj, tj], [5, 6, 8, 10])
svm_tree.train([tj, tj, tj, tj], [5, 6, 8, 10], 2)