import torch
import pickle
import operator
import numpy as np

def inference_dict(x, svm_dict):
    inferenced = {}
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    for key in svm_dict:
        inferenced[key] = torch.dot(svm_dict[key][0], x) + svm_dict[key][1]
    return max(inferenced.items(), key=operator.itemgetter(1))[0]

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    preturbed_image = image + epsilon*sign_data_grad
    preturbed_image = torch.clamp(preturbed_image, -1, 1)
    return preturbed_image

def stage(x_arr, y_arr, svm_path, epsilon):
    svm_dict = pickle.load(open(svm_path, 'rb'))
