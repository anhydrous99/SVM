import torch
import torch.nn.functional as F
import pickle
import operator
from tqdm import tqdm

def inference_dict(x, svm):
    output = torch.matmul(svm[0], x.flatten()) + svm[1]
    return torch.argmax(output), output

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    preturbed_image = image + epsilon * sign_data_grad
    preturbed_image = torch.clamp(preturbed_image, -1, 1)
    return preturbed_image

def stage(x_arr, y_arr, svm_path, epsilon):
    svm = pickle.load(open(svm_path, 'rb'))
    correct = 0
    adv_examples = []
    grad_examples = []
    bro_examples = []

    for index in range(x_arr.shape[0]):
        x = x_arr[index, :, :]
        y = y_arr[index]
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        pred, output = inference_dict(x, svm)

        # If prediction is incorrect
        if pred != y:
            bro_examples.append(x.squeeze().detach().numpy())
            continue

        # Calculate loss
        loss = F.nll_loss(output.view((1, -1)), y.view((1)))

        # Calculate gradients through backwards propagation
        loss.backward()

        # Get the gradients
        data_grad = x.grad.data

        # Call attack
        perturbed_data = fgsm_attack(x, epsilon, data_grad)

        # Re-classify the perturbed image
        pred, output = inference_dict(perturbed_data, svm)

        if pred == y:
            correct += 1
        else:
            adv_examples.append(perturbed_data.squeeze().detach().numpy())
            grad_examples.append(data_grad.squeeze().detach().numpy())

    accuracy = correct / float(x_arr.shape[0])
    print(f'Epsilon: {epsilon}\tTest Accuracy = {accuracy}')
    return adv_examples, bro_examples, grad_examples
