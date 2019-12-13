import torch
import torch.nn.functional as F
from tqdm import tqdm
from SVM_model import SVMTree


# The inferencing function, returns both the inferenced class and the output tensor
def inference_dict(x, svm):
    output = svm.forward(x.flatten())
    return torch.argmax(output), output


# Performs the FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    preturbed_image = image + epsilon * sign_data_grad
    preturbed_image = torch.clamp(preturbed_image, -1, 1)
    return preturbed_image


def stage(x_arr, y_arr, svm_path, epsilon, svm=None):
    if svm is None:
        svm = SVMTree(None, None, None)
        svm = svm.load(svm_path)
    correct = 0
    correct_prev = 0
    adv_examples = []
    grad_examples = []
    bro_examples = []

    for index in tqdm(range(x_arr.shape[0])):
        x = x_arr[index, :, :]
        y = y_arr[index]
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        pred, output = inference_dict(x, svm)

        # If prediction is incorrect
        if pred != y:
            bro_examples.append((
                x.squeeze().detach().numpy(),
                y.item(),
                pred.item(),
                index))
            continue
        correct_prev += 1

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
            adv_examples.append((
                perturbed_data.squeeze().detach().numpy(),
                y.item(),
                pred.item(),
                index))
            grad_examples.append((
                data_grad.squeeze().detach().numpy(),
                y.item(),
                pred.item(),
                index))

    accuracy = correct / float(x_arr.shape[0])
    prev_accuracy = correct_prev / (x_arr.shape[0])
    print(f'Epsilon: {epsilon}\tTest Accuracy = {accuracy}\tPrevious Accuracy = {prev_accuracy}')
    return adv_examples, bro_examples, grad_examples, accuracy
