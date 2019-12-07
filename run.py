import time
import mnist
import argparse
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from SVM_model import SVMTree
from fgsm import stage
from utils import fig_creator, frange
from tqdm import tqdm
from FDA import calculate_fisher_discriminant

parser = argparse.ArgumentParser(description='Create an SVM and Break it using an FGSM attack')
subparsers = parser.add_subparsers(dest='subparser')
svm_tune_parser = subparsers.add_parser('svm_tune', help='Uses optuna to find best hyper-parameters')
svm_parser = subparsers.add_parser('svm', help='Create and train SVM for classifing mnist')
svm_parser.add_argument('-l', '--lr', default=0.09388254962230703, type=float, help='The learning rate')
svm_parser.add_argument('-e', '--epochs', default=14, type=int, help='The number of epochs to train for')
svm_parser.add_argument('-g', '--gamma', default=0.00333990187, type=float, help='The RBF kernel approximation gamma')
svm_parser.add_argument('--dims', default=834, type=int, help='The number of dimensions to use with RBF kernel')
svm_parser.add_argument('-s', '--save', default='SVM_tree.pickle', help='Where to save the pickled data')
svm_parser.add_argument('-c', '--cutoff', default=0.22884, help='The cutoff percentage to use with FDA')
att_parser = subparsers.add_parser('att', help='Attack the created SVMs')
att_parser.add_argument('-e', '--epsilon', default=0.08, type=float, help='Aggressiveness of attack')
att_parser.add_argument('-d', '--data', default='SVM_tree.pickle', help='The saved svm to attack')
att_parser.add_argument('--save')
fda_parser = subparsers.add_parser('fda', help='Run FDA tests')
args = parser.parse_args()

training_images = None
test_images = None
training_labels = None
test_labels = None
training_images_flat = None
test_images_flat = None

if args.subparser == 'svm' or args.subparser == 'svm_tune' or args.subparser == 'fda':
    # Download and get mnist
    training_images = mnist.train_images()
    test_images = mnist.test_images()
    training_labels = mnist.train_labels()
    test_labels = mnist.test_labels()
    # Set images to between 0 and 1
    training_images = training_images / 127.5 - 1
    test_images = test_images / 127.5 - 1
    # Flatten to 2 dimensions
    training_images_flat = training_images.reshape((training_images.shape[0],
                                                    training_images.shape[1] * training_images.shape[2]))
    test_images_flat = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))


if args.subparser is None:
    parser.print_help()
    exit(0)

if args.subparser == 'svm':
    x, y = calculate_fisher_discriminant(training_images_flat, training_labels, args.cutoff)
    svm = SVMTree(x.shape[1], list(range(10)), args.lr,
                  dimensions=args.dims, gamma=args.gamma)
    t1 = time.time()
    svm.train(x, y, args.epochs)
    t2 = time.time()
    print(f'Test Accuracy: {round(svm.evaluate(test_images_flat, test_labels) * 100, 2)}%')
    print(f'It took: {t2 - t1}s to train')
    svm.save('SVM_tree.pickle')

if args.subparser == 'svm_tune':
    def objective(trial):
        learning_rate = trial.suggest_uniform('learning_rate', 0.0, 0.1)
        epochs = trial.suggest_int('epochs', 1, 20)
        gamma = trial.suggest_uniform('gamma', 0.0, 0.1)
        dimensions = trial.suggest_int('dimensions', 750, 1000)
        cutoff = trial.suggest_uniform('cutoff', 0.1, 0.9)
        x, y = calculate_fisher_discriminant(training_images_flat, training_labels, cutoff)
        local_svm = SVMTree(x.shape[1], list(range(10)), learning_rate,
                            dimensions=dimensions, gamma=gamma)
        return local_svm.train(x, y, epochs)
    study = optuna.create_study()
    study.optimize(objective, n_trials=500)
    print(study.best_params)

if args.subparser == 'att':
    test_images = mnist.test_images() / 127.5 - 1
    test_labels = mnist.test_labels()
    adv_ex, bro_ex, grad_ex = stage(test_images, test_labels, args.data, args.epsilon)

    for index, adv in tqdm(enumerate(adv_ex), total=len(adv_ex)):
        fig_creator(test_images[adv[3]], grad_ex[index][0], adv[0], adv[1], adv[2], False, f'samples/{adv[3]}.png')

if args.subparser == 'fda':
    def objective(trial, ctf):
        learning_rate = trial.suggest_uniform('learning_rate', 0.0, 0.1)
        epochs = trial.suggest_int('epochs', 1, 20)
        gamma = trial.suggest_uniform('gamma', 0.0, 0.1)
        dimensions = trial.suggest_int('dimensions', 750, 1000)
        x, y = calculate_fisher_discriminant(training_images_flat, training_labels, ctf)
        local_svm = SVMTree(x.shape[1], list(range(10)), learning_rate,
                            dimensions=dimensions, gamma=gamma)
        return local_svm.train(x, y, epochs)
    data = []
    for ctf in frange(0.2, 0.8, 0.2):
        study = optuna.create_study(pruner=MedianPruner())
        study.optimize(lambda trial: objective(trial, ctf), n_trials=100)
        best = study.best_params

        x, y = calculate_fisher_discriminant(training_images_flat, training_labels, ctf)
        svm = SVMTree(x.shape[1], list(range(10)), best['learning_rate'],
                      dimensions=best['dimensions'], gamma=best['gamma'])
        t1 = time.time()
        svm.train(x, y, best['epochs'])
        t2 = time.time()
        print(f'Test Accuracy: {round(svm.evaluate(test_images_flat, test_labels) * 100, 2)}%')
        print(f'It took: {t2 - t1}s to train')
        data.append({'accuracy': round(svm.evaluate(test_images_flat, test_labels) * 100, 2),
                     'cutoff': ctf,
                     'n_samples': x.shape[0]})
    df = pd.DataFrame(data)
    df.to_csv('data.csv')
