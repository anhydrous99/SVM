import mnist
import argparse
import optuna
from SVM_model import SVMTree
from fgsm import stage
from utils import fig_creator
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create an SVM and Break it using an FGSM attack')
subparsers = parser.add_subparsers(dest='subparser')
svm_tune_parser = subparsers.add_parser('svm_tune', help='Uses optuna to find best hyper-parameters')
svm_tune_parser.add_argument('--use_fda', action='store_true', help='Uses FDA to find best points %')
svm_parser = subparsers.add_parser('svm', help='Create and train SVM for classifing mnist')
svm_parser.add_argument('-l', '--lr', default=0.07, type=float, help='The learning rate')
svm_parser.add_argument('-e', '--epochs', default=20, type=int, help='The number of epochs to train for')
svm_parser.add_argument('-g', '--gamma', default=0.003, type=float, help='The RBF kernel approximation gamma')
svm_parser.add_argument('--dims', default=813, type=int, help='The number of dimensions to use with RBF kernel')
svm_parser.add_argument('-s', '--save', default='SVM_tree.pickle', help='Where to save the pickled data')
att_parser = subparsers.add_parser('att', help='Attack the created SVMs')
att_parser.add_argument('-e', '--epsilon', default=0.08, type=float, help='Aggressiveness of attack')
att_parser.add_argument('-d', '--data', default='SVM_tree.pickle', help='The saved svm to attack')
att_parser.add_argument('--save')
args = parser.parse_args()

training_images = None
test_images = None
training_labels = None
test_labels = None
training_images_flat = None
test_images_flat = None

if args.subparser == 'svm' or args.subparser == 'svm_tune':
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
    svm = SVMTree(training_images_flat.shape[1], list(range(10)), args.lr,
                  dimensions=args.dims, gamma=args.gamma)
    svm.train(training_images_flat, training_labels, args.epochs)
    print(f'Test Accuracy: {round(svm.evaluate(test_images_flat, test_labels) * 100, 2)}%')
    svm.save('SVM_tree.pickle')

if args.subparser == 'svm_tune':
    def objective(trial):
        learning_rate = trial.suggest_uniform('learning_rate', 0.0, 0.1)
        epochs = trial.suggest_int('epochs', 1, 20)
        gamma = trial.suggest_uniform('gamma', 0.0, 0.1)
        dimensions = trial.suggest_int('dimensions', 750, 1000)
        local_svm = SVMTree(training_images_flat.shape[1], list(range(10)), learning_rate,
                            dimensions=dimensions, gamma=gamma)
        return local_svm.train(training_images_flat, training_labels, epochs)

    study = optuna.create_study()
    study.optimize(objective, n_trials=400)
    print(study.best_params)

if args.subparser == 'att':
    test_images = mnist.test_images() / 127.5 - 1
    test_labels = mnist.test_labels()
    adv_ex, bro_ex, grad_ex = stage(test_images, test_labels, args.data, args.epsilon)

    for index, adv in tqdm(enumerate(adv_ex), total=len(adv_ex)):
        fig_creator(test_images[adv[3]], grad_ex[index][0], adv[0], adv[1], adv[2], False, f'samples/{adv[3]}.png')
