import mnist
import argparse
from SVM_model import SVMTree
from fgsm import stage
from utils import fig_creator

parser = argparse.ArgumentParser(description='Create an SVM and Break it using an FGSM attack')
subparsers = parser.add_subparsers(dest='subparser')
svm_parser = subparsers.add_parser('svm', help='Create and train SVM for classifing mnist')
svm_parser.add_argument('-l', '--learning_rate', default=0.007, type=float, help='The learning rate')
svm_parser.add_argument('-e', '--epochs', default=6, type=int, help='The number of epochs to train for')
svm_parser.add_argument('-s', '--save', default='SVM_tree.pickle', help='Where to save the pickled data')
att_parser = subparsers.add_parser('att', help='Attack the created SVMs')
att_parser.add_argument('-e', '--epsilon', default=0.08, type=float, help='Aggressiveness of attack')
att_parser.add_argument('-d', '--data', default='SVM_tree.pickle', help='The saved svm to attack')
att_parser.add_argument('--save')
args = parser.parse_args()

if args.subparser is None:
    parser.print_help()
    exit(0)

if args.subparser == 'svm':
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

    # Create svm tree
    svm = SVMTree(training_images_flat.shape[1], list(range(10)), args.learning_rate)

    # Train svm
    svm.train(training_images_flat, training_labels, args.epochs)
    svm.save_tree(args.save)
    print(f'Test Accuracy: {round(svm.evaluate(test_images_flat, test_labels) * 100, 2)}%')

if args.subparser == 'att':
    test_images = mnist.test_images() / 127.5 - 1
    test_labels = mnist.test_labels()
    adv_ex, bro_ex, grad_ex = stage(test_images, test_labels, args.data, args.epsilon)

    fig_creator(test_images[adv_ex[0][3]], grad_ex[0][0], adv_ex[0][0], adv_ex[0][1], adv_ex[0][2], True,
                'example.png')
