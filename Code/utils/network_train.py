"""
This file contains useful functions for the networks training steps
"""
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
from utils.preprocessing import get_all_files


def append_history(losses, val_losses, accuracies, val_accuracies, history):
    """
    Append training and validation's history of loss and accuracy.
    :param losses: List, current training loss history.
    :param val_losses: List, current validation loss history.
    :param accuracies: List, current training accuracy history.
    :param val_accuracies: List, current training validation accuracy history.
    :param history: History, allows to access specific history loss or accuracy value.
    :return: Full history list of loss, validation loss, accuracy and validation accuracies.
    """
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracies = accuracies + history.history["accuracy"]
    val_accuracies = val_accuracies + history.history["val_accuracy"]
    return losses, val_losses, accuracies, val_accuracies


def plot_history(losses, val_losses, accuracies, val_accuracies):
    """
    Display two plots: one with loss and validation loss histories, and another with accuracy and validation
    accuracy histories.
    :param losses: List, current training loss history.
    :param val_losses: List, current validation loss history.
    :param accuracies: List, current training accuracy history.
    :param val_accuracies: List, current training validation accuracy history.
    """
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def load_to_numpy(path):
    """
    Convert an image in a Numpy array.
    :param path: String, path of the image.
    :return: Numpy array, input image as ndarray.
    """
    image = io.imread(path)
    return np.array(image)


def get_X_y_file_names(path):
    """
    Load input data from a path as Numpy arrays.
    :param path: String, input data's path.
    :return: Numpy array, input data.
             Numpy array, input labels.
             Numpy array, input file names.
    """
    file_list = get_all_files(path)
    patch_size = np.load(file_list[0]).shape[0]
    X = np.empty((len(file_list), patch_size, patch_size))
    y = np.full((len(X),), 0)
    for i, current_patch_path in enumerate(file_list):
        X[i] = np.load(current_patch_path)
        if "no_vessel" not in current_patch_path:  # it means there is a vessel in the current patch
            y[i] = 1
    return X, y, file_list


def random_under_sampling(X, y):
    """
    Balance the dataset in terms of classes presence.
    TODO: fix test set cause it should remain unbalanced
    :param X: Numpy array, input data.
    :param y: Numpy array, input labels.
    :return: Numpy array, under sampled input data.
             Numpy array, under sampled input labels.
    """
    indices = np.array(range(len(y)))
    # Non vessels are at the beginning
    indices_non_vessels = np.array(range(len(y[y == 0])))
    indices_vessels = np.random.choice(indices, size=len(y[y == 0]))
    indices_under_sampling = indices_non_vessels.tolist() + indices_vessels.tolist()
    return X[indices_under_sampling], y[indices_under_sampling]


def random_over_sampling(X, y):
    """
    Enlarge the dataset through over sampling the least present class.
    :param X: Numpy array, input data.
    :param y: Numpy array, input labels.
    :return: Numpy array, over sampled input data.
             Numpy array, over sampled input labels.
    """
    y_vessels = y[y == 1]
    y_non_vessels = y[y == 0]
    difference = len(y_vessels) - len(y_non_vessels)
    X_non_vessels = X[np.where(y == 0)[0]]
    X_to_add = X_non_vessels
    for i in range(int(np.floor(difference / len(X_non_vessels)) - 1)):
        X_to_add = np.concatenate((X_to_add, X_non_vessels))
    len_remaining_samples_to_add = difference - len(X_to_add)
    index_remaining_samples_to_add = np.random.choice(X_non_vessels.shape[0], len_remaining_samples_to_add)
    X_to_add = np.concatenate((X_to_add, X_non_vessels[index_remaining_samples_to_add]))
    y_to_add = np.array([0] * len(X_to_add))
    X_over_sampled = np.concatenate((X, X_to_add))
    y_over_sampled = np.concatenate((y, y_to_add.reshape(len(y_to_add), 1)))
    return X_over_sampled, y_over_sampled


def normalize(X):
    """
    Normalize the input.
    :param X: Numpy array, input data.
    :return: Numpy array, normalized input.
             Float, input mean.
             Float, input standard deviation.
    """
    train_mean = np.mean(X)  # mean for data centering
    train_std = np.std(X)  # std for data normalization
    X -= train_mean
    X /= train_std
    return X, train_mean, train_std


def shuffle_data(X, y, file_names):
    """
    Shuffle the input data keeping the mapping with the file names.
    :param X: Numpy array, input data.
    :param y: Numpy array, input labels.
    :param file_names: Numpy array, input file names.
    :return: Numpy array, shuffled input data.
             Numpy array, shuffled input labels.
             Numpy array, shuffled input file names.
    """
    indices = np.array(range(len(y)))
    np.random.shuffle(indices)
    file_names = np.array(file_names)
    return X[indices], y[indices], file_names[indices]


# used for active learning
def shuffle_and_split(X, y, file_names, X_test_final, train_size):
    """
    Shuffle, split and normalize input data.
    :param X: Numpy array, input data.
    :param y: Numpy array, input labels.
    :param file_names: Numpy array, input train file names.
    :param X_test_final: Numpy array, input external test data.
    :param train_size: Float, train set size which range between 0 and 1.
    :return: Numpy array, normalized input train data.
             Numpy array, normalized input test data.
             Numpy array, normalized input train labels.
             Numpy array, normalized input test labels.
             Numpy array, normalized input train file names.
             Numpy array, normalized input test file names.
             Numpy array, normalized input external test data.
    """
    indices = np.array(range(len(y)))
    np.random.shuffle(indices)
    indices_train = np.random.choice(len(y), size=int(train_size*len(y)), replace=False)
    indices_test = np.setxor1d(indices, indices_train)
    X_train = X[indices_train]
    y_train = y[indices_train]
    file_names_train = np.array(file_names)[indices_train]
    X_test = X[indices_test]
    y_test = y[indices_test]
    file_names_test = np.array(file_names)[indices_test]
    # train_mean = np.mean(X)  # mean for data centering
    # train_std = np.std(X)  # std for data normalization
    # X_train -= train_mean
    # X_train /= train_std
    # X_test -= train_mean
    # X_test /= train_std
    # X_test_final -= train_mean
    # X_test_final /= train_std
    return X_train, X_test, y_train, y_test, file_names_train, file_names_test, X_test_final
