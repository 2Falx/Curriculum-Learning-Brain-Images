"""
This file contains useful functions for the networks training steps
"""
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
from utils.preprocessing import get_all_files
from keras import backend as K
import cv2

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
    file_list = get_all_files(path)[1:]  # Ignore desktop.ini (shared project on Drive)
    patch_size = np.load(file_list[0]).shape[0]
    X = np.empty((len(file_list), patch_size, patch_size))
    y = np.full((len(X),), 0)
    for i, current_patch_path in enumerate(file_list):
        X[i] = np.load(current_patch_path)
        if "no_vessel" not in current_patch_path \
                and "_lab" not in current_patch_path:
            y[i] = 1
    return X.astype(np.uint64), y.astype(np.uint64), file_list


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
    X_vessels = X[np.where(y == 1)[0]]
    X_vessels_augmented = np.empty((3 * X_vessels.shape[0], X_vessels.shape[1], X_vessels.shape[2]))
    i = 0
    for curr_patch in X_vessels:
        curr_X_vessel_rotated = cv2.rotate(curr_patch, cv2.ROTATE_90_CLOCKWISE)
        X_vessels_augmented[i] = curr_X_vessel_rotated
        i += 1
        curr_X_vessel_rotated = cv2.rotate(curr_patch, cv2.ROTATE_180)
        X_vessels_augmented[i] = curr_X_vessel_rotated
        i += 1
        curr_X_vessel_rotated = cv2.rotate(curr_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        X_vessels_augmented[i] = curr_X_vessel_rotated
        i += 1

    y_vessels_augmented = np.array([1] * len(X_vessels_augmented))
    X = np.concatenate((X, X_vessels_augmented))
    y = np.concatenate((y, y_vessels_augmented))
    return X, y


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


def shuffle_data(X, y, file_names=None):
    """
    Shuffle the input data keeping the mapping with the file names.
    :param X: Numpy array, input data.
    :param y: Numpy array, input labels.
    :param file_names: Numpy array, input file names, optional since not used in curriculum learning.
    :return: Numpy array, shuffled input data.
             Numpy array, shuffled input labels.
             Numpy array, shuffled input file names if not curriculum learning.
    """
    indices = np.array(range(len(y)))
    np.random.seed(42)
    np.random.shuffle(indices)
    if file_names:
        file_names = np.array(file_names)
        return X[indices], y[indices], file_names[indices]
    return X[indices], y[indices]


# used for active learning
def shuffle_and_split(X, y, file_names, X_test_final, train_size,normalize=False):
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
    # Find indices to split the data into train and test sets
    indices = np.array(range(len(y)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    indices_train = np.random.choice(len(y), size=int(train_size*len(y)), replace=False)
    indices_test = np.setxor1d(indices, indices_train)
    
    # Select the training data
    X_train = X[indices_train]
    y_train = y[indices_train]
    file_names_train = np.array(file_names)[indices_train]
    
    #Select the test data (NOTE: Problem here)
    
    X_test = X[indices_test]
    y_test = y[indices_test]

    file_names_test = np.array(file_names)[indices_test]

    # You can choose whether to normalize the input data or not
    if normalize:
        train_mean = np.mean(X)  # mean for data centering
        train_std = np.std(X)  # std for data normalization
        
        X_train -= train_mean
        X_train /= train_std
        
        X_test -= train_mean
        X_test /= train_std
        
        X_test_final -= train_mean
        X_test_final /= train_std
    
    return X_train, X_test, y_train, y_test, file_names_train, file_names_test, X_test_final


def dice_coef(y_true, y_pred, smooth=0):
    """
    Computes the DICE coefficient, also known as F1-score or F-measure.
    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :param smooth: Smoothing factor.
    :return: DICE coefficient of the positive class in binary classification.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Computes the DICE loss function value.
    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :return: Negative value of DICE coefficient of the positive class in binary classification.
    """
    return -dice_coef(y_true, y_pred, 1)
