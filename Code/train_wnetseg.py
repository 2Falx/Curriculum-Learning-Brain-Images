"""
This is the script for training the wnetseg model.
"""
import time
from utils.preprocessing import get_all_files, compute_number_of_train_images
from nets.wnetseg import *
from utils.network_train import *
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from reconstructor import *


def train(patches_path, patches_label_path):
    """
    Train the segmentation network.
    :param patches_path: String, path of input patches.
    :param patches_label_path: String, path of input patches' labels.
    :return: Trained model.
             Train mean.
             Train standard deviation.
    """
    patches_list = get_all_files(patches_path)
    patches_labels_list = get_all_files(patches_label_path)
    
    X_train, y_train = [], []
    
    for brain_patch, brain_patch_label in zip(patches_list, patches_labels_list):
        X_train.append(np.load(brain_patch))
        y_train.append(np.load(brain_patch_label))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    patch_size = X_train[0].shape[0]

    # Normalization
    mean_train = np.mean(X_train)
    std_train = np.std(y_train)
    X_train -= mean_train
    X_train /= std_train

    # Set hyperparameters
    batch_size = 64
    num_channels = 1
    activation = 'relu'
    final_activation = 'sigmoid'
    optimizer = Adam
    lr = 1e-4
    dropout = 0.1
    num_epochs = 40
    loss = dice_coef_loss
    metrics = [dice_coef, 'accuracy']

    print("Loading model...")
    model = get_wnetseg(patch_size, num_channels, activation, final_activation,
                        optimizer, lr, dropout, loss, metrics)

    # Creating DataGenerator
    # How many times to augment training samples with the ImageDataGenerator per one epoch
    factor_train_samples = 2
    rotation_range = 30
    horizontal_flip = False
    vertical_flip = True
    shear_range = 20
    width_shift_range = 0
    height_shift_range = 0
    
    data_gen_args = dict(rotation_range=rotation_range,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip,
                         shear_range=shear_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         fill_mode='constant')
    
    # Data augmentation
    X_datagen = ImageDataGenerator(**data_gen_args)
    y_datagen = ImageDataGenerator(**data_gen_args)

    # Add 4th dimension for datagen
    X_train = np.expand_dims(X_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    X_datagen.fit(X_train, augment=True, seed=seed)
    y_datagen.fit(y_train, augment=True, seed=seed)

    X_generator = X_datagen.flow(X_train, batch_size=batch_size, seed=seed, shuffle=True)
    y_generator = y_datagen.flow(y_train, batch_size=batch_size, seed=seed, shuffle=True)

    # Combine generators into one which yields image and label
    train_generator = zip(X_generator, y_generator)
    
    # Training model
    start_train = time.time()
    
    print("Training model...")
    model.fit(train_generator,
              steps_per_epoch=factor_train_samples * len(X_train) // batch_size,
              epochs=num_epochs,
              verbose=2,
              shuffle=True)
    
    duration_train = int(time.time() - start_train)
    
    print('training took:',
          (duration_train // 3600) % 60, 'hours',
          (duration_train // 60) % 60, 'minutes',
          (duration_train % 60), 'seconds')

    return model, mean_train, std_train


def predict_test_set(test_patches_path, model, mean_train, std_train):
    """
    Make predictions on test set.
    :param test_patches_path: String, path of input test patches.
    :param model: Keras model, trained segmentation model.
    :param mean_train: Float, mean of the train set.
    :param std_train: Float, standard deviation of the train set.
    """
    # TODO: use true test patches labels
    patches_list = get_all_files(test_patches_path)
    
    X_test = []
    
    for brain_patch in patches_list:
        X_test.append(np.load(brain_patch))
    
    X_test = np.array(X_test)

    # Add 4th dimension
    X_test = np.expand_dims(X_test, axis=3)

    # Normalization using train mean and std
    X_test -= mean_train
    X_test /= std_train

    print("Get predictions...")
    y_pred = model.predict(X_test).squeeze()

    file_names_test = get_all_files(test_patches_path)
    tot_images = compute_number_of_train_images(test_patches_path)
    
    # TODO: automize it
    x_patches_per_image = 9 * 2
    y_patches_per_image = 12 * 2

    # Reconstruct full images by patches segmentation predictions
    print("Reconstructing full images from patches segmentation predictions...")
    reconstructed_images = reconstruct(y_pred, file_names_test, tot_images,
                                       x_patches_per_image, y_patches_per_image, test_flag=True)

    print("Saving reconstructed images...")
    
    Path("predictions/").mkdir(parents=True, exist_ok=True)
    np.save("predictions/test_predictions", reconstructed_images)
