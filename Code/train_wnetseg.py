"""
This is the script for training the wnetseg model.
"""
import time
from utils.preprocessing import get_all_files
from nets.wnetseg import *
from utils.network_train import *
from utils.unsupervised import *
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def train(patches_path, patches_label_path):
    patches_list = get_all_files(patches_path)
    patches_labels_list = get_all_files(patches_label_path)
    patch_size = 64
    X_train, y_train = [], []
    for brain_patch, brain_patch_label in zip(patches_list, patches_labels_list):
        X_train.append(np.load(brain_patch))
        y_train.append(np.load(brain_patch_label))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Set hyperparameters
    batch_size = 64
    num_channels = 1
    activation = 'relu'
    final_activation = 'sigmoid'
    optimizer = Adam
    lr = 1e-4
    dropout = 0.1
    num_epochs = 10
    loss = dice_coef_loss
    metrics = [dice_coef, 'accuracy']

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
    model.fit(train_generator,
              steps_per_epoch=factor_train_samples * len(X_train) // batch_size,
              epochs=num_epochs,
              verbose=2, shuffle=True)
    duration_train = int(time.time() - start_train)
    print('training took:', (duration_train // 3600) % 60, 'hours', (duration_train // 60) % 60,
          'minutes', duration_train % 60, 'seconds')
