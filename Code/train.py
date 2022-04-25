"""
This is the script for training the classification model.
"""
import keras.callbacks
from utils.preprocessing import get_all_files, load_nifti_mat_from_file, compute_number_of_train_images
from nets.pnet import *
from nets.resnet import *
from nets.vgg import *
from nets.wnetseg import *
from utils.network_train import *
from utils.unsupervised import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from reconstructor import *
import reconstructor_AL
from scipy.stats import entropy


def train_whole_dataset(train_patches_path, test_patches_path, input_images_shape, method):
    """
    Train the classification model using the whole training dataset.
    :param train_patches_path: String, path of train patches.
    :param test_patches_path: String, path of test patches.
    :param input_images_shape: Tuple, input images shape.
    :param method: String, clustering method.
    :return: List, list of label images reconstructed from patches segmented with the specified unsupervised method.
    """
    np.random.seed(42)
    # X, y origin dataset and list of names of files (useful for reconstructing images at the end)
    X, y, file_names = get_X_y_file_names(train_patches_path)
    # Uncomment if you want to try under sampling
    # X, y = random_under_sampling(X, y)
    X_train, y_train, file_names_train = shuffle_data(X, y, file_names)
    X_test, y_test, file_names_test = get_X_y_file_names(test_patches_path)

    # Compute the number of images and patches per image
    patch_size = X[0].shape[0]
    x_patches_per_image = int(input_images_shape[0] / patch_size)
    y_patches_per_image = int(input_images_shape[1] / patch_size)
    tot_images = compute_number_of_train_images(train_patches_path)

    # NOTE: training the whole dataset with the classification network was done to assess performances and comparing
    #       them with active learning. Here we are simulating the case where we have all the labels thanks to the
    #       oracle (human expert). So, if you just want to exploit that annotations (at patch level) and passing them
    #       to the K-means or to Canny methods, you can skip the following CNN.

    # Choose the model
    model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    # model = get_vgg(patch_size)

    print('Training model...')
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "FullModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )

    plot_history(
        history.history["loss"],
        history.history["val_loss"],
        history.history["accuracy"],
        history.history["val_accuracy"],
    )

    y_pred = model.predict(X_test)
    y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

    # Save predictions in a .npy , check them after training with np.load
    np.save("predictions/pnetpred.npy", y_pred_rounded)

    accuracy_score_test = accuracy_score(y_test, y_pred_rounded)
    precision_score_test = precision_score(y_test, y_pred_rounded)
    recall_score_test = recall_score(y_test, y_pred_rounded)
    f1_score_test = f1_score(y_test, y_pred_rounded)

    print(f"Accuracy on test: {accuracy_score_test}")
    print(f"Precision on test: {precision_score_test}")
    print(f"Recall on test: {recall_score_test}")
    print(f"f1 on test: {f1_score_test}")
    print(classification_report(y_test, y_pred_rounded))

    print()
    print('DONE')

    # Path to save labels generated by an unsupervised algorithm
    path_lab = "images/patched_images/train/predlabels/"
    # Choose either kmeans or canny method to get a first approximation of pixel-level labels.
    if method == "kmeans":
        clustered_patches = []
        for index, X_train_sample in enumerate(X_train):
            clustered_patch = kmeans(X_train_sample, y_train[index], viz=True)
            np.save(path_lab + file_names_train[index].split("/")[-1], clustered_patch)
            clustered_patches.append(clustered_patch)
        images_to_rec = np.array(clustered_patches)
    else:
        canny_patches = []
        for index, X_train_sample in enumerate(X_train):
            canny_patch = canny(X_train_sample, y_train[index], viz=False)
            np.save(path_lab + file_names_train[index].split("/")[-1], canny_patch)
            canny_patches.append(canny_patch)
        images_to_rec = np.array(canny_patches)

    # Reconstruct segmented image by patches
    reconstructed_images = reconstruct(images_to_rec, file_names_train, tot_images,
                                       x_patches_per_image, y_patches_per_image)
    return reconstructed_images


def train_active_learning(train_patches_path, test_patches_path, input_images_shape, num_iterations, metrics, method):
    np.random.seed(42)
    X, y, file_names = get_X_y_file_names(train_patches_path)
    # X, y = random_under_sampling(X, y)
    # We start with a 15% of the samples
    X_test_final, y_test_final, _ = get_X_y_file_names(test_patches_path)

    # Compute the number of patches per image
    patch_size = X[0].shape[0]
    x_patches_per_image = int(input_images_shape[0] / patch_size)
    y_patches_per_image = int(input_images_shape[1] / patch_size)

    # Make images appear as 3-channels images to use architecture like VGG, etc.
    X = np.repeat(X[..., np.newaxis], 3, -1)
    X_test_final = np.repeat(X_test_final[..., np.newaxis], 3, -1)

    X_train, X_test, y_train, y_test, \
        file_names_train, file_names_test, X_test_final = shuffle_and_split(X, y, file_names, X_test_final,
                                                                            train_size=0.15)
    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    # Choose the model
    model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    # model = get_vgg(patch_size)

    print("Starting to train... ")
    history = model.fit(
        X_train,
        y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.2,
        callbacks=[keras.callbacks.ModelCheckpoint(
            "ALModelCheckpoint.h5", verbose=1, save_best_only=True
        ),
        ],
    )

    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    #  Active Learning iterations
    for iteration in range(num_iterations):
        y_pred = model.predict(X_test_final)
        y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

        accuracy_score_test = accuracy_score(y_test_final, y_pred_rounded)
        precision_score_test = precision_score(y_test_final, y_pred_rounded)
        recall_score_test = recall_score(y_test_final, y_pred_rounded)
        f1_score_test = f1_score(y_test_final, y_pred_rounded)

        print(f"Accuracy on test: {accuracy_score_test}")
        print(f"Precision on test: {precision_score_test}")
        print(f"Recall on test: {recall_score_test}")
        print(f"f1 on test: {f1_score_test}")
        print(classification_report(y_test_final, y_pred_rounded))

        # Uncertain values count fixed
        count_uncertain_values = 50
        most_uncertain_indices = []

        if metrics == "least_confidence":
            most_uncertain_indices = np.argsort(np.abs(model.predict(X_test) - 0.5), axis=0)
            most_uncertain_indices = most_uncertain_indices[:count_uncertain_values].flatten()

        elif metrics == "entropy":
            entropy_y = np.transpose(entropy(np.transpose(model.predict(X_test))))
            most_uncertain_indices = np.argpartition(-entropy_y, count_uncertain_values - 1, axis=0)[
                                     :count_uncertain_values]

        print(f"X_train.shape: {X_train.shape}")
        print(f"X_test[most_uncertain_indices, :, :, :].shape: {X_test[most_uncertain_indices, :, :, :].shape}")

        # Reconstruct images from patches, highlighting patches selected by active learning and patches of actual train
        # take highlighted elements to show uncertain patches
        X_test_highlighted = X_test.copy()  # we will highlight only patches most uncertain
        X_train_highlighted = X_train.copy()
        for i in range(len(X_train_highlighted)):  # green squares for patches of train
            X_train_highlighted[i, 0, :, 1] = 255
            X_train_highlighted[i, len(X_train_highlighted[i]) - 1, :, 1] = 255
            X_train_highlighted[i, :, 0, 1] = 255
            X_train_highlighted[i, :, len(X_train_highlighted[i]) - 1, 1] = 255

        for i in most_uncertain_indices:
            X_test_highlighted[i, 0, :, 2] = 255  # blue squares for AL patches
            X_test_highlighted[i, len(X_test_highlighted[i]) - 1, :, 2] = 255
            X_test_highlighted[i, :, 0, 2] = 255
            X_test_highlighted[i, :, len(X_test_highlighted[i]) - 1, 2] = 255

        X_check = np.concatenate((X_train_highlighted, X_test_highlighted))
        file_names_check = np.concatenate((file_names_train, file_names_test))
        # reconstruct segmented image by patches
        reconstructor_AL.reconstruct(X_check, file_names_check, iteration)

        # Get most uncertain values from test and add them into the train
        X_train = np.vstack((X_train, X_test[most_uncertain_indices, :, :, :]))
        y_train = np.vstack((y_train, y_test[most_uncertain_indices, :]))
        file_names_train = np.concatenate((file_names_train, file_names_test[most_uncertain_indices]))

        # remove most uncertain values from test
        X_test = np.delete(X_test, most_uncertain_indices, axis=0)
        y_test = np.delete(y_test, most_uncertain_indices, axis=0)
        file_names_test = np.delete(file_names_test, most_uncertain_indices, axis=0)

        # Then I compile again and train again the model
        model.compile(optimizer="SGD",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            X_train,
            y_train,
            epochs=2,
            batch_size=32,
            validation_split=0.2,
            callbacks=[keras.callbacks.ModelCheckpoint(
                "ALModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
            ],
        )

        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

    # End of AL iterations

    # Loading the best model from the training loop
    model = keras.models.load_model("ALModelCheckpoint.h5")

    y_pred = model.predict(X_test_final)
    y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

    accuracy_score_test = accuracy_score(y_test_final, y_pred_rounded)
    precision_score_test = precision_score(y_test_final, y_pred_rounded)
    recall_score_test = recall_score(y_test_final, y_pred_rounded)
    f1_score_test = f1_score(y_test_final, y_pred_rounded)

    print(f"Accuracy on test: {accuracy_score_test}")
    print(f"Precision on test: {precision_score_test}")
    print(f"Recall on test: {recall_score_test}")
    print(f"f1 on test: {f1_score_test}")
    print(classification_report(y_test_final, y_pred_rounded))

    # Now put together train and test and use pass them to kmeans/canny for segmentation
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train.squeeze(), model.predict(
        X_test).squeeze()))  # we generate predictions on the whole dataset -> we will use them in kmeans/canny
    file_names = np.concatenate((file_names_train, file_names_test))

    # Choose either K-means or canny method to get a first approximation of pixel-level labels.
    if method == "kmeans":
        clustered_images = []
        for index, X_train_sample in enumerate(X):
            clustered_images.append(kmeans(X_train_sample, y[index]))
        images_to_rec = np.array(clustered_images)
    else:
        canny_images = []
        for index, X_train_sample in enumerate(X):
            canny_images.append(canny(X_train_sample, y[index]))
        images_to_rec = np.array(canny_images)

    # reconstruct segmented image by patches
    reconstructed_images = reconstruct(images_to_rec, file_names, x_patches_per_image, y_patches_per_image)
    return reconstructed_images

