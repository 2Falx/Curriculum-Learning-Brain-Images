"""
This file reconstruct the full image from its patches.
"""
import nibabel
import numpy as np
import matplotlib.pyplot as plt


def reconstruct(clustered_patches, file_names, tot_images, x_patches_per_image, y_patches_per_image, test_flag=False):
    """
    Reconstruct a full 2D image from its patches.
    :param clustered_patches: Numpy array, input patches.
    :param file_names: Numpy array, input file names list.
    :param tot_images: Integer, number of total 2D images.
    :param x_patches_per_image: Integer, number of patches along x-axis of the 2D image.
    :param y_patches_per_image: Integer, number of patches along y-axis of the 2D image.
    :param test_flag: Boolean, optional, adjust the reconstructor in case of train or test.
    :return: List of reconstructed images.
    """
    # Create a numpy array which will contain ordered patches
    # Since full black patches were discarded we initialize this array with full zeros (black)
    patch_size = clustered_patches.shape[-1]
    patches_per_image = x_patches_per_image * y_patches_per_image
    tot_patches = patches_per_image * tot_images
    shape = (tot_patches, patch_size, patch_size)  # shape of list of gray patches
    ordered_patches = np.zeros(shape)  # np.full(shape, 0)
    # we iterate over the patches name and we put the correspondent clustered image in the array of ordered patches
    for i, file_name in enumerate(file_names):
        # pick the image id and the position
        data = []  # x,y,z,id
        file_name_extract = file_name[:-4].replace("_", " ")
        for word in file_name_extract.split():
            if word.isdigit():
                data.append(int(word))
        offset = 60 + 40 * test_flag  # TODO: adjust it in future for multiple NIfTI images
        curr_index = patches_per_image * int((data[-2] - offset) / 5) + data[0] * y_patches_per_image + data[1] + 1
        ordered_patches[curr_index] = clustered_patches[i]

    final_images = np.zeros((tot_images, int(patch_size * patches_per_image / y_patches_per_image),
                            int(patch_size * patches_per_image / x_patches_per_image)))  # images 768x576
    for iteration in range(tot_images):
        for i in range(x_patches_per_image):  # along the rows
            for j in range(y_patches_per_image):  # along the columns
                if j == 0:
                    toAttachH = ordered_patches[iteration * patches_per_image + i * y_patches_per_image + j]
                else:
                    toAttachH = np.hstack((toAttachH, ordered_patches[iteration * patches_per_image + i * y_patches_per_image + j]))
            if i == 0:
                toAttachV = toAttachH
            else:
                toAttachV = np.vstack((toAttachV, toAttachH))
        final_images[iteration] = toAttachV

    # for i, image in enumerate(final_images):
    #     original_image = nibabel.load("images/skull_stripped_images/brain2_img.nii").get_fdata()[:, :, 60 + 5 * i]
    #     label = nibabel.load("images/skull_stripped_images/brain2_label.nii").get_fdata()[:, :, 60 + 5 * i]
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
    #     axs[0].imshow(original_image, "gray")
    #     axs[0].set_title("Original image")
    #     axs[1].imshow(image, "gray")
    #     axs[1].set_title("Reconstructed image")
    #     axs[2].imshow(label, "gray")
    #     axs[2].set_title("Ground truth")
    #     plt.show()

    return final_images
