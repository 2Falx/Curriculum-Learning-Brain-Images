from pathlib import Path
from utility import *
import matplotlib.pyplot as plt


def reconstruct(clustered_patches, file_names, tot_images, x_patches_per_image, y_patches_per_image, iterationAL):
    """
    Reconstruct a full 2D image from its patches.
    :param clustered_patches: Numpy array, input patches.
    :param file_names: Numpy array, input file names list.
    :param tot_images: Integer, number of total 2D images.
    :param x_patches_per_image: Integer, number of patches along x-axis of the 2D image.
    :param y_patches_per_image: Integer, number of patches along y-axis of the 2D image.
    :param iterationAL: Integer, active learning iteration.
    :return: List of reconstructed images.
    """
    # Create a numpy array which will contain ordered patches
    # Since full black patches were discarded we initialize this array with full zeros (black)
    patch_size = clustered_patches.shape[-2]
    patches_per_image = x_patches_per_image * y_patches_per_image
    tot_patches = patches_per_image * tot_images
    shape = (tot_patches, patch_size, patch_size, 3)  # shape of list of colored patches
    ordered_patches = np.full(shape, 0)
    # we iterate over the patches name and we put the correspondent clustered image in the array of ordered patches
    for i, file_name in enumerate(file_names):
        # pick the image id and the position
        data = []  # x,y,z,id
        file_name_extract = file_name[:-4].replace("_", " ")
        for word in file_name_extract.split():
            if word.isdigit():
                data.append(int(word))
        offset = 60  # TODO: adjust it in future for multiple NIfTI images
        curr_index = patches_per_image * int((data[-2] - offset) / 5) + data[0] * y_patches_per_image + data[1] + 1
        ordered_patches[curr_index] = clustered_patches[i]

    final_images = np.full((tot_images, int(patch_size * patches_per_image / y_patches_per_image),
                            int(patch_size * patches_per_image / x_patches_per_image), 3), 0)  # images 768x576x3
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

    for i, image in enumerate(final_images):
        plt.imshow(final_images[i])
        plt.show()
        break

    # for i in range(len(final_images)):
    #     Path(f'PatchesAL_brain/image_{i}').mkdir(parents=True, exist_ok=True)
    #     savePath = f"PatchesAL_brain/image_{str(i)}/image_iteration_{str(iterationAL)}.jpg"
    #     createAndSaveImage(final_images[i], savePath)