import numpy as np
import matplotlib.pyplot as plt


def reconstruct(clustered_patches, file_names, x_patches_per_image, y_patches_per_image):

    # Create a numpy array which will contain ordered patches
    # Since full black patches were discarded we initialize this array with full zeros (black)
    patch_size = clustered_patches.shape[-1]
    tot_images = 8  # TODO: automatize it
    patches_per_image = x_patches_per_image * y_patches_per_image
    tot_patches = patches_per_image * tot_images
    shape = (tot_patches, patch_size, patch_size)  # shape of list of gray patches
    ordered_patches = np.full(shape, 0)  # np.zeros(shape)
    # we iterate over the patches name and we put the correspondent clustered image in the array of ordered patches
    for i, file_name in enumerate(file_names):
        # pick the image id and the position
        data = []  # x,y,z,id
        file_name_extract = file_name[:-4].replace("_", " ")
        for word in file_name_extract.split():
            if word.isdigit():
                data.append(int(word))
        ordered_patches[patches_per_image * (data[-1] - 1) + data[0]*y_patches_per_image + data[1] + 1] = clustered_patches[i]  # cv2.cvtColor(clustered_images[i], cv2.COLOR_RGB2GRAY)

    final_images = np.zeros((tot_images, int(patch_size * patches_per_image / y_patches_per_image),
                            int(patch_size * patches_per_image / x_patches_per_image)))  # 8 images 768x576
    for iteration in range(tot_images):
        for i in range(x_patches_per_image):  # along the rows
            for j in range(y_patches_per_image):  # along the columns
                if j == 0:
                    toAttachH = ordered_patches[iteration * patches_per_image + i * x_patches_per_image + j]
                else:
                    toAttachH = np.hstack((toAttachH, ordered_patches[iteration * patches_per_image + i * x_patches_per_image + j]))
            if i == 0:
                toAttachV = toAttachH
            else:
                toAttachV = np.vstack((toAttachV, toAttachH))
        final_images[iteration] = toAttachV

    # for image in final_images:
    #    plt.imshow(image,"gray")
    #    plt.show()
    return final_images
