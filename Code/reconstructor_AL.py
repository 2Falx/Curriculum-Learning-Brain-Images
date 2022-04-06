import matplotlib.pyplot as plt
from pathlib import Path
from utility import *

def reconstruct(clustered_images, file_names, iterationAL):
    # create a numpy array which will contain ordered patches
    # since we discarded some black patches we initialize this array with full zeros (black)
    # originally we had 19*19 = 361 patches (32x32 size) per image, and we have 16 images in train so:
    clustered_images = clustered_images.astype(int)
    shape = (5776, 32, 32, 3)
    ordered_patches = np.full(shape, 0)
    # we iterate over the patches name and we put the correspondant clustered image in the array of ordered patches
    for i,file_name in enumerate(file_names):
        patch_image = clustered_images[i]
        # if it is a data of the additional dataset we skip it -> we want to reconstruct original data only
        if file_name == '0.0':
            continue
        # pick the image id and the position
        data = []  # x,y,id
        file_name_extract = file_name.replace("_", " ")
        for word in file_name_extract.split():
            if word.isdigit(): data.append(int(word))
        ordered_patches[361 * (data[2]-21) + data[0]*19 + data[1]] = patch_image

    final_images = np.full((16,608,608,3), 0)  # 16 colored images 608x608
    for iteration in range(16):
        for i in range(19): # along the rows
            for j in range(19):  # along the columns
                if j == 0:
                    toAttachH = ordered_patches[iteration*361 + i*19 + j]
                else:
                    toAttachH = np.hstack((toAttachH, ordered_patches[iteration*361 + i*19 + j]))
            if i == 0:
                toAttachV = toAttachH
            else:
                toAttachV = np.vstack((toAttachV, toAttachH))
        final_images[iteration] = toAttachV

    for i in range(len(final_images)):
        Path(f'PatchesAL/image_{i}').mkdir(parents=True, exist_ok=True)
        savePath = f"PatchesAL/image_{str(i)}/image_iteration_{str(iterationAL)}.jpg"
        createAndSaveImage(final_images[i], savePath)

    # for image in final_images:
    plt.imshow(final_images[0])
    plt.show()
    return final_images