"""
This file implements the segmentation of the patch through clustering.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def kmeans(patch_image, prediction, file_name):

    patch_size = patch_image.shape[0]

    # map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'

    norm_patch = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_patch = norm_patch.astype(np.uint8)
    blur_patch = cv2.medianBlur(norm_patch, 5)
    vectorized = np.float32(blur_patch.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10

    # Value that we want to add to the classes in an image (to be more precise avoiding noise)
    addition = 1

    if prediction == 'non-vessel':
        K = 1  # Mask it with only one cluster
    else:
        K = 2 + addition  # cluster for vessel, for rest of the eye and additional clusters for specificity

    _, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # Assign to each pixel its centroid
    res = center[label.flatten()]
    result_patch = res.reshape(blur_patch.shape)

    # If more than 15% of the image is classified as vessel, it is a probable wrong classification -> mask it
    # The loss would be small because it means the vessel in the patch is very small and it is not segmented by K-means
    if prediction == 'vessel' and np.sum(result_patch >= (result_patch.max() - 0)) <= 0.15 * patch_size ** 2:
        lightest_pixel = result_patch.max()  # the ones where it's likely to have vessels
        # Color the final patch in white and black
        result_patch_final = result_patch.copy()
        result_patch_final[result_patch >= lightest_pixel - 1] = 255
        result_patch_final[result_patch < lightest_pixel - 1] = 0
    # If the prediction is vessel, but the K-means segmented too much noise, we implement a segmentation by hand
    # considering the lightest pixel and a threshold
    elif prediction == 'vessel':
        myKM_patch = blur_patch.copy()
        myKM_patch[myKM_patch >= blur_patch.max() - 50] = 255  # threshold set to 50
        myKM_patch[myKM_patch < blur_patch.max() - 50] = 0
        # Check if the hand-clustering didn't capture too much noise as well
        if np.sum(myKM_patch == 255) <= 0.15 * patch_size ** 2:
            result_patch_final = myKM_patch
        else:  # Mask it to avoid capturing the noise
            result_patch[:, :] = 0
            result_patch_final = result_patch
    else:
        result_patch[:, :] = 0  # mask it since it is no-vessel
        result_patch_final = result_patch

    # Debugging purposes
    # if prediction == "vessel":
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
    #     axs[0].imshow(patch_image, "gray")
    #     axs[0].set_title("Original patch")
    #     axs[1].imshow(myKM_patch, "gray")
    #     axs[1].set_title("Hand-clustering")
    #     axs[2].imshow(result_patch_final, "gray")
    #     axs[2].set_title("K-means")
    #     plt.show()

    # if prediction == "vessel":
    #     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    #     axs[0].imshow(patch_image, "gray")
    #     axs[0].set_title("Original patch")
    #     axs[1].imshow(result_patch_final, "gray")
    #     axs[1].set_title("Segmentation")
    #     plt.show()

    return result_patch_final
