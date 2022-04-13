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

    # value that we want to add to the classes in an image (to be more precise avoiding noise)
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

    # If more than 25% of the image is classified as vessel, it is a probable wrong classification -> mask it
    # The loss would be small because it means the vessel in the patch is very small and it is not segmented by K-means
    if prediction == 'vessel' and np.sum(result_patch >= (result_patch.max() - 0)) <= (0.25 * patch_size ** 2):
        lightest_pixel = result_patch.max()  # the ones where it's likely to have vessels
        # Color the final patch in white and black
        result_patch_final = result_patch.copy()
        result_patch_final[result_patch >= lightest_pixel - 1] = 255
        result_patch_final[result_patch < lightest_pixel - 1] = 0
    else:
        result_patch[:, :] = 0  # mask it since it is no-vessel
        result_patch_final = result_patch.copy()

    # debugging purposes
    # if prediction == "vessel":
    #     plt.imshow(patch_image, "gray")
    #     plt.title(file_name)
    #     plt.show()
    #     plt.imshow(result_patch_final, "gray")
    #     plt.title(file_name)
    #     plt.show()

    return result_patch_final
