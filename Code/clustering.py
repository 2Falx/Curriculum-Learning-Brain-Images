import numpy as np
import cv2


def kmeans(patch_image, prediction):

    # map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'

    norm_image = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    vectorized = np.float32(norm_image.reshape((-1, 3)))
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
    result_patch = res.reshape(norm_image.shape)

    if prediction == 'vessel':
        # First let's convert to gray
        result_patch = cv2.cvtColor(result_patch, cv2.COLOR_RGB2GRAY)
        lightest_pixel = result_patch.max()  # the ones where it's likely to have vessels
        # Color the final patch in white and black
        result_patch_final = result_patch.copy()
        result_patch_final[result_patch >= lightest_pixel - 1] = 255
        result_patch_final[result_patch < lightest_pixel - 1] = 0
    else:
        result_patch = cv2.cvtColor(result_patch, cv2.COLOR_RGB2GRAY)
        result_patch[:, :] = 0  # mask it since it is no-vessel
        result_patch_final = result_patch.copy()

    return result_patch_final
