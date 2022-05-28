"""
This file contains functions to perform the patch segmentation.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def kmeans(patch_image, prediction, viz=False):
    """
    Perform K-means on a given patch. K is chosen depending on the prediction on vessel's presence in the patch.
    :param patch_image: Numpy array, input patch.
    :param prediction: Integer, prediction's label, 0 if non-vessel, 1 if vessel.
    :param viz: Boolean, default False, if set to True it will display K-means result on each patch with a predicted vessel.
    """
    patch_size = patch_image.shape[0]
    # Map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'

    norm_patch = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_patch = norm_patch.astype(np.uint8)
    blur_patch = cv2.medianBlur(norm_patch, 5)
    vectorized = np.float32(blur_patch.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    # Value that we want to add to the classes in an image (to be more precise avoiding noise)
    addition = 1

    # K-means clustering
    if prediction == 'non-vessel':
        K = 1  # Mask it with only one cluster
    else:
        K = 2 + addition  # cluster for vessel, for rest of the eye and additional clusters for specificity
    _, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # Assign to each pixel its centroid
    res = center[label.flatten()]
    result_patch = res.reshape(blur_patch.shape)

    # Define threshold - clustering tuning
    th_noise = 0.15  # threshold on the noise captured
    th_lightest = 1  # threshold on the lightest pixel
    th_manual = 50  # threshold on white pixels, hand clustering

    # If more than th_noise of the image is classified as vessel, it is a probable wrong classification -> mask it
    # The loss would be small because it means the vessel in the patch is very small and it is not segmented by K-means
    if prediction == 'vessel' and np.sum(result_patch >= (result_patch.max() - 0)) <= th_noise * patch_size ** 2:
        lightest_pixel = result_patch.max()  # the ones where it's likely to have vessels
        # Color the final patch in white and black
        result_patch_final = result_patch.copy()
        result_patch_final[result_patch >= lightest_pixel - th_lightest] = 255
        result_patch_final[result_patch < lightest_pixel - th_lightest] = 0
    # If the prediction is vessel, but the K-means segmented too much noise, we implement a segmentation by hand
    # considering the lightest pixel and a threshold
    elif prediction == 'vessel':
        myKM_patch = blur_patch.copy()
        myKM_patch[myKM_patch >= blur_patch.max() - th_manual] = 255
        myKM_patch[myKM_patch < blur_patch.max() - th_manual] = 0
        # Check if the hand-clustering didn't capture too much noise as well
        if np.sum(myKM_patch == 255) <= th_noise * patch_size ** 2:
            result_patch_final = myKM_patch
        else:  # Mask it to avoid capturing the noise
            result_patch[:, :] = 0
            result_patch_final = result_patch
    else:
        result_patch[:, :] = 0  # mask it since it is no-vessel
        result_patch_final = result_patch
    # If you want to visualize set the function param to True

    return result_patch_final


def canny(patch_image, prediction, viz=False):
    """
    Apply OpenCV's Canny method on a given patch.
    :param patch_image: Numpy array, input patch.
    :param prediction: Integer, prediction's label, 0 if non-vessel, 1 if vessel.
    :param viz: Boolean, default False, if set to True it will display K-means result on each patch with a predicted vessel.
    """
    patch_size = patch_image.shape[0]
    # Map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'
    norm_patch = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_patch = norm_patch.astype(np.uint8)
    blur_patch = cv2.medianBlur(norm_patch, 5)
    edges = cv2.Canny(blur_patch, 150, 175)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]
    for c in contours:
        cv2.drawContours(blur_patch, [c], -1, (0, 0, 0), -1)

    # Define threshold - clustering tuning
    th_noise = 0.15  # threshold on the noise captured
    th_lightest = 1  # threshold on the lightest pixel

    # If more than th_noise of the image is classified as vessel, it is a probable wrong classification -> mask it
    # The loss would be small because it means the vessel in the patch is very small and it is not segmented by K-means
    if prediction == 'vessel' and np.sum(blur_patch >= (blur_patch.max() - 0)) <= th_noise * patch_size ** 2:
        lightest_pixel = blur_patch.max()  # the ones where it's likely to have vessels
        # Color the final patch in white and black
        result_patch_final = blur_patch.copy()
        result_patch_final[blur_patch >= lightest_pixel - th_lightest] = 255
        result_patch_final[blur_patch < lightest_pixel - th_lightest] = 0

    else:
        blur_patch[:, :] = 0  # mask it since it is no-vessel
        result_patch_final = blur_patch

    # If you want to visualize set the function param to True
    if prediction == "vessel" and viz:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        axs[0].imshow(patch_image, "gray")
        axs[0].set_title("Original patch")
        axs[1].imshow(result_patch_final, "gray")
        axs[1].set_title("Segmentation")
        plt.show()

    return result_patch_final
