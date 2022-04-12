import numpy as np
import cv2


def canny(patch_image, prediction):
    
    # map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'
    norm_image = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    gray_norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    gray_norm_image = cv2.medianBlur(gray_norm_image, 5)
    edges = cv2.Canny(gray_norm_image, 150, 175)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]
    for c in contours:
        cv2.drawContours(gray_norm_image, [c], -1, (0, 0, 0), -1)

    # Threshold for vessels
    if prediction == 'vessel':
        # make it white and white and black
        darkest_pixel = gray_norm_image.min()  # the ones where it's likely to have vessels
        # keep only black and switch it to white, switch all the rest to black
        # first check if too many pixels are classified as vessels (probably it would be a wrong classification)
        th = 10
        if np.sum(gray_norm_image <= (darkest_pixel + th)) >= (0.25 * patch_image.shape[0] ** 2):  # 1/4 of the patch
            # in this case ignore the patch by masking it
            gray_norm_image[:, :] = 0  # mask it since it is no-vessel
            result_image_final = gray_norm_image.copy()
        else:
            result_image_final = gray_norm_image.copy()
            result_image_final[gray_norm_image <= darkest_pixel + th] = 255
            result_image_final[gray_norm_image > darkest_pixel + th] = 0
    else:
        result_image = gray_norm_image.copy()
        result_image[:, :] = 0  # mask it since it is no-vessel
        result_image_final = result_image.copy()

    return result_image_final
