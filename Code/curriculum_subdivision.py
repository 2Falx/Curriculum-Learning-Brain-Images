"""
This algorithm subdivide the patches in different stages for training (curriculum learning)
the first implementation will use as metric the value of the vessels dimension and the value of the black images
"""
import re
from pathlib import Path
from utils.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import shutil


def main():
    patches_train_path = "images/patched_images/train/img/"
    patches_label_train_path = "images/patched_images/train/labels/"
    curriculum_train_path = "images/curriculum/"
    input_images = [item for item in os.listdir(patches_train_path) if re.search("vessel", item)]
    label_images = [item for item in os.listdir(patches_label_train_path) if re.search("vessel", item)]

    Path(curriculum_train_path).mkdir(parents=True, exist_ok=True)

    # I compute the value of the mean of how big the vessels are
    vessels_dimensions = []  # Dimension of the vessels %
    non_vessels_black_dimension = []  # Dimension of black part of the patches

    n_stages = 3
    stage_0, stage_1, stage_2 = [], [], []
    for i, el in enumerate(input_images):
        img_mat = np.load(patches_train_path + el)
        label_mat = np.load(patches_label_train_path + el)

        # If the patch contains a vessel
        if label_mat.max() != 0:
            mean_vessel = label_mat.mean()
            vessels_dimensions.append(mean_vessel)
            if mean_vessel > 0.05:
                stage_0.append(el)
            elif 0.05 >= mean_vessel > 0.01:
                stage_1.append(el)
            else:
                stage_2.append(el)

        else:
            # Black patch out of the brain
            percentage_of_black_image = np.count_nonzero(img_mat[img_mat == 0])/len(img_mat)
            if percentage_of_black_image < 0.1:
                stage_0.append(el)
            elif percentage_of_black_image < 0.5:
                stage_1.append(el)
            else:
                stage_2.append(el)

    for i, stage in enumerate([stage_0, stage_1, stage_2]):
        dir = curriculum_train_path + f"stage_{i}/"
        dirpath = Path(dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dir)  # Removes the directory and all the files in it
        Path(dir).mkdir(parents=True, exist_ok=True)
        for to_save in stage:
            img_mat = np.load(patches_train_path + to_save)
            create_and_save_image_as_ndarray(img_mat, dir + to_save)


if __name__ == "__main__":
    main()
