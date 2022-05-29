import re
from pathlib import Path
from utils.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import shutil

# This algorithm subdivide the patches in different stages for training (curriculum learning)
# the first implementation will use as metric the value of the vessels dimension and the value of the black images
def main():

    # load patches data
    patches_train_path = "../images/patched_images/train/img/"
    patches_label_train_path = "../images/patched_images/train/labels/"
    curriculum_train_path = "../images/curriculum/"
    input_images = [item for item in os.listdir(patches_train_path) if re.search("vessel", item)]
    label_images = [item for item in os.listdir(patches_label_train_path) if re.search("vessel", item)]

    Path(curriculum_train_path).mkdir(parents=True, exist_ok=True)

    # I compute the value of the mean of how big the vessels are
    vessels_dimensions = [] # Dimension of the vessels %
    non_vessels_black_dimension = [] # Dimension of black part of the patches

    n_stages = 3
    stages = [[]] * n_stages  # list of lists contains a list of name file for each stage of the curriculum learning
    for i, el in enumerate(input_images):
        img_mat = np.load(patches_train_path + el)
        label_mat = np.load(patches_label_train_path + el)


        # if the patch contains a vessel
        if label_mat.max() != 0:
            mean_vessel = label_mat.mean()
            vessels_dimensions.append(mean_vessel)
            if mean_vessel > 0.05:
                stages[0].append(el)
            elif mean_vessel <= 0.05 and mean_vessel > 0.01:
                stages[1].append(el)
            else:
                stages[2].append(el)

        else:
            # black patch out of the brain
            percentage_of_black_image = np.count_nonzero(img_mat[img_mat == 0])/len(img_mat)
            if percentage_of_black_image < 0.1:
                stages[0].append(el)
            elif percentage_of_black_image < 0.5:
                stages[1].append(el)
            else:
                stages[2].append(el)

    for i, stage in enumerate(stages):
        dir = curriculum_train_path + f"stage_{i}/"
        dirpath = Path(dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dir)  # removes the directory and all the files in it
        Path(dir).mkdir(parents=True, exist_ok=True)
        for to_save in stage:
            img_mat = np.load(patches_train_path + to_save)
            create_and_save_image_as_ndarray(img_mat, dir + to_save)



if __name__=="__main__":
    main()