"""
This algorithm subdivide the patches in different stages for training (curriculum learning)
the first implementation will use as metric the value of the vessels dimension and the value of the black images
"""
import re
from pathlib import Path
from utils.preprocessing import *
import numpy as np
import shutil
from tqdm import tqdm


def main():
    #Input path generated from create_patches.py
    patches_train_path = "images/patched_images/train/img/"
    patches_label_train_path = "images/patched_images/train/labels/"
    
    #Output path
    curriculum_train_path = "images/curriculum/"
    
    input_images = [item for item in os.listdir(patches_train_path) if re.search("vessel", item)]
    label_images = [item for item in os.listdir(patches_label_train_path) if re.search("vessel", item)]

    Path(curriculum_train_path).mkdir(parents=True, exist_ok=True)

    # Compute the value of the mean of how big the vessels are
    vessels_dimensions = []  # Dimension of the vessels %
    non_vessels_black_dimension = []  # Dimension of black part of the patches
    mean_of_values = []
    var_of_values = []
    n_stages = 3 # Number of stages: not used
    stage_0, stage_1, stage_2 = [], [], []
    
    print(f"Assignment of patches to {n_stages} different stages")
    for i, el in enumerate(tqdm(input_images)):
        
        #Load image and label
        img_mat = np.load(patches_train_path + el)
        label_mat = np.load(patches_label_train_path + el)
        
        # compute and store all the pixel-wise mean and variance from the images
        mean_of_values.append(img_mat.mean()) 
        var_of_values.append(img_mat.var()) 

        # If the patch contains a vessel
        if label_mat.max() != 0:
            mean_vessel = label_mat.mean()
            vessels_dimensions.append(mean_vessel)
            
            if mean_vessel > 0.05:
                stage_0.append(el)
            elif 0.05 >= mean_vessel > 0.015:
                stage_1.append(el)
            else:
                stage_2.append(el)

        else:
            # Black patch out of the brain
            percentage_of_black_image = 1 - np.count_nonzero(img_mat)/img_mat.size
            # check percentange of black image (low level of black in the image -> easier), and
            # variance of pixels in the image -> lower variance-> more homogeneity -> easier)
            
            # I CHANGED AGAIN VALUES W.R.T FINAL VERSION division 0.01, 0.5 | 200, 500, 1200
            if percentage_of_black_image < 0.1 and img_mat.var() < 200:
                stage_0.append(el)
            elif percentage_of_black_image < 0.6 and img_mat.var() < 800:
                stage_1.append(el)
            else:
                stage_2.append(el)



    # TODO balance of the dataset based on number of vessels images.
    # sample only the same amount of vessels image to put in the training

    print(f"Saving assigned patches to {curriculum_train_path}/stage_n")
    for i, stage in enumerate(tqdm([stage_0, stage_1, stage_2])):
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
