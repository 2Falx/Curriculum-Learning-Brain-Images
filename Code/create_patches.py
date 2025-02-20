"""
This file applies a grid on the image, saving patches and the image with the grid.
"""
import re
from pathlib import Path
from utils.preprocessing import *
from tqdm import tqdm


def main(patch_size,save_nifti=False):
    
    #Input path generated from the skull_stripping.py script
    masked_images_path = "images/skull_stripped_images/"  # images of brain without the skull
    
    #Created folders
    patches_train_path = "images/patched_images/train/img/"
    patches_test_path = "images/patched_images/test/img/"
    patches_label_train_path = "images/patched_images/train/labels/"
    patches_label_test_path = "images/patched_images/test/labels/"
    grid_path = "images/images_with_grid/"
    
    # Create list of all the images in the labels file (sorted in order to match the patients name)
    input_images = [item for item in sorted(os.listdir(masked_images_path)) if re.search("_img", item)]
    label_images = [item for item in sorted(os.listdir(masked_images_path)) if re.search("_label", item)]

    # Create folders if they don't exist already
    Path(patches_train_path).mkdir(parents=True, exist_ok=True)
    Path(patches_test_path).mkdir(parents=True, exist_ok=True)
    Path(patches_train_path[:-4] + "labels/").mkdir(parents=True, exist_ok=True)  # this will contain true labels
    Path(patches_test_path[:-4] + "labels/").mkdir(parents=True, exist_ok=True)
    Path(patches_train_path[:-4] + "predlabels/").mkdir(parents=True, exist_ok=True)  # this will contain clustered patches
    Path(patches_test_path[:-4] + "predlabels/").mkdir(parents=True, exist_ok=True)
    Path(grid_path).mkdir(parents=True, exist_ok=True)

    for i, img_name in enumerate(tqdm(input_images)):
        #Load image
        label_name = label_images[i]
        img_mat = load_nifti_mat_from_file(masked_images_path + img_name)
        
        #Reshape such that the img can be divided into the passed number of patches (!!!)
        img_mat = reshape_with_patch_size(img_mat, patch_size)
        img_mat_with_grid = img_mat.copy()
        
        # Squashes input between 0.0 and 1.0
        img_mat_with_grid -= img_mat.min()
        img_mat_with_grid /= img_mat.max()
        
        #Load and reshape label
        label_mat = load_nifti_mat_from_file(masked_images_path + label_name)
        label_mat = reshape_with_patch_size(label_mat, patch_size)
        
        # Check that the image and label have the same dimensions and same id
        assert img_mat_with_grid.shape == label_mat.shape
        assert img_name.split("_")[0]==label_name.split("_")[0]
        
        # Compute image dimensions once since all images have the same dimensions and perform a check on the patch size
        if i == 0:
            x_dim, y_dim, z_dim = img_mat.shape
            # Check that patches size fits in the image
            assert x_dim % patch_size == 0 and y_dim % patch_size == 0, "Image shape is not multiple of the patch size"
            # Create the grid
            x_min, y_min = 0, 0
            x_max, y_max = x_dim - 1, y_dim - 1
            # Calculate the number of patches in per image slice
            num_of_x_patches = int((x_max - x_min)/patch_size) + 1
            num_of_y_patches = int((y_max - y_min)/patch_size) + 1
        
        print(f" - img_shape = {img_mat.shape}, patch_size = {patch_size}, num_of_x_patches = {num_of_x_patches}, num_of_y_patches = {num_of_y_patches},")
        
        img_with_grid = img_mat.copy()  # this will contain the image with the grid placed on it, NIfTI format
        label_with_grid = label_mat.copy()  # this will contain the image with the grid placed on it, NIfTI format

        # TODO: change it when the train will be done with all the images (Just put num_selected_slices = z_dim and step=1)
        # NOTE: { BEFORE - 10 slices from 60 to 110 with step 5 (!!!)
        #         NOW - 10 slices from 30 to 80 with step 5 (Central slices of the images) }
        
        num_selected_slices = 10 # Choose the number of selected slices
        
        selected_slices = select_central_elements(np.arange(0, z_dim, step=5), num_selected_slices)
        assert(len(selected_slices==num_selected_slices))
        
        for current_slice in range(z_dim):
            if current_slice not in selected_slices:
                continue
            for m in range(num_of_x_patches):
                for n in range(num_of_y_patches):
                    # Get patch indices
                    patch_start_x = x_min + patch_size * m
                    patch_end_x = x_min + patch_size * (m + 1) - 1
                    patch_start_y = y_min + patch_size * n
                    patch_end_y = y_min + patch_size * (n + 1) - 1
                    
                    # Apply the grid on the NIfTI image and its NIfTI label
                    # TODO: change it adding "i" dimension when the train will be done with all the images
                    for grid in [img_with_grid, label_with_grid]:
                        grid[patch_start_x: patch_end_x, patch_start_y] = 1
                        grid[patch_start_x: patch_end_x, patch_end_y] = 1
                        grid[patch_start_x, patch_start_y: patch_end_y] = 1
                        grid[patch_end_x, patch_start_y: patch_end_y] = 1
                    
                    # Get the patch to be saved and its pixel-level label
                    current_patch = img_mat[patch_start_x: patch_end_x + 1, patch_start_y:patch_end_y + 1, current_slice]
                    current_patch_label = label_mat[patch_start_x: patch_end_x + 1, patch_start_y:patch_end_y + 1, current_slice].astype(np.uint8)

                    # Save with different names patches which contain vessels from those which don't
                    # Note: labels are NIfTI images where 0.0 is black (no-vessel) and 1.0 is white (vessel)
                    if 1 in current_patch_label:
                        label = "vessel"
                    else:
                        label = "no_vessel"

                    # Remove patches which are totally black
                    if np.max(current_patch) != 0:
                        # TODO: change it when the train will be done with all the images (Train/Test split should be done at the patient level)
                        
                        # NOTE: Keep last 2 slices images as test set, save train and test patches in the respective folders
                        # NOTE: save as npy files since NIfTI images' pixels have also values greater than 255
                        
                        if current_slice in selected_slices[-2:]:
                            create_and_save_image_as_ndarray(current_patch, patches_test_path + f"{label}_{m}_{n}_{current_slice}_{re.sub('[^0-9]','', img_name)}")
                            create_and_save_image_as_ndarray(current_patch_label, patches_label_test_path + f"{label}_{m}_{n}_{current_slice}_{re.sub('[^0-9]', '', label_name)}")
                        else:
                            create_and_save_image_as_ndarray(current_patch, patches_train_path + f"{label}_{m}_{n}_{current_slice}_{re.sub('[^0-9]','', img_name)}")
                            create_and_save_image_as_ndarray(current_patch_label, patches_label_train_path + f"{label}_{m}_{n}_{current_slice}_{re.sub('[^0-9]', '', label_name)}")
                        # Name of patch:  vesse/no_vessel + num x patches + num y patches + slice number + patient_number
        
        # Save the NIfTI image and label with grid
        if save_nifti:
            create_and_save_nifti(img_with_grid, grid_path + img_name)
            create_and_save_nifti(label_with_grid, grid_path + label_name)
        print(f" - Patches generated\n")
    print("DONE")

if __name__ == "__main__":
    
    # Choose the patch size
    patch_size = 64
    
    main(patch_size=patch_size)
