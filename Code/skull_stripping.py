"""
This file removes the skull from the MRI images.
"""
import re
from utils.preprocessing import *


def main(original_data_dir, target_dir):
    
    # Input data directory
    original_data_dir = os.path.expanduser(original_data_dir)
    
    #Created target directory
    target_dir = os.path.expanduser(target_dir)
    
    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Get images, masks and vessel labels lists
    unfiltered_file_list = get_all_files(original_data_dir)
    
    input_list = [item for item in unfiltered_file_list if re.search('_img|_ToF', item)]
    mask_list = [item for item in unfiltered_file_list if re.search('_mask', item)]
    label_list = [item for item in unfiltered_file_list if re.search('_label|_vessel', item)]
    
    assert len(input_list) == len(mask_list) == len(label_list)
    
    # Load image, mask and label stacks as matrices
    for i, img_name in enumerate(input_list):
        
        mask_name = mask_list[i]
        label_name = label_list[i]
        #Check if img, brain mask and vessel label belong to the same patient
        assert img_name.split('_')[0] == mask_name.split('_')[0] == label_name.split('_')[0]
        #Load brain image
        img_mat = load_nifti_mat_from_file(img_name)
        #ULoad and use the brain mask to remove the skull
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        #Load the vessel label
        label_mat = load_nifti_mat_from_file(label_list[i])
        
        # check the dimensions  
        assert img_mat.shape == mask_mat.shape == label_mat.shape

        # mask images and labels by using the brain labell as mask(skull stripping)
        img_mat = apply_mask(img_mat, mask_mat)
        label_mat = apply_mask(label_mat, mask_mat)
        
        #print(img_name.split(os.sep)[-1].split('_')[0])
        
        # save to new file as masked version of original data -> skull stripped brain and vessel labels
        create_and_save_nifti(img_mat, target_dir + img_name.split('/')[-1].split('_')[0] + '_img.nii')
        create_and_save_nifti(label_mat, target_dir + label_name.split('/')[-1].split('_')[0] + '_label.nii')
        print()
        
    print('DONE')


if __name__ == '__main__':
    main("images/original_images/", "images/skull_stripped_images/")
