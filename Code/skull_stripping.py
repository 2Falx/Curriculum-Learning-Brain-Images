"""
This file removes the skull from the MRI images.
"""
import re
from utils.preprocessing import *


def main(original_data_dir, target_dir):
    original_data_dir = os.path.expanduser(original_data_dir)
    target_dir = os.path.expanduser(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Get images, masks and vessel labels lists
    unfiltered_file_list = get_all_files(original_data_dir)
    input_list = [item for item in unfiltered_file_list if re.search('_img', item)]
    mask_list = [item for item in unfiltered_file_list if re.search('_mask', item)]
    label_list = [item for item in unfiltered_file_list if re.search('_label', item)]
    # Load image, mask and label stacks as matrices
    for i, j in enumerate(input_list):
        img_mat = load_nifti_mat_from_file(j)
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        label_mat = load_nifti_mat_from_file(label_list[i])
        # check the dimensions
        assert img_mat.shape == mask_mat.shape == label_mat.shape

        # mask images and labels (skull stripping)
        img_mat = apply_mask(img_mat, mask_mat)
        label_mat = apply_mask(label_mat, mask_mat)
        print(j.split(os.sep)[-1].split('_')[0])
        # save to new file as masked version of original data -> skull stripped brain and vessel labels
        create_and_save_nifti(img_mat, target_dir + j.split(os.sep)[-1].split('_')[0] + '_img.nii')
        create_and_save_nifti(label_mat, target_dir + j.split(os.sep)[-1].split('_')[0] + '_label.nii')

    print('DONE')


if __name__ == '__main__':
    main("images", "skull_stripped_images/")
