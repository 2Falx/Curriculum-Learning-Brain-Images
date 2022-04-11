"""
This file contains functions useful for the images' preprocessing step
"""
import os
import numpy as np
import imageio as io
import nibabel as nib


def load_nifti_mat_from_file(path_orig):
    """
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: String, path from where to load the nifti.
    :return: Nifti data as numpy array.
    """
    nifti_orig = nib.load(path_orig)
    print(' - nifti loaded from:', path_orig)
    return nifti_orig.get_data()  # transform the images into np.ndarrays


def create_and_save_nifti(mat, path_target):
    """
    Creates a nifti image from numpy array and saves it to given path.
    :param mat: Numpy array.
    :param path_target: String, path where to store the created nifti.
    """
    new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
    nib.save(new_nifti, path_target)  # save nifti to target dir
    print('New nifti saved to:', path_target)


def apply_mask(image, mask):
    """
    Masks the image with the given mask.
    :param image: Numpy array, image to be masked.
    :param mask: Numpy array, mask.
    :return: Numpy array, masked image.
    """
    masked = image
    masked[np.where(mask == 0)] = 0
    return masked


def get_all_files(directory, files_list=None):
    """
    Get recursively all the files in a directory.
    :param directory: String, path of the directory
    :param files_list: List, files' list in the directory
    """
    if files_list is None:
        files_list = []
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            get_all_files(entry_path, files_list)
        else:
            files_list.append(entry_path)
    files_list = sorted(files_list)
    return files_list


def create_and_save_image(image, path_target):
    """
    Creates a jpg image from numpy array and saves it to given path.
    :param image: Numpy array.
    :param path_target: String, path where to store the created jpg.
    """
    io.imwrite(path_target, image)

