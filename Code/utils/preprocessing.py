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
    return nifti_orig.get_fdata()  # transform the images into np.ndarrays - float64


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
    Create a jpg image from numpy array and saves it to given path.
    :param image: Numpy array.
    :param path_target: String, path where to store the created jpg.
    """
    io.imwrite(path_target, image)


def create_and_save_image_as_ndarray(image, path_target):
    """
    Save a Numpy array to a given path in npy format.
    :param image: Numpy array.
    :param path_target: String, path where to store the created jpg.
    """
    np.save(path_target, image)


def compute_number_of_train_images(path):
    """
    Compute the number of images in the given patches' folder.
    :param path: String, path of the patches' folder.
    :return: Integer, number of images in the specified folder.
    """
    file_names = get_all_files(path)[1:]  # ignore desktop.ini
    images_ids = np.array([patch.split("_")[-2] for patch in file_names])
    return np.unique(images_ids).size
