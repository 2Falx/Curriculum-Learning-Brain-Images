"""
This file contains functions useful for the images' preprocessing step
"""
import os
import numpy as np
import imageio as io
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize


def load_nifti_mat_from_file(path_orig):
    """
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: String, path from where to load the nifti.
    :return: Nifti data as numpy array.
    """
    nifti_orig = nib.load(path_orig)
    print(' - nifti loaded from:', path_orig)
    return nifti_orig.get_fdata()  # transform the images into np.ndarrays - float64

def reshape_with_patch_size(img_data, patch_size):
    """
    Reshapes the image data so that its new dimension is a multiple of the given patch size.
    :param img_data: Numpy array, image data to be reshaped.
    :param patch_size: Integer, patch size.
    :return: Numpy array, reshaped image data.
    """
    old_shape = img_data.shape
    old_x_size = old_shape[0]
    old_y_size = old_shape[1]
    old_z_size = old_shape[2]
    
    new_x_size = (old_x_size//patch_size) * patch_size
    new_y_size = (old_y_size//patch_size) * patch_size
    new_z_size = old_z_size
    
    new_shape = (new_x_size, new_y_size, new_z_size)
    resized_img_data = resize(img_data, new_shape, 3, cval=0, mode='edge', anti_aliasing=False) if new_shape != img_data.shape else img_data 
    return resized_img_data

def select_central_elements(slice_list, n):
    list_length = len(slice_list)
    
    if n >= list_length:
        return slice_list
    
    half_n = int(n/2)
    
    if list_length % 2 == 0:
        # list has even number of elements
        start_index = int(list_length / 2) - half_n
        end_index = start_index + n
    else:
        # list has odd number of elements
        start_index = int(list_length / 2) - half_n
        end_index = start_index + n
        
    return slice_list[start_index:end_index]

def plot_nifty(img_data):
    """
    Plots the image data.
    :param img_data: Numpy array, image data to be plotted.
    """
    plt.imshow(img_data[:,:,img_data.shape[2]//2], cmap='gray')
    plt.show()

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
