# Curriculum Learning with Brain Images

## Introduction
The firstaim of the project is the reduction in the number of annotated images for a weakly supervised task of segmentation of retinal vessels, using an active learning framework which provides to the Oracle only the most uncertain images to label.
The second goal of the project is to explore the use of the curriculum learning to improve the performance of the model by learning through different stage based on the *difficulty* of the labels
The brain images used in these experiments are taken from a private dataset. 

## Index
1. Preprocessing
2. Classification task to identify the presence of vessels inside the images (with Active or Curriculum learning framework)
3. Clustering model (k-means) to identify generate weak-label of the vessel-patches at pixel-level
4. Segmentation task on the images previously identified as vessels

## Preprocessing 
In the preprocessing phase, there is a preliminary extraction of some slices from 3D brain images and a second retrieval of patches from the input images. Only images with at least a partial not-black background are consider in the training, the others are simply neglected.
<p align="center">
  <img src="readme_imgs/Overview.jpg" alt="Image of preprocessing"/>
</p> 

## Classification task with active learning framework
We used an Active Learning technique on the classifier that consists in selecting the most useful samples from the unlabeled dataset and send them to an oracle for the annotation.
As a classification network, we mainly used the PNET architecture, but we also tried well known networks as ResNet50 and VGG16, both pre-trained on ImageNet dataset and fine-tuned on eye’s images.
As uncertainty measures we implemented Least Confidence and Entropy, that both returns the sample on which the classifier
is more uncertain on.
Data augmentation is also applied to the images.
<p align="center">
  <img src="readme_imgs/AL-Framework.jpg" alt="Image of classification"/>
</p> 

In order to obtain a first approximation of pixel-level labels we used K-means clustering algorithm and OpenCV Canny method.

## Classification task with Curriculum learning framework
We also used a Curriculum Learning technique on the classifier that consists in different stages of training based on the difficulty of the labels, as we can see in the following image.

<p align="center">
  <img src="readme_imgs/CL-Framework.jpg" alt="CL Framework"/>
</p> 

As before, for the different stages of training we used a VGG16 model and K-means/Canny clustering algorithm to obtain a first approximation of pixel-level labels.

## Segmentation task
As segmentation network we exploited a Unet:
in particular we used two 2D-Unets in cascade which was extremely useful since the first Unet will recover rough-mask
labels coming from Canny and K-means and the second Unet will produce better segmentations thanks to the output of the
first Unet.

<p align="center">
  <img src="readme_imgs/DUnet.jpg" alt="Double U-Net"/>
</p> 

Here is an example of a segmented brain image.

<p align="center">
  <img src="readme_imgs/SegClust.jpg" alt="SegClust"/>
</p> 


## Train (TODO: Modify snippets)

### 0) Optional - Create and Activate virtual environment (using Conda)
```bash
  $ conda create --name RetinalAL python=3.8
  $ conda activate RetinalAL
```

### 1) Install required packages from requirements.txt
```bash
  $ pip install -r requirements.txt
```

You should have all yout input images (n_ToF), brain (n_mask) and vessel (n_vessel) mask in a folder images/original_images 

### 2) Skull Stripping
```bash
  $ python3 Code/skull_stripping.py
```

### 3) Create patched images
```bash
  $ python3 Code/create_patches.py
```

### 4) Split the training dataset into 3 stages by the *difficulty* of the patches (curriculum subdivision)
```bash
  $ python3 Code/curriculum_subdivision.py
```

### 3) Start training
```bash
  $ python3 Code/models_training.py
```
