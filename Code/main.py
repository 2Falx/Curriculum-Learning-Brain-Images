"""
This file will run the classification network with Active Learning or without it, depending on the user's choice.
It also performs clustering with either Canny or K-means and it runs the final segmentation network.
"""
import train


def main():
    train_images_path = "images/skull_stripped_images/"
    train_patches_path = "images/patched_images/train/"
    test_patches_path = "images/patched_images/test/"

    # Get the labels for segmentation through Canny or K-means
    # labels = train.train_whole_dataset(train_patches_path, test_patches_path, method="canny")
    labels = train.train_active_learning(train_patches_path, test_patches_path, num_iterations=10,
                                         metrics="least_confidence", method="canny")
    # train.seg_net(train_images_path, labels)

    print("End")


if __name__ == "__main__":
    main()
