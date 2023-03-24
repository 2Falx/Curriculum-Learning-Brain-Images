"""
This file will run the classification network with Active Learning or without it, depending on the user's choice.
It also performs clustering with either Canny or K-means and it runs the final segmentation network.
"""
import train
import train_wnetseg


def main():
    train_patches_path = "images/patched_images/train/img/"
    test_patches_path = "images/patched_images/test/img/"
    curriculum_patches_path = "images/curriculum/"
    train_patches_labels_path = "images/patched_images/train/labels/"
    test_patches_labels_path = "images/patched_images/test/labels/"
    input_images_shape = (576, 768, 136)  # NIfTI images
    
    pipeline_name = "active"  # "curriculum_learning" or "active_learning"
    model_name = "vgg"
    do_segmentation = True # True or False
    predict_segmentation_bool = False # True or False
    predict_segmentation = False if do_segmentation is True else predict_segmentation_bool
    
    
    # Get the labels for segmentation through Canny or K-means
    if "curriculum" in pipeline_name:
        print("Running curriculum learning Training...")
        labels = train.train_curriculum_dataset(curriculum_patches_path, train_patches_path, test_patches_path,
                                                input_images_shape, method="kmeans")
    
    elif "active" in pipeline_name:
        print("Running active learning Training...")
        labels = train.train_active_learning(train_patches_path, test_patches_path, input_images_shape,
                                            num_iterations=20, uncertainty_metric="least_confidence", method="kmeans",
                                            model_name=model_name, do_undesampling=False,
                                            )
    else:
        print("Starting whole dataset training...")
        print("Please choose a valid pipeline name: active_learning or curriculum to change the training pipeline.")
        labels = train.train_whole_dataset(train_patches_path, test_patches_path, input_images_shape, method="kmeans",model_name=model_name)
        
    if do_segmentation:
        model, mean_train, std_train = train_wnetseg.train(train_patches_path, train_patches_labels_path)
        # TODO: return the model from wnetseg and test it on test images (possibly take them full shape and create patches on the flight)
        if predict_segmentation:
            train_wnetseg.predict_test_set(test_patches_path, model, mean_train, std_train)
    print("End")


if __name__ == "__main__":
    main()
