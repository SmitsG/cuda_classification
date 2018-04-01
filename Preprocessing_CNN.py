import os
import random

import cv2
import numpy as np


def main():
    sick_healthy_dict = get_image_titles()

    # Get the names of the images and convert the dict.keys object to a list
    dataset = list(sick_healthy_dict.keys())

    # Set the size (as fraction) for the training set and the validation set. The remaining samples will be used for testing
    training_set_size = (1 / 2)
    validation_set_size = (1 / 4)

    # Calculate the absolute number of training and validation instances
    number_training_instances = calculate_number_of_instances(dataset, training_set_size)
    number_validation_instances = calculate_number_of_instances(dataset, validation_set_size)

    # Create the training, validation and testing sets
    training_data, dataset = get_random_instances(dataset, number_training_instances)
    validation_data, dataset = get_random_instances(dataset, number_validation_instances)
    testing_data = dataset

    # Create lists with the training, validating and testing labels
    training_labels_list = create_labels_list(training_data, sick_healthy_dict)
    validating_labels_list = create_labels_list(validation_data, sick_healthy_dict)
    testing_labels_list = create_labels_list(testing_data, sick_healthy_dict)



    training_path = "/home/silvie/Pictures/Tuberculose_images_divided/Train"
    validating_path = "/home/silvie/Pictures/Tuberculose_images_divided/Valid"
    testing_path = "/home/silvie/Pictures/Tuberculose_images_divided/Test"
    image_origin = "/home/silvie/Pictures/ChinaSet_AllFiles/CXR_png/"
    write_images_to_folder(training_data, sick_healthy_dict, training_path, image_origin)
    print("Done training images")
    write_images_to_folder(validation_data, sick_healthy_dict, validating_path, image_origin)
    print("Done validating images")
    write_images_to_folder(testing_data, sick_healthy_dict, testing_path, image_origin)
    print("Done testing images")


# This function retrieves a list of file and directory namen from the current working directory
# This function filters the filenames out
# This function checks if a filename ends with a certain string
# This function creates a dictionary key=image title and value=image class
def get_image_titles():
    mypath = "/home/silvie/Pictures/ChinaSet_AllFiles/CXR_png"
    listdir = os.listdir(mypath)
    keys = []
    values = []
    for record_name in listdir:
        if os.path.isfile(os.path.join(mypath, record_name)):
            if record_name.endswith("_0.png"):
                keys.append(record_name)
                values.append(0)
            elif record_name.endswith("_1.png"):
                keys.append(record_name)
                values.append(1)
    return dict(zip(keys, values))

#This function returns the number of instances in a set, caculated with the length of the original dataset
def calculate_number_of_instances(dataset, instances_set_size):
    return int(round((len(dataset) * instances_set_size), 0))

#This function retrieves a random instance out of a dataset, adds these to a new dataset and removes it from the old one. 
#This function returns the new (validation)dataset and the updated old dataset
def get_random_instances(dataset, number_of_instances):
    random.shuffle(dataset)
    validation_data = []
    for item in range(number_of_instances):
        random.shuffle(dataset)
        index = random.randint(0, len(dataset) - 1)
        placeholder = dataset.pop(index)
        validation_data.append(placeholder)
    return validation_data, dataset

#this function creates a labellist with the filenames of the images out of the dataset.
def create_labels_list(datalist, sick_healthy_dict):
    labels_list = []
    for item in datalist:
        placeholder = sick_healthy_dict.get(item)
        labels_list.append(placeholder)
    return np.array(labels_list)

#this function writes the images in the right folders for each set there are two folders created, 
#one with positive(sick) images and one with negative(healthy) images.
def write_images_to_folder(samples_list, sick_healthy_dict, path, image_origin):
    for image_title in samples_list:
        class_id = sick_healthy_dict.get(image_title)
        image_origin_extended = image_origin + "/" + image_title

        if class_id == 0:
            path_extended = path + "/Negative_TB"
            img = cv2.imread(image_origin_extended, 0)
            cv2.imwrite(os.path.join(path_extended, image_title), img)


        else:
            path_extended = path + "/Positive_TB"
            img = cv2.imread(image_origin_extended, 0)
            cv2.imwrite(os.path.join(path_extended, image_title), img)



# item = cv2.imread('/home/silvie/Pictures/CXR_png/CHNCXR_0018_0.png')

# nd.imread("/home/silvie/Pictures/ChinaSet_AllFiles/CXR_png/CHNCXR_0018_0.png")


main()
