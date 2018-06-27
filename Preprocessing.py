import os
import random

import cv2
import numpy as np
import argument_parser


def main():
    # Receive the input arguments from command line.
    args = argument_parser.parse_args()

    # This path is command line input from the user. This is the root for the filetree of the processed images
    root_path_tuberculose_images_divided = args.path_target

    # This path is command line input from the user. This is where the downloaded raw images are stored right now
    image_origin = args.path_source

    # Define the paths for training, validating and testing
    train_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Train")
    valid_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Valid")
    test_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Test")

    # Create the folders for storing the training, validating and testing images
    create_folders(train_paths)
    create_folders(valid_paths)
    create_folders(test_paths)

    sick_healthy_dict = get_image_titles(image_origin)

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

    write_images_to_folder(training_data, sick_healthy_dict, train_paths, image_origin)
    print("Done training images")
    write_images_to_folder(validation_data, sick_healthy_dict, valid_paths, image_origin)
    print("Done validating images")
    write_images_to_folder(testing_data, sick_healthy_dict, test_paths, image_origin)
    print("Done testing images")


def define_paths(path_base, step):
    step_path = os.path.join(path_base, step)
    base_path_positive = os.path.join(step_path, "Positive_TB")
    base_path_negative = os.path.join(step_path, "Negative_TB")

    path_list = [step_path, base_path_negative, base_path_positive]
    return path_list


# Creates all the needed folders at the path in path_list
# If a folder already exists, all files from this folder are removed. Subdirectories won't be removed
# path_list = a list with all the paths for the folders
def create_folders(path_list):
    for path in path_list:

        if not os.path.exists(path):
            os.makedirs(path)
            print("Created ", path)

        else:
            file_list = os.listdir(path)
            for filename in file_list:
                filepath = os.path.join(path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            print("Folder already existed. Removed all files from folder. ")

# This function retrieves a list of file and directory names from the current working directory
# This function filters the filenames out
# This function checks if a filename ends with a certain string
# This function creates a dictionary key=image title and value=image class
# image_origin is the folder where the complete dataset can be found
def get_image_titles(image_origin):
    listdir = os.listdir(image_origin)
    keys = []
    values = []
    for record_name in listdir:
        if os.path.isfile(os.path.join(image_origin, record_name)):
            if record_name.endswith("_0.png"):
                keys.append(record_name)
                values.append(0)
            elif record_name.endswith("_1.png"):
                keys.append(record_name)
                values.append(1)
    return dict(zip(keys, values))


# Calculate how many instances are used for training and validating
# dataset is the complete dataset with images
# instances_set_size is the size (as a fraction) of the training or validating set
def calculate_number_of_instances(dataset, instances_set_size):
    return int(round((len(dataset) * instances_set_size), 0))


# Get random instances from dataset for training and validating
# dataset is the complete dataset with images
# number_of_instances is the number of instances that will be selected randomly from the dataset
def get_random_instances(dataset, number_of_instances):
    random.shuffle(dataset)
    list_data = []
    for item in range(number_of_instances):
        random.shuffle(dataset)
        index = random.randint(0, len(dataset) - 1)
        placeholder = dataset.pop(index)
        list_data.append(placeholder)
    return list_data, dataset


# Create a NumPy array with the labels of the images (0 means healthy, 1 means sick)
# sick_healthy_dict is a dictionary with the healthy / sick status of each image
def create_labels_list(datalist, sick_healthy_dict):
    labels_list = []
    for item in datalist:
        placeholder = sick_healthy_dict.get(item)
        labels_list.append(placeholder)
    return np.array(labels_list)


# Write the images to their respective folders
# samples_list is a list with the respective samples
# sick_healthy_dict is a dictionary with the healthy / sick status of each image
# image_origin is the folder where the complete dataset can be found
def write_images_to_folder(samples_list, sick_healthy_dict, path_list, image_origin):
    for image_title in samples_list:
        class_id = sick_healthy_dict.get(image_title)
        #image_origin_extended = image_origin + "/" + image_title
        image_origin_extended = os.path.join(image_origin, image_title)

        if class_id == 0:
            img = cv2.imread(image_origin_extended, 0)
            # cv2.imwrite(os.path.join(path_extended, image_title), img)
            #path_negative = os.path.join(path_list, "Negative_TB")
            cv2.imwrite(os.path.join(path_list[1], image_title), img)


        else:
            img = cv2.imread(image_origin_extended, 0)
            #path_positive = os.path.join(path_list, "Positive_TB")
            cv2.imwrite(os.path.join(path_list[2], image_title), img)


INFO = 'Preproccesing CNN'


main()
