import os


def main():
    # This path is command line input from the user
    root_path_tuberculose_images_divided = r"/home/silvie/Pictures/Tuberculose_images_divided"

    path_list = define_paths(root_path_tuberculose_images_divided)
    create_folders(path_list)


def define_paths(root_path_tuberculose_images_divided):
    # Define all the paths for testing
    test_path = os.path.join(root_path_tuberculose_images_divided, "Test")
    test_path_positive = os.path.join(test_path, "Positive_TB")
    test_path_negative = os.path.join(test_path, "Negative_TB")

    # Define all the paths for validating
    valid_path = os.path.join(root_path_tuberculose_images_divided, "Valid")
    valid_path_positive = os.path.join(valid_path, "Positive_TB")
    valid_path_negative = os.path.join(valid_path, "Negative_TB")

    # Define all the paths for training
    train_path = os.path.join(root_path_tuberculose_images_divided, "Train")
    train_path_positive = os.path.join(train_path, "Positive_TB")
    train_path_negative = os.path.join(train_path, "Negative_TB")

    # Add all the paths to a list
    path_list = [root_path_tuberculose_images_divided, test_path, test_path_positive, test_path_negative, valid_path,
                 valid_path_positive, valid_path_negative, train_path, train_path_positive, train_path_negative]
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


main()
