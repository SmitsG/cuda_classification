import os


def main():
    # This path is command line input from the user
    root_path_tuberculose_images_divided = r"/home/silvie/Pictures/Tuberculose_images_divided"

    path_list = define_paths(root_path_tuberculose_images_divided)
    create_folders(path_list)

    train_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Train")
    test_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Test")
    valid_paths = define_paths(path_base=root_path_tuberculose_images_divided, step="Valid")


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


main()
