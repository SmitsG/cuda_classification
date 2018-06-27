import argparse

INFO = 'Preprocessing CNN'


def parse_args():
    """
    Argument parser
    :return: args: All arguments are parsed to the args.
    """

    parser = argparse.ArgumentParser(description=INFO,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # ------------------------------------------Main information-------------------------------------------------------
    parser.add_argument("--path_target",
                        type=str,
                        required=True,
                        default=r"/home/silvie/Pictures/Tuberculose_images_divided",
                        help="This is the root for the filetree of the processed images.")

    parser.add_argument("--path_source",
                        type=str,
                        required=True,
                        default=r"/home/silvie/Pictures/ChinaSet_AllFiles/CXR_png",
                        help="This path is command line input from the user. This is where the downloaded raw images should be stored")

    # parse all arguments
    args = parser.parse_args()

    return args
