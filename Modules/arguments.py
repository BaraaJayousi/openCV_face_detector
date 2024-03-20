import argparse

def detector_arguments():
    parser = argparse.ArgumentParser(description="Recognize faces in an image")
    parser.add_argument("--train", action="store_true", help="Train on input data")
    parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
    parser.add_argument('--validate', action="store_true", help= "Validate the models accuracy ")
    parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)")
    parser.add_argument("-f", action="store", help="Path to an image with an unknown face(s)")
    return parser.parse_args()