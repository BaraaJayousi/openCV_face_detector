from Modules.arguments import detector_arguments
from Modules.trainer import encode_known_faces
from Modules.face_detector import recognize_faces
from Modules.constants import *


Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

args = detector_arguments()


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model= args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)