from pathlib import Path
import face_recognition
import pickle
from .constants import *

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    print("training on new faces")
    names = []
    encodings = []
    
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model= model)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"encoded {name} - {filepath.name}")
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names" : names, "encodings": encodings}
    with encodings_location.open(mode = "wb") as f:
        pickle.dump(name_encodings, f)
    print("training is done, I know it all now")
