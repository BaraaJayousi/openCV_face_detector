from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from .constants import *

font_size = int(FONT_SIZE)
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    




def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )

    #draws a rectangle around the text
    draw.rectangle(
        ((text_left, text_top), (text_right+font_size*2.5, text_bottom+font_size)),
        fill="blue",
        outline="blue",
    )

    # draws the desired text
    font = ImageFont.truetype('arial.ttf', font_size)
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
        font=font
    )




def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    validation = False
) -> None:
    print("recognizing faces....")
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model = model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    #load the image from an array using pillow to process the image and draw on it
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "unknown"
        
        if not validation:
            _display_face(draw, bounding_box, name)
    
    print("recognizing faces is done!>>>>> Opening Image...")
    
    del draw
    pillow_image.show()