import cv2
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

#### Just as example, this is the code to detect objects with the coral TPU ####
## todo run this code with live drone feed


def detect_objects(image_path, model_path, label_path, threshold=0.4):
    # Load the labels
    labels = read_label_file(label_path)

    # Initialize the TPU interpreter
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Load the image
    image = cv2.imread(image_path)
    _, scale = common.set_resized_input(
        interpreter, image.shape[:2], lambda size: cv2.resize(image, size)
    )

    # Run inference
    interpreter.invoke()

    # Get the detection results
    objects = detect.get_objects(interpreter, threshold, scale)

    # Print detected objects
    for obj in objects:
        print(f"Detected {labels[obj.id]} with score {obj.score} at {obj.bbox}")

    return objects


image_path = 'path/to/your/image.jpg'
model_path = 'path/to/your/model.tflite'
label_path = 'path/to/your/labels.txt'

detected_objects = detect_objects(image_path, model_path, label_path)