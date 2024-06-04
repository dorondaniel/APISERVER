from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

class ImageClassifier:
    def __init__(self, model_path='../mobilenet_v2.tflite', class_labels_path='../class_labels.txt'):
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class labels
        with open(class_labels_path) as f:
            self.class_labels = [line.strip() for line in f.readlines()]

    def classify_image(self, image):
        image = image.resize((224, 224))
        input_image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argsort(output_data[0])[::-1]

def get_top_k_classes(class_indices, k, class_labels):
    top_k_classes = [class_labels[ix] for ix in class_indices[:k]]
    return top_k_classes

classifier = ImageClassifier()

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    class_indices = classifier.classify_image(image)
    top_k_classes = get_top_k_classes(class_indices, k=1, class_labels=classifier.class_labels)

    return jsonify({'predictions': top_k_classes})

if __name__ == "__main__":
    app.run(debug=True)
