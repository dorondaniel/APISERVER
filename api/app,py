from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

class ImageClassifier:
    def __init__(self, model_path='mobilenet_v2.pt', class_labels_path='class_labels.txt'):
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load class labels
        with open(class_labels_path) as f:
            self.class_labels = [line.strip() for line in f.readlines()]

    def classify_image(self, image_path):
        image = Image.open(image_path)
        input_image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_image)
        probabilities = torch.softmax(output, dim=1)
        probabilities = probabilities.numpy()[0]
        return probabilities.argsort()[::-1]

def get_top_k_classes(class_indices, k, class_labels):
    top_k_classes = [class_labels[ix] for ix in class_indices[:k]]
    return top_k_classes

classifier = ImageClassifier()

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image_file.save('temp_image.jpg')
    class_indices = classifier.classify_image('temp_image.jpg')
    top_k_classes = get_top_k_classes(class_indices, k=1, class_labels=classifier.class_labels)

    return jsonify({'predictions': top_k_classes})

if __name__ == "__main__":
    app.run(debug=True)
