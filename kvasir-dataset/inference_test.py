# inference_test.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle

# Paths
model_path = 'stomach_classification_model_10_eopch_weight1.pth'  # The saved model path
class_to_idx_path = 'class_to_idx.pkl'  # Path to the saved class-to-index mapping

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations for inference
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the class_to_idx mapping
with open(class_to_idx_path, 'rb') as f:
    class_to_idx = pickle.load(f)
# Create idx_to_class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)
model = model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Function for making predictions on single images
def predict_image(image_path, model, transform, class_names):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds.item()]
    return predicted_class

# Example usage:
if __name__ == '__main__':
    # Provide the path to an image for inference
    image_path = '/workspace/data_split/test/esophagitis/0cbddd47-c7f6-474b-9e9a-b5d2429a2312.jpg'

    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
    else:
        predicted_class = predict_image(image_path, model, val_test_transforms, class_names)
        print(f'Predicted class: {predicted_class}')
