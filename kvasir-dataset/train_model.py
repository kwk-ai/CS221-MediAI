# train_model.py

import os
import random
import numpy as np
import time
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
data_split_dir = 'data_split'  # The directory where data was split
model_save_path = 'stomach_classification_model_10_eopch_weight2.pth'  # Path to save the trained model
class_to_idx_path = 'class_to_idx.pkl'  # Path to save the class-to-index mapping

# Define data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(os.path.join(data_split_dir, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_split_dir, 'val'), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_split_dir, 'test'), transform=val_test_transforms)

# Create data loaders
batch_size = 32
num_workers = 4  # Adjust based on your system
if os.name == 'nt':  # Windows
    num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Verify class indices
print("Class to index mapping:", train_dataset.class_to_idx)

# Save class_to_idx mapping
with open(class_to_idx_path, 'wb') as f:
    pickle.dump(train_dataset.class_to_idx, f)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Determine and print the GPU device information
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_device_name = torch.cuda.get_device_name(current_device)
    print(f"Using GPU: {current_device} - {gpu_device_name}")
else:
    print("Using CPU")

# Load the pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers for fine-tuning
for name, param in model.named_parameters():
    if 'layer4' in name or 'layer3' in name:
        param.requires_grad = True

# Modify the final layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(train_dataset.classes))
)

# Move the model to the specified device
model = model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Redefine the optimizer to include parameters that require gradients
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Adjust the learning rate scheduler if needed
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model with fine-tuning
best_model = train_model(model, criterion, optimizer, scheduler, num_epochs=15)

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    class_names = train_dataset.classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

# Evaluate on test set
evaluate_model(best_model, test_loader)

# Save the model
torch.save(best_model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')