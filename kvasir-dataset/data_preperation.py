# data_preparation.py

import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = '/workspace/data'  # Replace with the path to your data directory containing the four folders
data_split_dir = 'data_split'  # Destination directory for the split data

# The four classes
classes = ['esophagitis', 'normal-pylorus', 'normal-z-line', 'polyps']

# Function to split data into train, val, test
def split_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    os.makedirs(dest_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
    
    for cls in classes:
        cls_src_dir = os.path.join(source_dir, cls)
        images = os.listdir(cls_src_dir)
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        total = len(images)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for img_name in train_images:
            src = os.path.join(cls_src_dir, img_name)
            dst = os.path.join(dest_dir, 'train', cls, img_name)
            shutil.copyfile(src, dst)
        
        for img_name in val_images:
            src = os.path.join(cls_src_dir, img_name)
            dst = os.path.join(dest_dir, 'val', cls, img_name)
            shutil.copyfile(src, dst)
        
        for img_name in test_images:
            src = os.path.join(cls_src_dir, img_name)
            dst = os.path.join(dest_dir, 'test', cls, img_name)
            shutil.copyfile(src, dst)
    
    print("Data split completed.")

# Split the data
split_data(data_dir, data_split_dir)