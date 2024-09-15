import os
import random
import shutil
import json
import time

validation_split = 0.2  # Percentage for validation

thesis_dir = "C:\\Users\\josej\\Thesis"
images_dir = os.path.join(thesis_dir, 'images')  # New folder containing 120 breeds

train_dir = os.path.join(thesis_dir, 'images', 'train')
val_dir = os.path.join(thesis_dir, 'images', 'val')

# Ensure train/val directories are clean
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(train_dir)
os.makedirs(val_dir)

def copy_with_retry(src, dst, retries=5):
    attempt = 0
    while attempt < retries:
        try:
            shutil.copy(src, dst)
            break  # Exit loop if copy is successful
        except PermissionError:
            attempt += 1
            if attempt >= retries:
                print(f"Permission denied for file: {src}. Failed after {retries} attempts.")
            else:
                print(f"Retrying... Attempt {attempt} for file: {src}")
                time.sleep(1)  # Add a small delay before retrying

def split_data(images_dir, train_dir, val_dir, validation_split):
    for breed in os.listdir(images_dir):
        breed_path = os.path.join(images_dir, breed)
        if os.path.isdir(breed_path):
            # Extract the breed name after the '-' character
            breed_name = breed.split('-')[-1]

            # Only select files (ignore directories)
            images = [img for img in os.listdir(breed_path) if os.path.isfile(os.path.join(breed_path, img))]
            
            random.shuffle(images)
            split_idx = int(len(images) * (1 - validation_split))
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Create breed-specific subfolders in train/val directories using the extracted breed name
            os.makedirs(os.path.join(train_dir, breed_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, breed_name), exist_ok=True)
            
            # Copy images to train/val directories with retry mechanism
            for image in train_images:
                src = os.path.join(breed_path, image)
                dst = os.path.join(train_dir, breed_name, image)
                copy_with_retry(src, dst)
            
            for image in val_images:
                src = os.path.join(breed_path, image)
                dst = os.path.join(val_dir, breed_name, image)
                copy_with_retry(src, dst)


split_data(images_dir, train_dir, val_dir, validation_split)
