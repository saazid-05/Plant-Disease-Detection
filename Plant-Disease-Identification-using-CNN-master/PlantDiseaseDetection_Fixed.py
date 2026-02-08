# -*- coding: utf-8 -*-
"""
Plant Disease Detection Using CNN - Fixed Version
Compatible with modern TensorFlow/Keras versions
"""

import numpy as np
import pickle
import cv2
import os
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Configuration
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
width = 256
height = 256
depth = 3

# Dataset path - modify this to point to your dataset
dataset_path = './plantvillage'  # Changed from '../input/plantvillage/'

def convert_image_to_array(image_dir):
    """Convert image to array format"""
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def check_dataset_exists():
    """Check if dataset exists"""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("\nTo run this project, you need to:")
        print("1. Download the PlantVillage dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("2. Extract it to a folder named 'plantvillage' in this directory")
        print("3. The structure should be: ./plantvillage/[plant_folders]/[disease_folders]/[images]")
        return False
    return True

def load_dataset():
    """Load and preprocess the dataset"""
    if not check_dataset_exists():
        return None, None
    
    image_list, label_list = [], []
    
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(dataset_path)
        
        # Remove .DS_Store if present
        root_dir = [d for d in root_dir if d != ".DS_Store"]

        for plant_folder in root_dir:
            plant_disease_folder_list = listdir(f"{dataset_path}/{plant_folder}")
            plant_disease_folder_list = [d for d in plant_disease_folder_list if d != ".DS_Store"]

            for plant_disease_folder in plant_disease_folder_list:
                print(f"[INFO] Processing {plant_disease_folder} ...")
                plant_disease_image_list = listdir(f"{dataset_path}/{plant_folder}/{plant_disease_folder}/")
                plant_disease_image_list = [img for img in plant_disease_image_list if img != ".DS_Store"]

                # Limit to 200 images per class for faster training
                for image in plant_disease_image_list[:200]:
                    image_directory = f"{dataset_path}/{plant_folder}/{plant_disease_folder}/{image}"
                    if image_directory.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_array = convert_image_to_array(image_directory)
                        if image_array is not None and image_array.size > 0:
                            image_list.append(image_array)
                            label_list.append(plant_disease_folder)
        
        print(f"[INFO] Image loading completed. Total images: {len(image_list)}")
        return image_list, label_list
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def create_model(n_classes):
    """Create the CNN model"""
    model = Sequential()
    inputShape = (height, width, depth)
    
    # First Conv Block
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    
    # Second Conv Block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Conv Block
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    
    return model

def main():
    """Main function to run the training"""
    print("Plant Disease Detection using CNN")
    print("=" * 50)
    
    # Load dataset
    image_list, label_list = load_dataset()
    
    if image_list is None or len(image_list) == 0:
        print("[ERROR] No images loaded. Please check your dataset.")
        return
    
    # Prepare labels
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    
    # Save label binarizer for later use
    with open('label_transform.pkl', 'wb') as f:
        pickle.dump(label_binarizer, f)
    
    n_classes = len(label_binarizer.classes_)
    print(f"[INFO] Number of classes: {n_classes}")
    print(f"[INFO] Classes: {label_binarizer.classes_}")
    
    # Convert to numpy array and normalize
    np_image_list = np.array(image_list, dtype=np.float32) / 255.0
    
    print("[INFO] Splitting data to train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        np_image_list, image_labels, test_size=0.2, random_state=42
    )
    
    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=25, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True, 
        fill_mode="nearest"
    )
    
    # Create model
    model = create_model(n_classes)
    
    # Compile model
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    # Display model summary
    print("\n[INFO] Model Summary:")
    model.summary()
    
    # Train the network
    print("\n[INFO] Training network...")
    history = model.fit(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, 
        verbose=1
    )
    
    # Plot training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # Training and validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate model
    print("\n[INFO] Calculating model accuracy")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {scores[1]*100:.2f}%")
    
    # Save the model
    print("[INFO] Saving model...")
    model.save('plant_disease_model.h5')
    
    print("\n[INFO] Training completed successfully!")
    print("Model saved as: plant_disease_model.h5")
    print("Label transformer saved as: label_transform.pkl")

if __name__ == "__main__":
    main()