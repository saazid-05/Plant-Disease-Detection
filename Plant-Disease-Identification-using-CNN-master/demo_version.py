# -*- coding: utf-8 -*-
"""
Plant Disease Detection Demo - Creates synthetic data for testing
This version creates fake data to demonstrate the CNN architecture
"""

import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
EPOCHS = 5  # Reduced for demo
INIT_LR = 1e-3
BS = 32
width = 256
height = 256
depth = 3

def create_synthetic_data():
    """Create synthetic plant disease data for demonstration"""
    print("[INFO] Creating synthetic dataset for demo...")
    
    # Define some sample plant disease classes
    classes = [
        'Apple_Black_rot', 'Apple_healthy', 'Apple_rust',
        'Corn_Gray_leaf_spot', 'Corn_healthy', 'Corn_rust',
        'Tomato_Bacterial_spot', 'Tomato_healthy', 'Tomato_Late_blight'
    ]
    
    # Create synthetic images (random noise that resembles plant images)
    n_samples_per_class = 50  # Small dataset for demo
    image_list = []
    label_list = []
    
    for class_name in classes:
        for i in range(n_samples_per_class):
            # Create synthetic image data
            # Add some structure to make it more realistic
            base_color = np.random.rand(3) * 255  # Random base color
            synthetic_image = np.random.rand(height, width, depth) * 100 + base_color
            
            # Add some patterns to simulate plant features
            if 'healthy' in class_name:
                # Healthy plants - more green
                synthetic_image[:, :, 1] += 50  # More green channel
            else:
                # Diseased plants - more brown/yellow
                synthetic_image[:, :, 0] += 30  # More red
                synthetic_image[:, :, 1] += 20  # Less green
            
            # Normalize to 0-255 range
            synthetic_image = np.clip(synthetic_image, 0, 255)
            
            image_list.append(synthetic_image)
            label_list.append(class_name)
    
    print(f"[INFO] Created {len(image_list)} synthetic images across {len(classes)} classes")
    return image_list, label_list, classes

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
    """Main function to run the demo"""
    print("Plant Disease Detection CNN - DEMO VERSION")
    print("=" * 50)
    print("This demo uses synthetic data to show how the model works")
    print("For real results, use the actual PlantVillage dataset")
    print("=" * 50)
    
    # Create synthetic dataset
    image_list, label_list, classes = create_synthetic_data()
    
    # Prepare labels
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    
    # Save label binarizer
    with open('demo_label_transform.pkl', 'wb') as f:
        pickle.dump(label_binarizer, f)
    
    n_classes = len(label_binarizer.classes_)
    print(f"[INFO] Number of classes: {n_classes}")
    print(f"[INFO] Classes: {classes}")
    
    # Convert to numpy array and normalize
    np_image_list = np.array(image_list, dtype=np.float32) / 255.0
    
    print("[INFO] Splitting data to train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        np_image_list, image_labels, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Training samples: {len(x_train)}")
    print(f"[INFO] Testing samples: {len(x_test)}")
    
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
    print(f"\n[INFO] Training network for {EPOCHS} epochs...")
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
    epochs_range = range(1, len(acc) + 1)
    
    # Training and validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b', label='Training accuracy')
    plt.plot(epochs_range, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy (Demo)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b', label='Training loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss (Demo)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('demo_training_history.png')
    plt.show()
    
    # Evaluate model
    print("\n[INFO] Calculating model accuracy")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"Demo Test Accuracy: {scores[1]*100:.2f}%")
    
    # Save the model
    print("[INFO] Saving demo model...")
    model.save('demo_plant_disease_model.h5')
    
    print("\n[INFO] Demo completed successfully!")
    print("Demo model saved as: demo_plant_disease_model.h5")
    print("Demo label transformer saved as: demo_label_transform.pkl")
    print("\nNote: This was a demo with synthetic data.")
    print("For real plant disease detection, download the PlantVillage dataset.")

if __name__ == "__main__":
    main()