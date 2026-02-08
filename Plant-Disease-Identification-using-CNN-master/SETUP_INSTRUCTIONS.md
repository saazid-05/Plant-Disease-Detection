# Plant Disease Detection CNN - Setup Instructions

## Project Overview
This project uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images. The model can classify various plant diseases across different plant species.

## Files in this Project

1. **PlantDiseaseDetection.py** - Original code (has compatibility issues)
2. **PlantDiseaseDetection_Fixed.py** - Updated version compatible with modern TensorFlow
3. **demo_version.py** - Demo version that works with synthetic data
4. **SETUP_INSTRUCTIONS.md** - This file

## Prerequisites

✅ **Already Installed:**
- Python 3.13.7
- TensorFlow 2.20.0
- OpenCV, scikit-learn, matplotlib, numpy, pillow

## How to Run the Project

### Option 1: Demo Version (Quick Test)
```bash
python demo_version.py
```
This runs with synthetic data to demonstrate the CNN architecture.

### Option 2: Full Version with Real Dataset

#### Step 1: Download the PlantVillage Dataset
1. Go to Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. You'll need a Kaggle account (free)
3. Download the dataset (about 1.2GB)

#### Step 2: Extract and Organize Dataset
1. Extract the downloaded zip file
2. Create a folder named `plantvillage` in this project directory
3. Copy the plant folders into `./plantvillage/`

The structure should look like:
```
Plant-Disease-Identification-using-CNN-master/
├── plantvillage/
│   ├── Apple/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   ├── Apple___Cedar_apple_rust/
│   │   └── Apple___healthy/
│   ├── Corn_(maize)/
│   │   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
│   │   ├── Corn_(maize)___Common_rust_/
│   │   ├── Corn_(maize)___Northern_Leaf_Blight/
│   │   └── Corn_(maize)___healthy/
│   └── [other plant folders...]
├── PlantDiseaseDetection_Fixed.py
└── [other files...]
```

#### Step 3: Run the Full Training
```bash
python PlantDiseaseDetection_Fixed.py
```

## Expected Results

### With Real Dataset:
- Training time: 30-60 minutes (depending on your CPU)
- Expected accuracy: 85-95%
- Model size: ~220MB
- Classes: 38 different plant disease categories

### Training Process:
1. Loads and preprocesses images (256x256 pixels)
2. Splits data into training (80%) and testing (20%)
3. Applies data augmentation (rotation, zoom, flip)
4. Trains CNN for 25 epochs
5. Saves trained model and label encoder
6. Displays training/validation curves
7. Reports final test accuracy

## Output Files

After successful training:
- `plant_disease_model.h5` - Trained CNN model
- `label_transform.pkl` - Label encoder for class names
- `training_history.png` - Training/validation curves

## Model Architecture

The CNN consists of:
- 3 Convolutional blocks with BatchNormalization and Dropout
- MaxPooling layers for dimensionality reduction
- Dense layers with 1024 neurons
- Softmax output for multi-class classification
- Total parameters: ~58M

## Troubleshooting

### Common Issues:

1. **"Dataset not found" error**
   - Make sure the `plantvillage` folder exists in the project directory
   - Check that plant folders are directly inside `plantvillage/`

2. **Memory errors**
   - Reduce batch size (BS) from 32 to 16 or 8
   - Reduce image limit per class from 200 to 100

3. **Slow training**
   - This is normal on CPU. Training 25 epochs can take 30-60 minutes
   - Consider reducing EPOCHS from 25 to 10 for faster testing

4. **Low accuracy**
   - Make sure you're using the real PlantVillage dataset, not synthetic data
   - Check that images are loading correctly

## Alternative: Using Kaggle Notebook

If you prefer not to download the dataset locally:
1. Visit: https://www.kaggle.com/sumanismcse/plant-disease-detection-using-keras
2. Fork the notebook
3. Run it directly on Kaggle with GPU acceleration

## Next Steps

After training:
1. Use the saved model for predictions on new plant images
2. Create a web interface for easy disease detection
3. Experiment with different CNN architectures
4. Try transfer learning with pre-trained models

## Performance Tips

- **GPU**: If you have a GPU, install `tensorflow-gpu` for faster training
- **Data**: More diverse training data improves accuracy
- **Augmentation**: Current augmentation helps prevent overfitting
- **Epochs**: Monitor validation accuracy to avoid overfitting