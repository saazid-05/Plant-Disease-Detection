# Plant Disease Identification using CNN

A deep learning project that uses Convolutional Neural Networks (CNN) to identify plant diseases from leaf images. The model can classify 38 different plant disease categories across various plant species with 85-95% accuracy.

## 🌱 Project Overview

This project implements a CNN-based solution for automated plant disease detection, helping farmers and agricultural professionals quickly identify plant diseases from leaf images.

## 🚀 Features

- **Multi-class Classification**: Identifies 38 different plant diseases
- **High Accuracy**: Achieves 85-95% accuracy on real dataset
- **Modern Implementation**: Compatible with TensorFlow 2.x
- **Data Augmentation**: Includes rotation, zoom, and flip transformations
- **Demo Mode**: Test the architecture with synthetic data
- **Visualization**: Training/validation curves and model performance plots

## 📁 Project Structure

```
Plant-Disease-Identification-using-CNN/
├── PlantDiseaseDetection_Fixed.py    # Main training script (updated)
├── demo_version.py                   # Demo with synthetic data
├── PlantDiseaseDetection.py         # Original script
├── SETUP_INSTRUCTIONS.md            # Detailed setup guide
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Plant-Disease-Identification-using-CNN.git
cd Plant-Disease-Identification-using-CNN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Option 1: Demo Version (No dataset required)
```bash
python demo_version.py
```

### Option 2: Full Training (Requires PlantVillage dataset)
```bash
python PlantDiseaseDetection_Fixed.py
```

## 📊 Dataset

This project uses the **PlantVillage Dataset** from Kaggle:
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Size**: ~1.2GB
- **Classes**: 38 plant disease categories
- **Images**: 50,000+ plant leaf images

### Dataset Setup:
1. Download from Kaggle
2. Extract to `./plantvillage/` folder
3. Ensure structure: `./plantvillage/[plant]/[disease]/[images]`

## 🏗️ Model Architecture

- **Input**: 256x256x3 RGB images
- **Architecture**: 3 Convolutional blocks + Dense layers
- **Parameters**: ~58M trainable parameters
- **Optimizer**: Adam with learning rate 1e-3
- **Loss**: Categorical crossentropy
- **Regularization**: Dropout and BatchNormalization

## 📈 Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: 85-95%
- **Training Time**: 30-60 minutes (CPU)

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- matplotlib
- numpy
- pillow

## 📝 Usage Examples

### Training the Model
```python
python PlantDiseaseDetection_Fixed.py
```

### Output Files
- `plant_disease_model.h5` - Trained model
- `label_transform.pkl` - Label encoder
- `training_history.png` - Training curves

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PlantVillage Dataset creators
- TensorFlow and Keras communities
- Original research on plant disease detection

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/Plant-Disease-Identification-using-CNN

## 🔗 References

- [Original Kaggle Notebook](https://www.kaggle.com/sumanismcse/plant-disease-detection-using-keras)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) 
