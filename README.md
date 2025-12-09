# CIFAR-10 Image Classification with CNN

This project showcases image classification using the CIFAR-10 dataset by comparing two different deep learning approaches: a baseline Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN). The full pipeline includes image visualization, preprocessing, model training, and performance evaluation.

## 📊 Dataset
The CIFAR-10 dataset contains:
- **50,000 training images**
- **10,000 test images**
- Image size: **32x32 pixels**, RGB
- 10 target classes:
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## 🧠 Models Implemented

### 🔹 Baseline ANN
- Fully connected layers
- Evaluated to provide performance comparison against the CNN
- Activation functions: ReLU & Sigmoid

### 🔹 Convolutional Neural Network (CNN)
- Extracts spatial features with convolution + pooling operations
- Final dense layers for classification
- Activation functions: ReLU & Softmax

## ⚙️ Workflow Summary
- Load CIFAR-10 dataset
- Visualize sample images
- Normalize pixel values to improve training
- Train ANN and CNN models
- Generate accuracy scores and classification reports

## 🚀 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-Learn

## ▶️ Running the Code
```bash
python image_classification_using_cnn.py
