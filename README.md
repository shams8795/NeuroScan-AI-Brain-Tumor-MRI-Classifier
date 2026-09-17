<h1 align="center">🧠 NeuroScan AI</h1>

<h3 align="center">Brain Tumor MRI Classification Using Deep Learning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Deep_Learning-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/ResNet50-Transfer_Learning-6366F1?style=flat-square">
</p>

<p align="center">
An interactive deep learning application for classifying brain MRI scans into four tumor categories using transfer learning.
</p>

---

# Overview

NeuroScan AI is a deep learning project developed for brain tumor classification from MRI images.

The project compares a Convolutional Neural Network trained from scratch with transfer learning approaches based on ResNet50 and EfficientNetB0.

After evaluating the models, ResNet50 achieved the strongest performance and was integrated into an interactive Streamlit application for MRI image classification.

The application allows users to upload a brain MRI scan, analyze the image, and view the predicted class together with confidence scores and probability visualizations.

---

# Classification Categories

The system classifies MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

---

# Key Features

### MRI Image Upload

Users can upload MRI scans directly through the Streamlit interface.

Supported image formats include:

```text
PNG
JPG
JPEG
```

---

### Deep Learning Classification

The uploaded MRI image is processed and passed to the trained ResNet50-based classifier.

The application automatically identifies the class with the highest predicted probability.

---

### Confidence Score

After classification, the application displays the confidence percentage associated with the predicted tumor class.

For example:

```text
Prediction: Glioma
Confidence: 95.17%
```

---

### Detailed Probability Scores

The application displays prediction probabilities for all four classes:

```text
Glioma
Meningioma
No Tumor
Pituitary
```

This makes it possible to see how the model distributes confidence across the available classes.

---

### Interactive Visualization

Prediction results are visualized using Plotly.

The interface includes:

- Probability bar chart
- Donut chart
- Detailed prediction cards
- Highlighted predicted class

---

# Model Development

The project compares three deep learning approaches.

### CNN from Scratch

A custom Convolutional Neural Network was trained as the baseline model.

Test Accuracy:

```text
22.9%
```

The result showed limited performance when training without pretrained visual features.

---

### EfficientNetB0

EfficientNetB0 was evaluated using transfer learning.

Test Accuracy:

```text
82.9%
```

The pretrained architecture significantly improved classification performance compared with the baseline CNN.

---

### ResNet50

ResNet50 was used as another transfer learning approach.

Test Accuracy:

```text
94.8%
```

ResNet50 achieved the highest test accuracy among the evaluated models and was selected for the final classification application.

---

# Why Transfer Learning?

Medical imaging datasets can be challenging because training a deep neural network from scratch requires large amounts of labeled data.

Transfer learning allows the project to use features already learned from large image datasets.

Using ResNet50 provided stronger visual feature extraction and improved classification performance compared with the CNN trained from scratch.

---

# Image Preprocessing

Before prediction, uploaded MRI images are processed using the following pipeline:

```text
MRI Image
    ↓
Convert to RGB
    ↓
Resize to 224 × 224
    ↓
Convert to NumPy Array
    ↓
Apply ResNet50 Preprocessing
    ↓
Model Prediction
```

The application uses:

```python
tensorflow.keras.applications.resnet50.preprocess_input
```

to prepare images for the trained model.

---

# Prediction Pipeline

```text
Upload MRI Scan
        ↓
Image Preprocessing
        ↓
ResNet50 Model
        ↓
Four-Class Prediction
        ↓
Predicted Tumor Type
        ↓
Confidence Score
        ↓
Detailed Probabilities
        ↓
Interactive Charts
```

---

# Application Interface

The Streamlit application is organized into three main sections.

### Upload MRI Scan

Allows the user to upload an MRI image and start the analysis.

### Prediction Results

Displays:

- Predicted tumor class
- Confidence percentage
- Detailed class probabilities

### Probability Analysis

Provides graphical visualization of model probabilities using:

- Bar chart
- Donut chart

---

# Model Output Example

A sample prediction may look like:

```text
Glioma      95.17%
Meningioma   4.82%
No Tumor     0.01%
Pituitary    0.00%
```

The class with the highest probability is selected as the final prediction.

---

# Technologies Used

### Python

Used for model development, preprocessing, and application logic.

### TensorFlow

Used as the main deep learning framework.

### Keras

Used for model training, loading, and inference.

### ResNet50

Used as the final transfer learning architecture.

### EfficientNetB0

Used as an additional transfer learning model for performance comparison.

### Convolutional Neural Networks

Used for the baseline classification approach.

### Streamlit

Used to build the interactive web application.

### NumPy

Used for image and numerical processing.

### Pillow

Used for image loading and manipulation.

### Plotly

Used for interactive prediction charts.

---

# Project Structure

```text
NeuroScan-AI/
│
├── app_streamlit.py
├── deep_final_project.ipynb
├── resnet_brain_tumor.keras
├── README.md
│
└── screenshots/
    ├── interface.png
    └── prediction-result.png
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroScan-AI.git
```

## 2. Open the Project Directory

```bash
cd NeuroScan-AI
```

## 3. Install the Required Libraries

```bash
pip install streamlit tensorflow numpy pillow plotly
```

## 4. Run the Application

```bash
streamlit run app_streamlit.py
```

The application will open automatically in your browser.

---

# How to Use

1. Open the Streamlit application.
2. Upload a brain MRI image.
3. Click `Analyze Image`.
4. Wait for the model to process the MRI scan.
5. View the predicted tumor class.
6. Review the confidence score.
7. Explore the probability charts for all four classes.

---

# Results

The experiments demonstrated a significant performance improvement when using transfer learning.

```text
CNN from Scratch     → 22.9% Test Accuracy
EfficientNetB0       → 82.9% Test Accuracy
ResNet50             → 94.8% Test Accuracy
```

ResNet50 achieved the strongest performance and was selected as the final model for the Streamlit application.

---

# Project Goal

The project explores the application of deep learning and transfer learning to brain MRI image classification.

It demonstrates how pretrained convolutional neural networks can provide stronger feature extraction than training a CNN from scratch for this dataset.

The final system combines the trained model with an interactive interface to demonstrate real-time MRI image classification.

---

# Disclaimer

This project was developed for educational and research purposes only.

It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical diagnosis or medical advice.

---

# About the Project

NeuroScan AI was designed and developed as a deep learning project combining:

- Medical image classification
- Convolutional Neural Networks
- Transfer learning
- Model performance comparison
- MRI image preprocessing
- Interactive AI deployment

The project demonstrates the complete workflow from model experimentation and evaluation to deployment through a Streamlit application.

---

<h2 align="center">🧠 NeuroScan AI</h2>

<p align="center">
Deep Learning for Brain Tumor MRI Classification
</p>

<p align="center">
Python • TensorFlow • Keras • ResNet50 • Streamlit
</p>
