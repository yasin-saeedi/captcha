# Persian Digits CAPTCHA Recognition

A deep learning model designed to recognize Persian numerical CAPTCHAs using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model is trained on a subset of 13,206 images from the [Persian Digits CAPTCHA dataset](https://www.kaggle.com/datasets/aliassareh1/persian-digits-captcha) and achieves an accuracy of 90.91% on the test set.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project automates the recognition of Persian numerical CAPTCHAs, commonly used for security verification on Persian websites. The model leverages a hybrid CNN-LSTM architecture to extract spatial features from CAPTCHA images and capture sequential patterns in digit sequences. It is implemented using TensorFlow 2.3.0, optimized for GPU training, and developed with Python 3.8.

## Dataset
The model is trained on a subset of the [Persian Digits CAPTCHA dataset](https://www.kaggle.com/datasets/aliassareh1/persian-digits-captcha) from Kaggle, consisting of 13,206 grayscale images of numerical CAPTCHAs in Persian. Each image is resized to 256x64 pixels and preprocessed to handle noise and distortions typical in CAPTCHAs. The dataset is split into 13,074 training samples and 132 test samples.

## Features
- **Automated CAPTCHA Solving**: Recognizes Persian digits (0-9) in CAPTCHA images with high accuracy.
- **Hybrid CNN-LSTM Model**: Combines convolutional layers for feature extraction and bidirectional LSTM for sequence modeling.
- **GPU Acceleration**: Optimized for training and inference on GPU for efficient computation.
- **Preprocessing Pipeline**: Includes grayscale conversion, resizing, and normalization to enhance model robustness.

## Model Architecture
The model is a hybrid CNN-LSTM architecture designed for digit sequence recognition:
- **Input**: Grayscale images of size 256x64x1.
- **Convolutional Layers**: Multiple Conv2D layers (16, 32, 64 filters) with ReLU activation and padding to extract spatial features.
- **Pooling Layers**: MaxPooling2D and AveragePooling2D to reduce spatial dimensions while preserving key features.
- **Dropout Layers**: Applied with a 0.3 rate to prevent overfitting.
- **Reshape Layer**: Converts 2D feature maps into sequences for LSTM processing.
- **Bidirectional LSTM**: A 64-unit bidirectional LSTM layer to capture sequential dependencies in digit sequences.
- **Dense Layers**: Final dense layer with softmax activation to classify each position into one of 10 digits (0-9).
- **Total Parameters**: 150,954 trainable parameters.

The model is compiled with the Adam optimizer and sparse categorical crossentropy loss.

## Performance
The model was trained for 40 epochs with a batch size of 32, achieving:
- **Training Accuracy**: 91.24%
- **Validation Accuracy**: 97.52%
- **Test Accuracy**: 90.91% (on 132 test samples)

The training process included checkpointing to save model weights after each epoch. The loss decreased consistently, reaching 0.2531 on the training set and 0.0887 on the validation set by epoch 40.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yasin-saeedi/captcha.git
   cd captcha
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the [Persian Digits CAPTCHA dataset](https://www.kaggle.com/datasets/aliassareh1/persian-digits-captcha) and place it in a `data/` directory.

## Usage
### Inference
To predict digits in a CAPTCHA image:
1. Ensure the trained model weights (`captcha.h5`) are in the project directory.
2. Run the prediction script:
   ```bash
   python predict.py --input data/captcha_image.jpg
   ```
   Example output:
   ```
   Input CAPTCHA: captcha_image.jpg
   Predicted Digits: 1234
   ```

### Training
To train the model from scratch:
1. Place the dataset images in the `data/` directory.
2. Run the training script:
   ```bash
   python train.py --data_path data/ --epochs 40
   ```
   This will train the model and save weights to the `weights/` directory after each epoch.

### Visualization
To visualize predictions on test images:
```bash
python visualize.py
```
This generates a plot of 10 random test images with their predicted digits.

## Requirements
The project requires Python 3.8 and the following dependencies (listed in `requirements.txt`):
- tensorflow==2.3.0
- numpy==1.18.5
- matplotlib==3.4.3
- tqdm==4.65.0

Install them using:
```bash
pip install -r requirements.txt
```

**Note**: A GPU is recommended for faster training and inference.

## Contributing
Contributions are welcome! Please submit a Pull Request or open an issue for suggestions or bug reports.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.