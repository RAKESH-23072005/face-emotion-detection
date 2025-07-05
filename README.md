# Face Expression Recognition

## Introduction

Facial expression is one of the main vectors of human communication. This project is based on the use of a Convolutional Neural Network (CNN) to analyze images of faces and extract the corresponding emotion. The goal is to provide real-time classification of facial expressions using a model trained with an annotated dataset.

## Features

- Face detection using OpenCV.
- Classification of facial expressions into different categories (joy, sadness, anger, surprise, etc.).
- Real-time prediction via webcam or on static images.
- Training and evaluation of a deep learning model with TensorFlow and Keras.

## Installation

### Prerequisites

Before running this project, make sure you have installed the necessary dependencies. If you are using a virtual environment, you can create it with the following command:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### Clone the GitHub repository

```bash
git clone https://github.com/sy895/facial_emotion_recognition.git
cd facial_emotion_recognition
```

## Usage

### Run real-time detection

```bash
python app.py
```

This script captures video from the webcam, detects faces, and displays the predicted emotion in real time.

### Test a static image

If you want to make a prediction on a single image, run:

```bash
python emotion_recognition.py --image path/to/image.jpg
```

### Train the model

If you want to retrain the model with a new dataset, run:

```bash
python train_model.py
```

Make sure the data is organized in `train/` and `test/` directories, with each class represented by a subfolder containing the corresponding images.

## Project Structure

```
face_expression_recognition/
│── app.py                  # Main script for real-time detection
│── train_model.py          # Model training
│── emotion_recognition.py  # Prediction on a static image
│── requirements.txt        # List of required dependencies
│── model/                  # Pre-trained model
│── dataset/                # Folder containing the training and test images
│── utils/                  # Auxiliary functions for data preprocessing
└── README.md               # Project documentation
```

## Dependencies

The main libraries used in this project are:

- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Pillow

To install all dependencies, use:

```bash
pip install -r requirements.txt
```

## Possible Improvements

- Optimize the model with other, more efficient architectures.
- Add a data augmentation mechanism to improve the robustness of the model.
- Support micro-expressions to refine emotion analysis.
- Integrate an interactive mode with display of prediction statistics.

## License

This project is under the MIT license. You are free to use and modify it as you wish.

<!-- face_expression_recognition/
│── app.py # Main script for real-time detection
│── train_model.py # Model training
│── emotion_recognition.py # Prediction on a static image
│── requirements.txt # List of required dependencies
│── model/  # Pre-trained model
│── dataset/ # Folder containing the training and test images
│── utils/ # Auxiliary functions for data preprocessing
└── README.md #