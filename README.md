# Lung-Cancer-Detection

```
# Cancer AI Diagnoser

This project implements a deep learning-based application for diagnosing cancer from CT images using a custom-trained neural network model. It consists of three main components:
- **Model Architecture**: Defines the deep learning model used for classification.
- **Inference Engine**: Performs real-time predictions on images using the trained model.
- **User Interface**: Provides a graphical user interface (GUI) for users to upload images and view diagnosis results.

---

## Table of Contents

1. Installation
2. Usage
3. File Descriptions
4. Model Overview
5. Credits

---

## Installation

To run this project locally, you'll need Python 3.7+ and the required libraries. You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not present, manually install the following libraries:
- TensorFlow (for deep learning models)
- Keras (for model architecture)
- OpenCV (for image processing)
- Tkinter (for GUI creation)
- PIL (for image handling)
- Scikit-learn (for evaluation metrics)

---

## Usage

1. **Start the GUI**: Run the `cancer_ai_diagnoser_ui.py` script. It will open a window where you can upload CT images of suspected lung cancer cases.
2. **Upload Image**: Click the "Load image to test for cancer" menu option and select the image you want to analyze.
3. **View Results**: The system will process the image, display it in the GUI, and show the diagnosis result (Cancer/Normal) in the diagnosis result field.
4. **Inference Recording**: The inference results are saved in a log file (`inference_record.txt`).

---

## File Descriptions

### `cancer_ai_diagnoser.py`

This script handles the online inference process for cancer detection. It loads the image, preprocesses it, and uses the trained model to classify it. It outputs a diagnosis (either "Cancer detected" or "Normal lungs detected") and logs each inference event with details about the image and the prediction result.

### `cancer_ai_diagnoser_optimal_model_architecture.py`

This script defines the model architecture used to detect cancer. It sets up the convolutional layers, dropout layers, and fully connected layers of the neural network, compiles the model, and loads the pre-trained weights. It also contains utility functions for evaluating the model, such as rendering confusion matrices and reporting accuracy metrics.

### `cancer_ai_diagnoser_ui.py`

This script sets up the user interface (UI) for interacting with the application. It uses Tkinter to create a GUI that allows users to load an image, display the image, and view the diagnosis results. It integrates with the `cancer_ai_diagnoser.py` module to perform the diagnosis and display the result.

---

## Model Overview

The model used for cancer detection is a custom convolutional neural network (CNN) architecture with the following key features:
- **Convolutional Layers**: Multiple convolutional layers to extract features from the input image.
- **Separable Convolutions**: To reduce the model's complexity while maintaining performance.
- **Batch Normalization**: To normalize the activations and improve model training stability.
- **Dropout**: Applied to reduce overfitting during training.
- **Fully Connected Layers**: To output the final prediction.

The model is trained on X-ray images labeled as either "Normal" or "Cancer," and uses a binary cross-entropy loss function to perform classification.

---

## Credits

- **TensorFlow** and **Keras** for deep learning model implementation.
- **Tkinter** for building the GUI.
- **OpenCV** for image processing.
- **Matplotlib** and **Seaborn** for generating confusion matrices.

---
```
