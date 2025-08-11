# Ship-Prediction

Project Overview
This project builds a deep learning model to classify images of ships into five categories: Cargo, Military, Carrier, Cruise, and Tankers. The model leverages transfer learning with MobileNetV2 pretrained on ImageNet for efficient feature extraction, enabling accurate ship type classification from images.

Dataset
Source: Kaggle - Game of Deep Learning Ship Datasets

Description: The dataset consists of thousands of labeled ship images divided into five categories.

Format: Images are accompanied by a CSV file with image filenames and their corresponding categories.

Environment
Python 3.8+

TensorFlow 2.x

Keras

OpenCV

Pandas, NumPy

Matplotlib, Seaborn

Google Colab (recommended for easy GPU access)

Key Steps
Data Loading: Download and load images and labels from Kaggle dataset.

Data Preprocessing: Resize images to 224x224 pixels, rescale pixel values, and split into training and validation sets.

Model Architecture: Use MobileNetV2 pretrained model as a base, add custom classification layers.

Training: Train the classification head with frozen base model layers, then optionally fine-tune.

Evaluation: Monitor accuracy and loss on training and validation data.

Usage
Clone this repository or download the notebook.

Download the dataset using Kaggle API or directly from Kaggle.

Run the notebook cells sequentially to preprocess data, train the model, and evaluate results.

Modify hyperparameters as needed to improve performance.

Results
Achieved approximately 87% validation accuracy after 10 epochs.

Demonstrated that transfer learning significantly accelerates training and improves accuracy.

Model can be further fine-tuned or deployed for real-time ship classification.
