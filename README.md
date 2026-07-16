Industrial Defect Classifier (ML vs. Deep Learning)
This repository contains a modular and strongly-typed Python project designed to classify industrial components as either Healthy (Class 0) or Defective (Class 1) using computer vision.

The project evaluates and compares two different approaches:

A Deep Learning Multi-Layer Perceptron (MLP) built with Keras/TensorFlow.

A classical Decision Tree Classifier built with Scikit-Learn, including pixel feature importance visualization.

Key Features & Best Practices
Strong Static Typing: Fully typed Python codebase using type hints (np.ndarray, Image.Image, list[str], etc.) for robust development.

Modular Architecture: Separated preprocessing pipeline from the model training workflow to optimize system memory (RAM).

Binary Serialization (.npz): Image matrices and labels are normalized, flattened, and exported as a compressed NumPy archive for lightning-fast loading.

Production-Ready Logging: Complete execution tracking via Python’s logging library, outputting status to both the console and specialized log files.

Stratified Splits: Ensures equal class representation (Healthy vs. Defective) in both training and testing datasets.

Project Structure
Plaintext
├── .gitignore                      # Tells Git which files to ignore (local dataset, logs, virtual environment)
├── README.md                       # Project presentation and documentation
├── Reti_neurali_preprocessing.py   # Script 1: Directory validation, image loading, normalization, and export
├── Reti_neurali_testing.py         # Script 2: Data loading, Keras MLP, and Decision Tree training
Pipeline Walkthrough
1. Data Preprocessing (Reti_neurali_preprocessing.py)
Validates local directories and checks for errors safely.

Loads images using Pillow (PIL), converting them into raw NumPy matrices.

Performs pixel intensity normalization (rescaling from 0-255 to [0.0, 1.0]).

Dynamically flattens 2D 64x64 images into a 4096-dimensional vector.

Automatically serializes arrays into a compact factory_dataset.npz archive.

2. Model Architecture & Training (Reti_neurali_testing.py)
Keras MLP Neural Network:

Input layer: 4096 nodes.

Hidden layer: 32 nodes (ReLU activation).

Output layer: 1 node (Sigmoid activation for binary classification).

Loss function: Binary Cross-Entropy.

Optimizer: Adam (lr = 0.0001).

Decision Tree Classifier:

Criterion: Gini impurity.

Max depth: 5.

Generates a Pixel Importance Heatmap using Seaborn to visually isolate which areas of the component images trigger a "defective" flag.

Getting Started
Prerequisites
Make sure you have the required libraries installed:

Bash
pip install numpy pillow matplotlib seaborn tensorflow scikit-learn
Running the Pipeline
Place your raw images inside local directories: dataset_fabbrica/sani/ and dataset_fabbrica/difettosi/.

Run the preprocessing step:

Bash
python Reti_neurali_preprocessing.py
Train and compare the models:

Bash
python Reti_neurali_testing.py
