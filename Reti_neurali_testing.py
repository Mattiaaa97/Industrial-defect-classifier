import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from keras import layers, Sequential
from keras.optimizers import Adam

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# =====================================================================
# 1. DATA LOADING AND SPLITTING
# =====================================================================
def load_preprocessed_data(file_path: str = 'factory_dataset.npz') -> Tuple[np.ndarray, np.ndarray]:
    try:
        data: np.lib.npyio.NpzFile = np.load(file_path)
        X: np.ndarray = data['X_data']
        y: np.ndarray = data['y_data']
        logging.info(f"Successfully loaded {X.shape[0]} samples from {file_path}")
        return X, y
    except FileNotFoundError:
        logging.error(f"Preprocessed file '{file_path}' not found! Run preprocessing first. 🚨")
        exit()


# =====================================================================
# 2. MODEL TRAINING (DEEP LEARNING & DECISION TREE)
# =====================================================================
def train_models(X: np.ndarray, y: np.ndarray) -> None:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
    )

    logging.info("Initializing and training Keras Neural Network...")

    model: Sequential = Sequential()
    model.add(layers.Input(shape=(4096,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)
    logging.info("Keras model trained successfully! 🚀")

    logging.info("Training Decision Tree Classifier...")

    decision_tree: DecisionTreeClassifier = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=5)
    decision_tree.fit(X_train, y_train)

    y_pred: np.ndarray = decision_tree.predict(X_test)

    accuracy: float = float(accuracy_score(y_test, y_pred))
    f1: float = float(f1_score(y_test, y_pred))
    recall: float = float(recall_score(y_test, y_pred))

    logging.info(f"Decision Tree Evaluation: Accuracy = {accuracy:.3f} | F1-Score = {f1:.3f} | Recall = {recall:.3f}")

    logging.info("Generating feature importance heatmap...")
    feature_importance: np.ndarray = decision_tree.feature_importances_.reshape(64, 64)

    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(feature_importance, cmap='hot', annot=False)
    heatmap.set_title("Decision Tree: Pixel Importance Map")
    plt.show()


# =====================================================================
# 3. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    X_data: np.ndarray
    y_data: np.ndarray
    X_data, y_data = load_preprocessed_data('factory_dataset.npz')
    train_models(X_data, y_data)