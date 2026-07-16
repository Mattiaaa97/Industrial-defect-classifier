import os
import numpy as np
from PIL import Image
import logging

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_execution.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

PATH_HEALTHY: str = r'C:\Users\tiain\PycharmProjects\PythonProject\dataset_fabbrica\sani'
PATH_DEFECTIVE: str = r'C:\Users\tiain\PycharmProjects\PythonProject\dataset_fabbrica\difettosi'

# =====================================================================
# 1. HELPER FUNCTIONS
# =====================================================================
def verify_directories() -> None:
    if not os.path.exists(PATH_HEALTHY) or not os.path.exists(PATH_DEFECTIVE):
        raise FileNotFoundError("One or both dataset directories could not be found! 🚨")

def get_healthy_images() -> list[str]:
    return os.listdir(PATH_HEALTHY)

def get_defective_images() -> list[str]:
    return os.listdir(PATH_DEFECTIVE)

# =====================================================================
# 2. CORE PIPELINE FUNCTION
# =====================================================================
def run_preprocessing() -> None:
    X: list[np.ndarray] = []
    y: list[int] = []

    logging.info("Starting to load healthy (class 0) images...")
    img_name: str
    for img_name in get_healthy_images():
        full_path: str = os.path.join(PATH_HEALTHY, img_name)
        try:
            img: Image.Image = Image.open(full_path)
            pixels: np.ndarray = np.array(img)
            X.append(pixels)
            y.append(0)
            logging.info(f"Healthy image {img_name} loaded successfully! ✅")
        except Exception as e:
            logging.warning(f"Failed to load healthy image {img_name}: {e}")

    logging.info("Starting to load defective (class 1) images...")
    for img_name in get_defective_images():
        full_path = os.path.join(PATH_DEFECTIVE, img_name)
        try:
            img = Image.open(full_path)
            pixels = np.array(img)
            X.append(pixels)
            y.append(1)
            logging.info(f"Defective image {img_name} loaded successfully! ⚠️")
        except Exception as e:
            logging.warning(f"Failed to load defective image {img_name}: {e}")

    X_union: np.ndarray = np.array(X)
    y_union: np.ndarray = np.array(y)

    normalized_X: np.ndarray = X_union / 255.0
    flattened_X: np.ndarray = normalized_X.reshape(normalized_X.shape[0], 4096)

    try:
        np.savez('factory_dataset.npz', X_data=flattened_X, y_data=y_union)
        logging.info("🎉 Preprocessing completed successfully! Clean data saved as 'factory_dataset.npz'")
    except Exception as e:
        logging.error(f"Error occurred while exporting .npz archive: {e}")

# =====================================================================
# 3. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    try:
        verify_directories()
        run_preprocessing()
    except Exception as e:
        logging.critical(f"Pipeline crashed: {e}")