import os
import cv2
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# ---------- Simulate Low-Light Images ----------
def simulate_low_light(img):
    """Simulate low-light image by reducing brightness and contrast."""
    alpha = 0.4  # contrast control
    beta = -40   # brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# ---------- Load Images & Generate Both Bright and Low-Light ----------
def load_images_and_labels(folder):
    """Loads images and extracts steering angles from filenames. Augments with low-light."""
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}!")

    images = []
    angles = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping image {img_path} (could not load)")
            continue

        img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2

        # Extract steering angle from filename (e.g., "0_-0.4439.jpg")
        filename = os.path.basename(img_path)
        parts = filename.split("_")

        try:
            angle = float(parts[1].replace(".jpg", ""))
        except ValueError:
            angle = np.random.uniform(-1, 1)

        # Add both bright and low-light images with the same label
        images.append(img)
        angles.append(angle)

        low_light_img = simulate_low_light(img)
        images.append(low_light_img)
        angles.append(angle)

    print(f"Loaded {len(images)} total images (bright + low-light).")
    return np.array(images), np.array(angles)

# ---------- Preprocess Images ----------
def preprocess_images(images):
    """Normalizes images (0-1 range)."""
    images = images / 255.0
    print(f"Preprocessed images shape: {images.shape}")
    return images

# ---------- Train-Test Split ----------
def split_data(images, labels):
    """Splits the dataset into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ---------- Define MobileNetV2 Model ----------
def build_mobilenetv2_model(input_shape):
    """Builds a MobileNetV2 model for steering angle prediction."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))

    for layer in base_model.layers:
        layer.trainable = False  # Freeze pretrained layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='linear')(x)  # Regression output

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])

    model.summary()
    return model

# ---------- Train Model ----------
def train_model(model, X_train, y_train, X_test, y_test):
    """Trains the model and returns history."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=5)
    return history

# ---------- Plot Training Results ----------
def plot_results(history):
    """Plots loss and MAE curves."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()

    plt.show()

# ---------- Run Everything ----------
if __name__ == "__main__":
    # Path to your dataset
    data_folder = r"J:\Univ of Florida\Sem_4\Vsc\Jess_new"

    # Load and augment images
    images, angles = load_images_and_labels(data_folder)

    # Preprocess
    images = preprocess_images(images)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(images, angles)

    # Define model
    input_shape = (224, 224, 3)
    model = build_mobilenetv2_model(input_shape)

    # Train
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Plot
    plot_results(history)

    # Save
    model.save("MobileNetV2_DART.h5")
    print("Model saved as MobileNetV2_DART.h5")
