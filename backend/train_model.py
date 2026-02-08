import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def train():
    print("Loading EMNIST (Balanced) dataset via Scikit-Learn (OpenML)...")
    X, y = fetch_openml('EMNIST_Balanced', version=1, return_X_y=True, as_frame=False, parser='auto')

    print(f"Loaded {X.shape[0]} samples.")

    X = X.reshape(-1, 28, 28)
    
    X = X.astype('float32') / 255.0
    
    X = X[..., np.newaxis]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size=0.1, random_state=42)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")

    num_classes = 47

    print("Building CNN model...")
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(256, activation="relu", name="layer1"), 
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu", name="layer2"), 
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    print("Starting training...")
    epochs = 15
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.path.dirname(__file__), "emnist_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    model.fit(
        x_train, 
        y_train, 
        batch_size=128, 
        epochs=epochs, 
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to emnist_model.h5")

if __name__ == "__main__":
    train()
