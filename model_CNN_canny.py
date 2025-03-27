import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

data_dir_path = 'data/asl_alphabet_train'
HEIGHT, WIDTH = 200, 200
def get_canny_array(data_dir_path, HEIGHT, WIDTH):
    """Reads images within category subdirectories of a specified directory.
    Preprocesses images by converting them to grayscale, applying Canny edge detection, resizing to specified dimensions,
    and normalizing the images.
    Saves the preprocessed image arrays into array X. X has a shape of (n_samples, HEIGHT, WIDTH, 1).
    Saves the labels into array y as numbers. y has a shape of (n_samples,).
    Returns X and y.
    """

    category_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    X = []
    y = []

    for category_name in os.listdir(data_dir_path):
        category_path = os.path.join(data_dir_path, category_name)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.Canny(image, 10, 70)
            image = cv2.resize(image, (HEIGHT, WIDTH))

            X.append(image)
            y.append(category_names.index(category_name))

    X = np.array(X) / 255.0
    X = X.reshape(-1, HEIGHT, WIDTH, 1)
    y = np.array(y)

    return X, y


X, y = get_canny_array(data_dir_path, HEIGHT, WIDTH)

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=10, stratify=y)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=4,
                               restore_best_weights=True
                               )

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(HEIGHT, WIDTH, 1), padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),

    Conv2D(64, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),

    Conv2D(128, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),

    Flatten(),
    Dense(128, activation='linear'),
    LeakyReLU(alpha=0.1),
    Dense(29, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train,
          epochs=20, batch_size=64,
          validation_data=(X_val, y_val),
          callbacks=[early_stopping],
          verbose=True
         )

model.save('models/model_CNN.h5')
