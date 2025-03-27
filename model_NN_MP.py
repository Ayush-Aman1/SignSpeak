# The 'nothing' class is omitted from the data.

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

data_dir_path = 'data/asl_alphabet_train'

def get_hand_landmarks(data_dir_path):
    """Reads images within category subdirectories of a specified directory.
    Saves hand landmarks into array X. X has a shape of (n_samples, 63).
    Saves the labels into array y as numbers. y has a shape of (n_samples,).
    Returns X and y.
    """

    category_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    X = []
    y = []

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    for category_name in os.listdir(data_dir_path):
        category_path = os.path.join(data_dir_path, category_name)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                single_hand_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]
                X.append(single_hand_landmarks)
                y.append(category_names.index(category_name))

    X = np.array(X).reshape(-1, 63)  # shape (n_samples, 63)
    y = np.array(y)  # shape (n_samples)

    return X, y


X, y = get_hand_landmarks(data_dir_path)

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=10, stratify=y)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=4,
                               restore_best_weights=True
                               )

model = models.Sequential([
    layers.InputLayer(input_shape=(63,)),

    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Dense(28, activation='softmax')
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

model.save('models/model_NN_MP_for_st.h5')
model.save('models/model_NN_MP_for_st.keras')
model.save('models/model_NN_MP_for_st.tf')
