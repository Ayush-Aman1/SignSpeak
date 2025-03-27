import os
import re
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer


# Initialization
model_path = 'models/model_NN_MP_for_st.h5'
model = tf.keras.models.load_model(model_path)

category_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
sample_image_paths = [f'sample_images/{image_name}' for image_name in os.listdir('sample_images')]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


# Model Code
def video_frame_callback(frame):
    image = frame.to_ndarray(format='bgr24')  # treat as cv2 image
    image = cv2.flip(image, 1)
    height, width = image.shape[:-1]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        single_hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in
                                 results.multi_hand_landmarks[0].landmark]

        # Saving landmarks for model input
        x = np.array(single_hand_landmarks).reshape(1, 63)

        mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                      color=(255, 0, 255), thickness=4, circle_radius=2),
                                  connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                      color=(20, 180, 90), thickness=2, circle_radius=2)
                                  )

        # interval of 3 is taken as coord z is included
        x_max = int(width * np.max(x[0, ::3]))
        x_min = int(width * np.min(x[0, ::3]))
        y_max = int(height * np.max(x[0, 1::3]))
        y_min = int(height * np.min(x[0, 1::3]))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        # Flipping X-axis landmarks for Left Hand
        if (results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x >
                results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP].x):
            x[:, ::3] = 1 - x[:, ::3]

        # Applying threshold
        if np.max(model.predict(x)) >= 0.5:
            y_pred_idx = np.argmax(model.predict(x))
            y_pred_text = category_names[y_pred_idx]
            cv2.putText(image, y_pred_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    return av.VideoFrame.from_ndarray(image, format='bgr24')


# Streamlit Code
st.header('SignSpeak', divider='green')

with st.container(border=True):
    st.write('''
    This app can detect the hand gestures of the American Sign Language. 
    Click on `START` to open the webcam. 
    Refer to the images below the webcam input for available gestures. 
    Please note that only one hand is detected at a time. 
    ''')
st.write('#')


# Callback Thread
webcam_placeholder = st.empty()

with webcam_placeholder:
    ctx = webrtc_streamer(key='webcam',
                          video_frame_callback=video_frame_callback,
                          rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                          media_stream_constraints={"video": True, "audio": False})

st.write('#')


# Show Gesture Images
for i in range(0, len(sample_image_paths), 7):
    row_sample_image_paths = sample_image_paths[i:i + 7]

    with st.container():
        for col, image_path in zip(st.columns(7), row_sample_image_paths):
            with col:
                st.subheader(re.search(r'/([^/]+)\.jpg$', image_path).group(1))  # category name
                st.image(image_path)
