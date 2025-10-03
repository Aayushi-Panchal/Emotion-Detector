# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

# Constants

RAF_DB_EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]


# Preprocess Function

def preprocess_image(image, target_size):
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)

    image = cv2.resize(image, target_size)        # Resize
    image = image.astype(np.float32) / 255.0      # Normalize
    image = np.expand_dims(image, axis=0)         # Add batch dim
    return image

# Face Detection

def detect_face(image, detector):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            rgb_image = image.copy()
        elif image.shape[2] == 4:
            rgb_image = image[:, :, :3]
        else:
            return None
    else:
        return None

    try:
        results = detector.detect_faces(rgb_image)
        if results:
            largest_face = max(results, key=lambda x: x["box"][2] * x["box"][3])
            x, y, w, h = largest_face["box"]
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(rgb_image.shape[1], x + w + padding)
            y2 = min(rgb_image.shape[0], y + h + padding)

            face = rgb_image[y1:y2, x1:x2]
            return face
        return None
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None

# Prediction

def predict_emotion(model, image):
    try:
        predictions = model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return RAF_DB_EMOTIONS[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# Streamlit App

st.title("🎭 Emotion Detection")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
model_file = st.sidebar.file_uploader("Upload Keras Model", type=["keras"])

if model_file:
    with open("temp_model.keras", "wb") as f:
        f.write(model_file.getbuffer())
    try:
        model = tf.keras.models.load_model("temp_model.keras")
        st.sidebar.success("✅ Model loaded!")
        st.sidebar.info(f"Input shape: {model.input_shape}")

        # Target size from model
        if len(model.input_shape) == 4:
            target_h, target_w = model.input_shape[1], model.input_shape[2]
            target_size = (target_w, target_h)
        else:
            target_size = (100, 100)

        st.sidebar.info(f"Target size: {target_size}")
        use_face_detection = st.sidebar.checkbox("Use Face Detection", value=True)

        # Upload Image
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supports both RGB and grayscale images"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Input Image", use_column_width=True)
            with col2:
                st.write("**Image Info**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                if hasattr(image, "format"):
                    st.write(f"Format: {image.format}")

            if st.button("🔍 Predict Emotion", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    processed_image = None
                    face_detected = False

                    if use_face_detection:
                        detector = MTCNN()
                        face = detect_face(image, detector)
                        if face is not None:
                            st.success("Face detected ✅")
                            st.image(face, caption="Detected Face", use_column_width=True)
                            processed_image = preprocess_image(face, target_size)
                            face_detected = True
                        else:
                            st.warning("⚠️ No face detected, using full image.")

                    if processed_image is None:
                        processed_image = preprocess_image(np.array(image), target_size)

                    emotion, confidence, all_preds = predict_emotion(model, processed_image)

                    if emotion is not None:
                        st.header("📊 Results")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("Predicted Emotion", emotion)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")

                        # All emotions
                        st.subheader("All Emotion Probabilities")
                        for emo, prob in zip(RAF_DB_EMOTIONS, all_preds):
                            st.progress(float(prob), text=f"{emo}: {prob:.3f}")

                        # Top 3
                        sorted_emotions = sorted(
                            zip(RAF_DB_EMOTIONS, all_preds),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3]

                        st.write("**🏆 Top 3 Predictions:**")
                        for i, (emo, prob) in enumerate(sorted_emotions, 1):
                            st.write(f"{i}. {emo} → {prob:.2%}")
                    else:
                        st.error("Prediction failed ❌")

    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
else:
    st.sidebar.warning("⚠️ Please upload a Keras model file")
    st.info("Upload your trained emotion detection model in the sidebar to begin")

st.markdown("---")
st.markdown(" Emotion Detection using CNN + MTCNN | Built with Streamlit")
