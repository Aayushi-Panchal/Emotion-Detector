<div align="center">

# Emotion Detector

### Machine Learning · Deep Learning · Computer Vision

Detecting human emotions from facial expressions using a **Convolutional Neural Network (CNN)** and an interactive **Streamlit** application.

<br>

<a href="https://github.com/Aayushi-Panchal/Emotion-Detector">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<a href="https://streamlit.io/">
<img src="https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</a>

</div>

---

## Overview

**Emotion Detector** is a Machine Learning and Computer Vision project that detects human emotions from facial expressions.

The application accepts an image, detects the face, and uses a trained CNN model to predict the probability distribution across **seven different emotions**.

The model was trained using a Kaggle dataset containing thousands of categorized facial images.

---

## How It Works

```text
Input Image
     ↓
Face Detection
     ↓
Image Preprocessing
     ↓
CNN Model
     ↓
Emotion Classification
     ↓
Probability Distribution
     ↓
Streamlit Web Application
```

The application displays the predicted probability for **all seven emotion classes** rather than returning only one emotion.

### Example Output

```text
Happy       65%
Neutral     20%
Surprise    10%
Sadness      5%
```

---

## Emotion Classes

<div align="center">

| Emotion | Emotion | Emotion |
|:---:|:---:|:---:|
| Surprise | Fear | Disgust |
| Happiness | Sadness | Anger |
| Neutral | | |

</div>

---

## Dataset & Preprocessing

The project uses a **Kaggle dataset** containing thousands of facial images categorized into seven emotions.

The dataset was divided into:

- Training Data — used for learning patterns from facial expressions
- Testing Data — used to evaluate the model on unseen images

### Data Augmentation

The original dataset was imbalanced, with some emotions having significantly more samples than others.

To address this issue, **Data Augmentation** was applied to create additional variations of existing images.

Transformations included:

- Rotation
- Flipping
- Zooming
- Brightness adjustment

This helped balance the dataset and allowed the model to learn from a wider variety of facial expressions.

---

## Machine Learning Model

The project uses a **Convolutional Neural Network (CNN)** for image classification.

CNNs are suitable for image recognition because they can learn important visual features from images, including patterns around:

- Eyes
- Mouth
- Eyebrows
- Facial structure

The model was trained using **Keras with TensorFlow**.

### Training

| Parameter | Purpose |
|:---|:---|
| Epochs | Complete passes through the training dataset |
| Loss | Measures how incorrect model predictions are |
| Accuracy | Measures the correctness of predictions |
| Learning Rate | Controls how quickly the model learns |
| Batch Size | Controls the number of samples processed at once |

After applying data balancing, augmentation and hyperparameter adjustments, model performance improved with decreasing loss and increasing accuracy.

---

## Model Performance

Training and validation metrics were plotted to monitor model performance during training.

```text
Accuracy  → Increasing
Loss      → Decreasing
```

Training and validation accuracy graphs, along with loss graphs, were used to visually track the learning process.

---

## Tech Stack

### Machine Learning & Computer Vision

<p>
<img src="https://skillicons.dev/icons?i=python,tensorflow,opencv" />
</p>

<p>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white"/>
</p>

### Application & Development

<p>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
<img src="https://skillicons.dev/icons?i=vscode" />
</p>

---

## Application

The trained CNN model is integrated into an interactive **Streamlit web application**.

### Workflow

```text
Upload Image
     ↓
Face Detection
     ↓
CNN Prediction
     ↓
Seven Emotion Probabilities
     ↓
Results Displayed in Web Interface
```

The trained CNN model acts as the core component of the application, while Streamlit provides the interactive user interface.

---

## Deployment

The application was developed using **VS Code**, connected with the trained model and uploaded to GitHub.

### Deployment Workflow

```text
Developed in VS Code
        ↓
Connected with Trained CNN Model
        ↓
Uploaded to GitHub
        ↓
Deployed using Streamlit Cloud
```

This allows the trained model to be accessed through an interactive web application.

---

## Real-World Applications

Potential applications of emotion detection include:

- Mental Health Monitoring
- Human-Computer Interaction
- E-learning and student engagement
- Customer Feedback and Marketing
- Emotional AI
- Virtual Classroom Environments
- Human Behavior Analysis

Further research and development could extend the project toward more advanced real-world applications.

---

## Project Highlights

<div align="center">

| Feature | Implementation |
|:---|:---:|
| Facial Emotion Detection | CNN |
| Computer Vision | OpenCV |
| Data Balancing | Data Augmentation |
| Model Training | TensorFlow / Keras |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Deployment | Streamlit Cloud |

</div>

---

## Key Learnings

Through this project, we worked on:

- Handling an imbalanced image dataset
- Applying data augmentation techniques
- Building and training a CNN
- Monitoring training and validation performance
- Connecting an ML model with a web interface
- Deploying a working ML application

---

<div align="center">

### Machine Learning · Computer Vision · Real-World Applications

<br>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>

<br><br>

**Built with Python, TensorFlow, Keras, OpenCV & Streamlit**

</div>
