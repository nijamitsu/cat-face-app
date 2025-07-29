I made this experimental AI-powered web app to detect cat breeds from photos. Using modern web tools and machine learning, I built it to explore client-side image classification with a custom-trained model.

Features:
Upload a cat photo to identify its breed
Real-time cat face detection using COCO-SSD
Custom image recognition model I trained on Google Colab GPUs
Fully browser-based, no server required

Built With:
HTML, CSS, JavaScript (Svelte) - UI & interaction
TensorFlow.js - Model loading & inference
COCO-SSD - Cat face detection
Google Colab - Model training with GPU acceleration

Dataset:
This app was trained on the Kaggle Cat Breeds Dataset (https://www.kaggle.com/datasets/denispotapov/cat-breeds-dataset-cleared), which I cleaned and augmented by:
Removing low-resolution and duplicate images.
Adding more cat images sourced from Wikimedia and Flickr to enhance model diversity and robustness.
Ensuring a balanced dataset to improve model accuracy.

How It Works:
Upload or take a photo of a cat
The app detects the cat's face using COCO-SSD
A custom-trained TensorFlow model classifies the breed
Results are shown instantly in the browser

Disclaimer:
This is an experimental project. Accuracy may vary depending on photo quality and breed coverage in the training data.
