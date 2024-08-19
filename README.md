# Potato & Tomato Disease Classification Web Application

This project is a web application developed using Flask that allows users to upload images of potato or tomato leaves and receive predictions regarding potential diseases. The application utilizes two deep learning models: one trained to classify potato leaf diseases and another for tomato leaf diseases. Both models were trained using convolutional neural networks (CNNs) and implemented using PyTorch.

## Features

- Image Upload: Users can upload images of potato or tomato leaves.
- Disease Prediction: The application predicts whether the leaf is healthy or affected by specific diseases.
- Dynamic Background: The background image of the web page dynamically changes based on whether the user selects potato or tomato.
- Probability Display: The probability of the predicted class is displayed as a percentage.


## Technologies

- **Python:** Core programming language used for model development and Flask backend.
- **Flask:** Web framework for developing the web application.
- **PyTorch:** Deep learning framework used to develop and train the models.
- **HTML/CSS:** For creating the frontend of the web application.
- **PIL (Pillow):** For image processing.
- **OpenCV:** For image display and preprocessing.
- **Torchvision:** For image transformation utilities.
## Models

- **Potato Disease Classification Model**

  - **Classes:** Potato Early Blight, Potato Late Blight, Potato Healthy
   - **Techniques Used:**
     - Convolutional layers for feature extraction.
     - Batch normalization and max pooling for enhanced training stability and performance.
     - Dropout layers to prevent overfitting.

- **Tomato Disease Classification Model**

  -  **Classes:** Tomato Early Blight, Tomato Late Blight, Tomato Healthy
  - **Techniques Used:**
    - Similar architecture to the potato model with appropriate adjustments for tomato disease classification.
    - Batch normalization, max pooling, and dropout layers are also used here.
## Usage

- Install the required dependencies using `pip install -r requirements.txt`.
- Download the pre-trained model weights and place them in the `models/` directory.
- Run the Flask web application using `python app.py`.
- Access the application in your web browser at `http://localhost:5000`.


## Outcome

- **Performance**
  - **Potato Model:** Achieved an accuracy of 98% on the validation set, with strong performance in classifying Early Blight, Late Blight, and Healthy leaves.
  - **Tomato Model:** Achieved an accuracy of 97% on the validation set, effectively distinguishing between Early Blight, Late Blight, and Healthy leaves.
- **Benefits**
  - **Disease Detection:** Helps farmers and agriculturists detect diseases in potato and tomato plants early, potentially preventing crop losses.
   - **User-Friendly Interface:** The web application provides a simple interface for non-technical users to diagnose plant diseases.

## App
![Potato](https://github.com/user-attachments/assets/334d5a10-ef6c-4720-8d82-de738a3b2871)
![Tomato](https://github.com/user-attachments/assets/f708d7ab-a782-4127-ad08-d39d903dd314)
