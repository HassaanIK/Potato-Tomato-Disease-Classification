from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from predict import predict_potato
from model import model
import torch

model.load_state_dict(torch.load("models\\potato_model_statedict__f.pth", map_location=torch.device('cpu')))
app = Flask(__name__)

# Your predict_mask function here...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    
    # Predict the mask
    class_name, probability, image = predict_potato(file, model)
    
    # Convert image to base64 format
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Pass the base64 encoded image to the frontend
    return render_template('index.html', image=img_str, class_name=class_name, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
