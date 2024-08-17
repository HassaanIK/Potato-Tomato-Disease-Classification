from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from predict import predict_potato, predict_tomato
from model import potato_model, tomato_model
import torch

app = Flask(__name__)

# Load models
potato_model.load_state_dict(torch.load("models\\potato_model_statedict__f.pth", map_location=torch.device('cpu')))
tomato_model.load_state_dict(torch.load("models\\tomato_model_statedict__f.pth", map_location=torch.device('cpu')))

# potato_model = torch.load("Models\\potato_model_statedict__f.pth", map_location=torch.device('cpu'))
# potato_model.load_state_dict(torch.load("Models\\potato_model_statedict__f.pth", map_location=torch.device('cpu')))
# tomato_model = torch.load("Models\\tomato_model_statedict__f.pth", map_location=torch.device('cpu'))
# potato_model.load_state_dict(torch.load("Models\\tomato_model_statedict__f.pth", map_location=torch.device('cpu')))


@app.route('/')
def home():
    # Default to potato model
    return render_template('index.html', model_type='potato')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model type
    model_type = request.form['model_type']
    
    # Get the image file from the request
    file = request.files['file']
    
    if model_type == 'tomato':
        class_name, probability, image = predict_tomato(file, tomato_model)
        background_image = r'static\\tomato_background.jpg'

    else:
        class_name, probability, image = predict_potato(file, potato_model)
        background_image = r'static\\potato_background.webp'      
    
    # Convert image to base64 format
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Pass the base64 encoded image and background image to the frontend
    return render_template('index.html', image=img_str, class_name=class_name, probability=f"{probability * 100:.2f}%", background_image=background_image)

if __name__ == '__main__':
    app.run(debug=True)
