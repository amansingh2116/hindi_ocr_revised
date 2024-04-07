from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import joblib

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained classifier
MODEL_PATH = r'final project/model.joblib'
loaded_model = joblib.load(MODEL_PATH)

# Function to predict labels using OCR
def predict_labels(image_path):
    img = cv2.imread(image_path)
    # Add your preprocessing and segmentation code here
    
    # Extract features
    hog_features = extract_hog_features(img)
    # Predict label
    predicted_label = loaded_model.predict([hog_features])
    return predicted_label[0]

# Route to upload image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Predict labels and generate new image
            predicted_label = predict_labels(file_path)
            # Add code to generate new image with predicted labels
            # Redirect to display the new image
            return redirect(url_for('display_image', filename=filename))
    return render_template('upload.html')

# Route to display the new image with predicted labels
@app.route('/display/<filename>')
def display_image(filename):
    return render_template('display.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
