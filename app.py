from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
model = load_model('crop_disease_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

def extract_info(result):
    parts = result.split('___')
    plant_name = parts[0].replace('_', ' ')
    
    if 'healthy' in result.lower():
        disease_name = "No disease detected"
        other_info = "The plant appears to be healthy"
    elif len(parts) > 1:
        disease_info = parts[1].replace('_', ' ')
        disease_name = disease_info.split('(')[0].strip()
        other_info = disease_info.split('(')[-1].replace(')', '').strip() if '(' in disease_info else ''
    else:
        disease_name = "Unknown"
        other_info = ""

    return plant_name, disease_name, other_info

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file and allowed_file(img_file.filename):
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            img_path = os.path.join('uploads', img_file.filename)
            img_file.save(img_path)

            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)

            raw_result = class_names[predicted_class[0]]
            plant_name, disease_name, other_info = extract_info(raw_result)

            if "healthy" in raw_result.lower():
                alert_class = "success"  # Green for healthy
            else:
                alert_class = "danger"  # Red for diseased

            image_url = url_for('uploaded_file', filename=img_file.filename)

            prediction_text = f"{plant_name}: {disease_name}. {other_info}"

            return render_template(
                'result.html',
                prediction=prediction_text,
                alert_class=alert_class,
                image_url=image_url
            )
        else:
            return render_template('error.html', error_message="Unsupported file format or upload error.")
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
