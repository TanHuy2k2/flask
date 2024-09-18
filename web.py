from flask import Flask, render_template, jsonify, Response, request, redirect, send_from_directory
import numpy as np
import os
from PIL import Image
import io
import numpy as np
import base64
from collections import defaultdict, Counter
from predictAgeGender import predict
from deepface import DeepFace
from DB import verify, load_face_data, save_face_data, save_face_file

app = Flask(__name__)
app.secret_key = os.urandom(24)

count_id = 0
data = defaultdict(lambda: [])

# Define the necessary functions
def img_to_encoding(image):
    embedding = DeepFace.represent(image, "Facenet", enforce_detection=False)[0]["embedding"]
    return embedding

@app.route('/predictAgeGender', methods=['POST'])
def process_image():
    global data, count_id

    req = request.get_json()
    image_base64 = req.get('image')
    
    if not image_base64:
        return jsonify({'error': 'No image provided'}), 400

    image_data = image_base64.split(',')[1]

    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes ))

    # Convert PIL Image to numpy array
    image_array = np.array(image)

    face_embed = img_to_encoding(image_array)

    if verify(face_embed):
        print("Verified")
        gender, age = load_face_data(face_embed)
    else:
        print("Not verified")
        gender_label, age_label = predict(image_array)
        data["age"].append(age_label)
        data["gender"].append(gender_label)

        if len(data['age']) > 3:
            age_counts = Counter(data["age"])
            common_age = age_counts.most_common(1)[0][0]
            gender_counts = Counter(data["gender"])
            common_gender = gender_counts.most_common(1)[0][0]
            save_face_file(image, count_id)
            save_face_data(face_embed, {"gender": common_gender, "age": common_age})

            count_id += 1

            data.clear()

    return jsonify({'age': age, 'gender': gender})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port= 2000)