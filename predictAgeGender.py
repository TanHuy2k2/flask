from huggingface_hub import hf_hub_download
import json
import onnxruntime as rt
import cv2
import numpy as np

# Model predict Age&Gender
repo_name = "nthuy652/MobileNetV3L"
model_files = ['MobileNetV3L.onnx', 'dic_labels_age.json', 'dic_labels_gender.json']
for i, j in enumerate(model_files):
  model_filename = j

# Tải file mô hình từ Hugging Face Hub
  model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)

  # Tải mô hình Keras từ file
  if i == 0:
     sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
  elif i == 1:
    labels_age = json.load(open(model_path))
  else:
    labels_gender = json.load(open(model_path))

def predict(face):
    global data

    faces = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    img_pre = cv2.resize(faces,(200,200))/255.0     # resize image to match model's expected sizing
    img_pre = img_pre.reshape(1,200,200,3)
                
    img_pre = np.array(img_pre).astype(np.float32)

    y_preds = sess.run(None, {sess.get_inputs()[0].name: img_pre})

    gender = round(y_preds[1][0][0])
    age = round(y_preds[0][0][0])

    gender_label = labels_gender.get(str(gender))
    age_label = labels_age.get(str(age))

    return gender_label, age_label