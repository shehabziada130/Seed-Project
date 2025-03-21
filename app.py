import numpy as np
import os
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify


pistachio_model=load_model('/models/pistachio_vgg_classifier.h5')
corn_model=load_model('/models/corn_classifier.h5')
soya_model=load_model('/models/soya_classifier.h5')
seed_model=load_model('/models/seed_classifier.h5')

#Preprocess Functions
def preprocess_pistachio(image_path,target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    return img

def preprocess_corn(image_path,target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def preprocess_soya(image_path,target_size=(224, 224)):
  img=cv2.imread(image_path)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img=cv2.resize(img,target_size)
  return img

def preprocess_seed(image_path,target_size=(224, 224)):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = cv2.resize(img,target_size)
  return img

# Predict Functions
def predict_pistachio(image_path):
  preprocessed_image = preprocess_pistachio(image_path)
  preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
  pred_prob = pistachio_model.predict(preprocessed_image)
  prediction=(pred_prob > 0.5).astype(int)
  if prediction[0][0]==0:
    return 'Unhealthy'
  else:
    return 'Healthy'

def predict_corn(image_path):
  img=preprocess_corn(image_path)
  img=np.expand_dims(img,axis=0)
  prob=corn_model.predict(img)
  prediction=(prob>0.5).astype(int)
  if prediction[0][0]==1:
    return 'Unhealthy'
  else:
    return 'Healthy'

def predict_soya(image_path):
  img=preprocess_soya(image_path)
  img=np.expand_dims(img,axis=0)
  prob=soya_model.predict(img)
  prediction=(prob>0.5).astype(int)
  if prediction[0][0]==1:
    return 'Unhealthy'
  else:
    return 'Healthy'


def predict_seed(image_path):
    class_labels=['Corn','Pistachio','Soybean']
    img=preprocess_seed(image_path)
    img=np.expand_dims(img,axis=0)
    prediction=seed_model.predict(img)
    predicted_class=np.argmax(prediction)
    return class_labels[predicted_class]

def predict_image(image_path):
    seed_class=predict_seed(image_path)
    print(seed_class)
    if seed_class=='Corn':
      return f"This Is A Corn Seed And it's {predict_corn(image_path)}"
    elif seed_class=='Soybean':
      return f"This Is A Soybean Seed And it's {predict_soya(image_path)}"
    else:
      return f"This Is A Pistachio Seed And it's {predict_pistachio(image_path)}"
    
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  if 'image' not in request.files:
    return jsonify({"error": "No image file provided."}), 400

  file = request.files['image']
  if file.filename == '':
    return jsonify({"error": "No image file provided."}), 400

  temp_path = "temp_image.jpg"
  file.save(temp_path)

  try:
    result = predict_image(temp_path)
  except Exception as e:
    os.remove(temp_path)
    return jsonify({"error": str(e)}), 500

  os.remove(temp_path)
  return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)