from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import numpy as np
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

model = load_model('sports_model_efficient_net.h5')

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        print("Invalid image file:", str(e))
        return False

def classify_image(image_path, model):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)

        label_index = np.argmax(pred)

        classes = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 
                   'baseball', 'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 
                   'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket', 'croquet', 
                   'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs', 
                   'figure skating women', 'fly fishing', 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 
                   'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping', 
                   'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 
                   'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling', 'luge', 'motorcycle racing', 'mushing', 
                   'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 
                   'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby', 
                   'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing', 'skydiving', 
                   'snow boarding', 'snowmobile racing', 'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing', 
                   'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 
                   'volleyball', 'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing', 
                   'wingsuit flying']
        
        predicted_class = classes[label_index]

        classification = '%s (%.2f%%)' % (predicted_class, pred[0, label_index]*100)
        return classification
    except Exception as e:
        print("Error during prediction:", str(e))
        return "Error during prediction. Please try again."

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    imagefile = None
    image_path = None

    if request.method == 'POST' and 'imagefile' in request.files:
        imagefile = request.files['imagefile']

        if imagefile.filename == '':
            return jsonify({"error": "No selected file"})

        destination_directory = request.form.get('destination_directory', 'images')
        os.makedirs(destination_directory, exist_ok=True)

        image_path = os.path.join(destination_directory, imagefile.filename)
        imagefile.save(image_path)

        if is_valid_image(image_path):
            prediction = classify_image(image_path, model)
            return jsonify({"prediction": prediction})
        else:
            return jsonify({"error": "Invalid image file. Please upload a valid image."})

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
