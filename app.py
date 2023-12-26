from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import base64
import functions as fn 
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

model_filename = 'model.h5'
loaded_model = load_model('model.h5')  


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



option = []
grayscale = None
edge = "scharr"
rotate = None
angle = 90
median_blur = ""
transform = "translate"
bilateral = ""
processed_images = {'grayscale':None, 'edge':None, 'rotate':None, 'transform':None, 'median_blur': None, 'bilateral': None}
image_path = None
edge = None
label_to_class_name = {0: 'Ajanta Caves',
 1: 'Charar-E- Sharif',
 2: 'Chhota_Imambara',
 3: 'Ellora Caves',
 4: 'Fatehpur Sikri',
 5: 'Gateway of India',
 6: 'Humayun_s Tomb',
 7: 'India gate pics',
 8: 'Khajuraho',
 9: 'Sun Temple Konark',
 10: 'alai_darwaza',
 11: 'alai_minar',
 12: 'basilica_of_bom_jesus',
 13: 'charminar',
 14: 'golden temple',
 15: 'hawa mahal pics',
 16: 'iron_pillar',
 17: 'jamali_kamali_tomb',
 18: 'lotus_temple',
 19: 'mysore_palace',
 20: 'qutub_minar',
 21: 'tajmahal',
 22: 'tanjavur temple',
 23: 'victoria memorial'}

df = pd.read_excel('MonumentDetection.xlsx')

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        image = request.files['image']
        if image.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            img = cv2.imread(image_path)
            resized_img = cv2.resize(img, (150, 150))
            
            resized_img = resized_img / 255.0  # Normalize the pixel values
            
            input_data = np.expand_dims(resized_img, axis=0)
            
            predictions = loaded_model.predict(input_data)
            
            max_index = np.argmax(predictions)
            
            predicted_class = label_to_class_name[max_index]
            description = ""
            print(df.columns)
            for index, row in df.iterrows():
                name = row['Name '].strip() 
                description = row['Description '].strip() 

                if name == predicted_class:
                    print(description)
                    break
            
            return render_template('detect.html', image=image_path, label=predicted_class, description=description)

    return render_template('detect.html')

@app.route('/enhancement', methods=['GET', 'POST'])
def enhancement():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
        grayscale = request.form.getlist('grayscale')
        rotate = request.form.getlist('rotate')
        edge = request.form.getlist('edge')
        transform = request.form.getlist('apply-transformation')
        median_blur = request.form.get('median_blur')
        bilateral = request.form.get('bilateral')
        if edge:
            edge_type = request.form.get('edge-detection')
            print(edge_type)
            print(edge)

        if grayscale:
            grayscale_image = fn.convert_to_grayscale(image_path)
            processed_images['grayscale'] = grayscale_image
        if edge:
            edge_detection = fn.edge_detection(image_path, edge_type)
            processed_images['edge']=edge_detection
        if rotate:
            angle = float(request.form.get('angle'))
            rotate_image = fn.rotation(image_path,angle=angle)
            processed_images['rotate']=rotate_image
        if transform:
            type = request.form.get('transform')
            x = request.form.get('tx')
            y = request.form.get('ty')
            processed_images['transform'] = fn.transformation_func(image_path, type, float(x),float(y))
            print(type)

        if median_blur:
            print(median_blur)
            blur_value =  int(request.form.get('blur'))
            print(blur_value)
            blur_value = 5
            processed_images['median_blur'] = fn.median_blur(image_path,blur_value)
        if bilateral:
            d = int(request.form.get('d'))
            sigma_color = int(request.form.get('sigma_color'))
            sigma_space = int(request.form.get('sigma_space'))
            processed_images['bilateral'] = fn.bilateral_filter(image_path, d, sigma_color, sigma_space)
        return render_template('enhance.html', image=image_path, processed_images=processed_images)
    else:
        return render_template('enhance.html', processed_images=processed_images)




if __name__ == "__main__":
    app.run(debug=True)




