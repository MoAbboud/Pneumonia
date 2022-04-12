import numpy as np
from flask import Flask, render_template, request
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

model = load_model('PNmodel.h5')
modelCT = load_model('PNmodel.h5')
Vgg19 = load_model('PNmodel.h5')
RFC = load_model('PNmodel.h5')

result = {0: 'Pneumonia', 1: 'Normal'}


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('Home.html')


@app.route("/submit", methods=['POST'])
def performPrediction():
    if request.method == 'POST':
        xray_img = request.files['my_image']
        if xray_img.filename != "":
            xray_img_path = "static/" + xray_img.filename
            xray_img.save(xray_img_path)

            x = image.load_img(xray_img_path, target_size=(150, 150))
            x = image.img_to_array(x) / 255
            x = x.reshape(-1, 150, 150, 1)

            prediction = (model.predict(x) > 0.5).astype("int32")
            predictionresult = result[prediction[0, 0]]
            return render_template("Home.html", xray_prediction=predictionresult, xray_img_path=xray_img_path)

    return render_template("Home.html")


if __name__ == "__main__":
    app.run(debug=True)
