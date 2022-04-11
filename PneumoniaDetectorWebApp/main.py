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

dic = {0: 'Pneumonia', 1: 'Normal'}


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('Home.html')


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img.filename != "":
            img_path = "static/" + img.filename
            img.save(img_path)

            i = image.load_img(img_path, target_size=(150, 150))
            i = image.img_to_array(i) / 255
            i = i.reshape(-1, 150, 150, 1)
            i = preprocess_input(i, mode='tf')

            preds = (model.predict(i) > 0.1).astype("int32")

            p = dic[preds[0, 0]]

            return render_template("Home.html", prediction=p, img_path=img_path)

    return render_template("Home.html")


if __name__ == "__main__":
    app.run(debug=True)
