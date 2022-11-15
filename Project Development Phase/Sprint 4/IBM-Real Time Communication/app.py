#Import necessary libraries
from flask import Flask, render_template, Response,url_for
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model


#Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    new_model = load_model('Hand-SignV2.h5')
    cap = cv2.VideoCapture(0)
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC,1000*1000)
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.imshow('Input', frame)
            if ret:
                cv2.imwrite("image.jpeg",frame)
                img = image.load_img("./image.jpeg", target_size=(64, 64))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                pred = np.argmax(new_model.predict(x))

                op = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            cv2.waitKey(250)
            yield op[pred]

    except KeyboardInterrupt:
        return -1
        

if __name__=='__main__':
    app.run(debug=True)