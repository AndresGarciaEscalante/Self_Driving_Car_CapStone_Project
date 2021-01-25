import cv2
import numpy as np
from keras.models import model_from_json
model_path = './my_model.json' #path to saved model
weight_path = './weights.h5' #path to saved best weights
def load_model(model_path, weight_path):
    json_model = open(model_path, 'r')
    loaded_model_json = json_model.read()
    json_model.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model

model = load_model(model_path, weight_path)
model._make_predict_function()
#im = cv2.imread('./0a386f5a-c71f-4fc4-bacb-013742ad5d06.jpg')
im = cv2.imread('./5a3baed4-13ed-4c95-849c-6d9fd864a2dd.jpg')
if im.all() == None: 
    raise Exception("could not load image !")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def preprocess(img):
    rgb_image = img.copy()
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh[thresh==255]=1
    rgb_image[thresh!=1]=0
    hsv_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv_img[:, :, 2]
    rgb_image[v<240]=0
    return np.expand_dims(cv2.resize(rgb_image,(299,299)), axis=0)

pp_image = preprocess(im)
pred = model.predict(pp_image)
print(np.argmax(pred,axis=1))
print(pred)
#model.summary()