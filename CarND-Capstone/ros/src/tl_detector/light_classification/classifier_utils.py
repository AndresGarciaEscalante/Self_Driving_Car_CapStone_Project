from keras.models import model_from_json
from keras.models import load_model
from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
# This function loads in our trained model w/ weights and compiles it 
def load_model(model_path, weight_path):
    json_model = open(model_path, 'r')
    loaded_model_json = json_model.read()
    json_model.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model

def preprocess(img):
    
    bgr_image = img.copy()
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh[thresh==255]=1
    rgb_image[thresh!=1]=0
    hsv_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv_img[:, :, 2]
    rgb_image[v<240]=0
    return np.expand_dims(cv2.resize(rgb_image,(299,299)), axis=0)

def traffic_state(predictions):
    top_index = np.argmax(predictions,axis=1)
    
    statesDic={0:TrafficLight.GREEN,1:TrafficLight.UNKNOWN,2:TrafficLight.RED,3:TrafficLight.YELLOW}
    statesDic2={0:'green',1:'none',2:'red',3:'yellow'}
    rospy.logwarn([statesDic2[top] for top in top_index][0])
    return [statesDic[top] for top in top_index][0]