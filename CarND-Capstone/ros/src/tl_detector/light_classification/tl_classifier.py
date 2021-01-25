from styx_msgs.msg import TrafficLight
from light_classification.classifier_utils import preprocess, traffic_state, load_model


class TLClassifier(object):
    def __init__(self):
        # load classifier
        model_path = 'light_classification/my_model.json' #path to saved model
        weight_path = 'light_classification/weights.h5' #path to saved best weights

        self.model = load_model(model_path, weight_path)
        self.model._make_predict_function()
        print(self.model.summary())

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        pp_image = preprocess(image)
        pred = self.model.predict(pp_image)
        return traffic_state(pred)
