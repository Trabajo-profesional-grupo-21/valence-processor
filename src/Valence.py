
import logging
import numpy as np
from scipy.special import softmax
import cv2
from feat import Detector
from .EmotionsModels import EmotionPredictor, FerplusModel, PyfeatModel



class ValenceCalculator():
    def __init__(self, valence_model: str):
        #Todo: tener en cuenta si se tiene varias insatancias instancias de la api 
        self.model: EmotionPredictor = self.load_model(valence_model)
        logging.info(f"MODELO INSTANCIADO")
        self.emotion_table = {'neutral':0, 'happiness':1, 'surprise':2, 
                              'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 
                              'contempt':7}
    
    def preprocess_image(self, image_bytes):
        input_shape = (1, 1, 64, 64)  
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        img_data = np.array(img, dtype=np.float32)
        img_data = np.resize(img_data, input_shape)
        return img_data
    
    def emotions_mapping(self, scores):
        prob = np.squeeze(softmax(scores))
        inverse_mapping = {v: k for k, v in self.emotion_table.items()}
        return {inverse_mapping[i]: prob for i, prob in enumerate(prob)}
    
    def calculate_valence(self, emotions):
        negative_emotions = ['sadness', 'anger', 'disgust', 'fear']
        negative_intensities = [emotions[emotion] for emotion in negative_emotions]
        happy_intensity = emotions.get('happiness', 0.0)
        valence = happy_intensity - max(negative_intensities, default=0.0)
        return valence
    
    def predict_valence(self, image_bytes):
        emotions = self.model.calculate_emotions(image_bytes)
        valence = self.calculate_valence(emotions)
        #logging.info(f"Valencia {valence}")
        return valence, emotions

    def load_model(self, valence_model: str):
        if valence_model == 'FERPLUS':
            return FerplusModel()
        else:
            return PyfeatModel()



    






