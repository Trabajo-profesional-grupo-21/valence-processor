import onnx
from onnxruntime import backend
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
from scipy.special import softmax
import numpy as np
import cv2
from PIL import Image
import io
from abc import abstractmethod

class EmotionPredictor:
    @abstractmethod
    def calculate_emotions(self, image_bytes):
        pass

class FerplusModel:
    def __init__(self):
        self.model = onnx.load('./src/valenceModels/emotion-ferplus-12.onnx')
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

    def calculate_emotions(self, image_bytes):
        inputs = self.preprocess_image(image_bytes)
        outputs = list(backend.run(self.model, inputs)) #Run the model
        emotions = self.emotions_mapping(outputs)
        return emotions



class PyfeatModel:
    def __init__(self):
        face_model = "retinaface"
        landmark_model = "mobilenet"
        au_model = "xgb"
        emotion_model = "resmasknet"
        detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
        self.model = detector
        self.emotions_list  = FEAT_EMOTION_COLUMNS

    def preprocess_emotions(self, emotions):
        face_info = emotions[0][0]
        map_emotions = {}
        i = 0
        for emotion in self.emotions_list: 
            map_emotions[emotion] = face_info[i]
            i += 1
        return map_emotions

    def calculate_emotions(self, image_bytes):
        

        # Crear un flujo de bytes desde los datos codificados
        image_stream = io.BytesIO(image_bytes)

        # Abrir la imagen desde el flujo de bytes utilizando PIL/Pillow
        img = Image.open(image_stream)
        detected_faces = self.model.detect_faces(img)
        detected_landmarks = self.model.detect_landmarks(img, detected_faces)
        emotions = self.model.detect_emotions(img, detected_faces, detected_landmarks)
        emotions = self.preprocess_emotions(emotions)
        return emotions
       