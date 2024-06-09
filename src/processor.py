from .Valence import ValenceCalculator
from .Arousal import ArousalCalculator
from .FPSTracker import FpsTracker
from common.connection import Connection
import ujson as json
import signal
import logging
import numpy as np
import cv2
from .Exceptions import MissingFace
from .config.config import settings


INT_LENGTH = 4

class Processor():
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        self.counter = 0
        self.valenceCalculator = ValenceCalculator(settings.VALENCE_MODEL)
        self.arousalCalculator = ArousalCalculator()
        self.fps_tracker = FpsTracker()
        self.init_conn()

    def init_conn(self):
        remote_rabbit = settings.REMOTE_RABBIT
        if remote_rabbit:
            self.connection = Connection(host=settings.RABBIT_HOST, 
                                    port=settings.RABBIT_PORT,
                                    virtual_host=settings.RABBIT_VHOST, 
                                    user=settings.RABBIT_USER, 
                                    password=settings.RABBIT_PASSWORD)
        else:
            # selfconnection = Connection(host="rabbitmq-0.rabbitmq.default.svc.cluster.local", port=5672)
            self.connection = Connection(host="rabbitmq", port=5672)

        self.input_queue = self.connection.Subscriber("frames", "fanout", "valence_frames")
        self.output_queue = self.connection.Producer(queue_name="processed")

    def _handle_sigterm(self, *args):
        """
        Handles SIGTERM signal
        """
        logging.info('SIGTERM received - Shutting server down')
        self.connection.close()

    def process_img(self, img_body):
        image_bytes = img_body["img"]["0"]
        image_reply = {}
        nparr = np.frombuffer(bytes(image_bytes), np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

        valence, emotions = self.valenceCalculator.predict_valence(image_bytes)
        image_reply[0] = {"valence": valence, "emotions": emotions}

        output_json = {}
        output_json["user_id"] = img_body["user_id"]
        output_json["img_name"] = img_body["img_name"]
        output_json["file_name"] = img_body["file_name"]
        output_json["upload"] = img_body["upload"]
        output_json["origin"] = "valence"
        output_json["reply"] = image_reply

        self.output_queue.send(json.dumps(output_json, default=str))

    def process_batch(self, batch_body):
        batch_info = {}
        output_json = {}
        for frame_id, img_queue in batch_body["batch"].items():
            self.fps_tracker.add_frame()
            nparr = np.frombuffer(bytes(img_queue), np.uint8)

            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

            try:
                valence, emotions = self.valenceCalculator.predict_valence(image_bytes)
            except MissingFace:
                valence = "Missing Face"
                emotions = "Missing Face"
            except Exception as err:
                raise err
            # logging.info(f"Valence: {valence}")
            batch_info[frame_id] = {"valence": valence, "emotions": emotions}
            self.counter += 1

        output_json["user_id"] = batch_body["user_id"]
        output_json["batch_id"] = batch_body["batch_id"]
        output_json["file_name"] = batch_body["file_name"]
        output_json["upload"] = batch_body["upload"]
        output_json["origin"] = "valence"
        output_json["replies"] = batch_info
        # logging.info(f"FPS: {self.fps_tracker.get_fps()}")
        self.output_queue.send(json.dumps(output_json, default=str))



    def _callback(self, body, ack_tag):
        decoded = body.decode()
        body_dec = json.loads(decoded)

        if "EOF" in body_dec:
            self.output_queue.send(decoded)
            return
        
        if "img" in body_dec:
            self.process_img(body_dec)
        elif "batch" in body_dec:
            self.process_batch(body_dec)

    
    def calculate_quadrant(self, arousal, valence):
        if valence >= 0 and arousal >= 0.5:
            # primer cuadrante 
            return 1
        elif valence < 0 and arousal > 0.5:
            #segundo cuadrante
            return 2
        elif valence < 0 and arousal < 0.5:
            #Tercer cuadrante 
            return 3
        else:
            #cuarto cuadrante 
            return 4  
    
    def run(self):
        self.input_queue.receive(self._callback)
        self.connection.start_consuming()
        self.connection.close()