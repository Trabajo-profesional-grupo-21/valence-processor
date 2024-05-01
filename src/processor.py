from .Valence import ValenceCalculator
from .Arousal import ArousalCalculator
from .FPSTracker import FpsTracker
from common.connection import Connection
import ujson as json
import signal
import logging
import numpy as np
import cv2
import os
from dotenv import load_dotenv


INT_LENGTH = 4

class Processor():
    def __init__(self):
        load_dotenv()
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        self.counter = 0
        self.valenceCalculator = ValenceCalculator(os.getenv('VALENCE_MODEL'))
        self.arousalCalculator = ArousalCalculator()
        self.fps_tracker = FpsTracker()
        self.init_conn()

    def init_conn(self):
        remote_rabbit = os.getenv('REMOTE_RABBIT', False)
        if remote_rabbit:
            self.connection = Connection(host=os.getenv('RABBIT_HOST'), 
                                    port=os.getenv('RABBIT_PORT'),
                                    virtual_host=os.getenv('RABBIT_VHOST'), 
                                    user=os.getenv('RABBIT_USER'), 
                                    password=os.getenv('RABBIT_PASSWORD'))
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

    def _callback(self, body, ack_tag):
        decoded = body.decode()
        body_dec = json.loads(decoded)

        if "EOF" in body_dec:
            self.output_queue.send(decoded)
            return
        

        batch_info = {}
        output_json = {}
        for frame_id, img_queue in body_dec["batch"].items():
            self.fps_tracker.add_frame()
            nparr = np.frombuffer(bytes(img_queue), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

            
            valence, emotions = self.valenceCalculator.predict_valence(image_bytes)
            # logging.info(f"Resultados frame {self.counter} -- Valence {valence} -- emociones {emotions}")
            batch_info[frame_id] = {"valence": valence, "emotions": emotions}
            self.counter += 1

        # logging.info(f"user id {body_dec['user_id']} and user_batch {body_dec['batch_id']}")
        output_json["user_id"] = body_dec["user_id"]
        output_json["batch_id"] = body_dec["batch_id"]
        output_json["origin"] = "valence"
        output_json["replies"] = batch_info
        logging.info(f"FPS: {self.fps_tracker.get_fps()}")
        self.output_queue.send(json.dumps(output_json, default=str))
		    
    
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