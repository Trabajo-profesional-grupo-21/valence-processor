from .Valence import ValenceCalculator
from .Arousal import ArousalCalculator
from common.connection import Connection
import ujson as json
import signal
import logging
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from multiprocessing import Process, Queue


INT_LENGTH = 4

class Processor():
    def __init__(self):
        load_dotenv()
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        self.counter = 0
        self.connection = Connection()
        self.input_queue = self.connection.Subscriber("frames", "fanout", "valence_frames")
        self.output_queue = self.connection.Producer(queue_name="processed")
        self.valenceCalculator = ValenceCalculator(os.getenv('VALENCE_MODEL'))
        self.arousalCalculator = ArousalCalculator()

    def _handle_sigterm(self, *args):
        """
        Handles SIGTERM signal
        """
        logging.info('SIGTERM received - Shutting server down')
        self.connection.close()

    def _process_image(self, frame_id, img_list, result_queue):
        nparr = np.array(img_list, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        valence, emotions = self.valenceCalculator.predict_valence(image_bytes)
        logging.info(f"Resultados frame {frame_id} -- Valence {valence} -- emociones {emotions}")
        result_queue.put((frame_id, {"valence": valence, "emotions": emotions}))
        
    def _callback(self, body, ack_tag):
        # logging.info(f"Received frame: {self.counter}")
        body = json.loads(body.decode())
        logging.info(type(body))
        logging.info(len(body["batch"]))
        batch_info = {}
        output_json = {}

        result_queue = Queue()

        # Procesar cada imagen en paralelo
        processes = []
        for frame_id, img_list in body["batch"].items():
            process = Process(target=self._process_image, args=(frame_id, img_list, result_queue))
            process.start()
            processes.append(process)

        # Esperar a que todos los procesos terminen
        for process in processes:
            process.join()

        # Recolectar resultados de la cola de mensajes
        while not result_queue.empty():
            frame_id, result_data = result_queue.get()
            batch_info[frame_id] = result_data

        logging.info(f"user id {body['user_id']} and user_batch {body['batch_id']}")
        output_json["user_id"] = body["user_id"]
        output_json["batch_id"] = body["batch_id"]
        output_json["origin"] = "valence"
        output_json["replies"] = batch_info
        logging.info(f"{output_json}")
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