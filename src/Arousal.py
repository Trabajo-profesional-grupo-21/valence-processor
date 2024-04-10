import numpy as np
import logging

TOP_BEST_CAV = 5

class ArousalCalculator():
    def __init__(self):
        self.relevant_ua = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14',
                    'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27']
        # por ahora lo dejamos aca pero es unico por video
        self.aav_history = {'AU1': [], 'AU2': [], 'AU4': [], 'AU5': [], 'AU6': [], 'AU7': [], 'AU9': [], 'AU10': [], 'AU12': [], 'AU14': [],
            'AU15': [], 'AU17': [], 'AU20': [], 'AU23': [], 'AU24': [], 'AU25': [], 'AU26': [], 'AU27': []}

    def getActivatedValue(self, au_name, au_info_actual_frame):
     av = au_info_actual_frame.get(au_name, 0.0)
     self.aav_history[au_name].append(av)
     return self.aav_history
 
    def calculateMeanActivatedValue(self, au_name, aus_values_mean, frame_count):
        frame_count = frame_count if frame_count <= 800 else 800
        #logging.warn(f"AV HISTORY {self.aav_history[au_name]}")
        au_av_promedio = np.mean(self.aav_history[au_name][-frame_count:], axis=0)  
        aus_values_mean[au_name] = au_av_promedio
        return aus_values_mean

     #TODO: ver que onda despues el ajuste    
    def getCorrectedValue(self, using_correction, au_actual, au_mean):
        actual_value = au_actual
        au_av_mean = 0
        if using_correction:
            au_av_mean = au_mean
        #logging.warn(f"Valor actual {actual_value} -- PROMEDIO {au_av_mean}")
        return np.maximum(0, actual_value - au_av_mean)

    def calculateArousal(self, au_actual_frame: dict, using_correction: bool, frame_count: int):
        aus_values_mean = {}
        cav_total = []
        for au_name in self.relevant_ua:
            #logging.warn(f"UNIT ACCION {au_name}")
            self.aav_history = self.getActivatedValue(au_name, au_actual_frame)
            aus_values_mean = self.calculateMeanActivatedValue(au_name, aus_values_mean, frame_count)
            cav = self.getCorrectedValue(using_correction, au_actual_frame[au_name], aus_values_mean[au_name])
            #logging.warn(f"UNIT ACCION {au_name} - CAV {cav}")
            cav_total.append(cav)
        #logging.warn(f"cav {cav_total}")
        cav_total.sort(reverse=True)
        #logging.warn(f"sorted cav {cav_total}")
        arousal = np.mean(cav_total[:TOP_BEST_CAV])
        return arousal

