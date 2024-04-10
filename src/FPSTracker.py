import time

class FpsTracker:
    def __init__(self):
        # Keep two seconds of history
        self.history_length = 2
        self.frame_times = []

    def add_frame(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        self._discard_old_frames()

    def get_fps(self):
        self._discard_old_frames()

        if len(self.frame_times) == 0:
            return 0

        current_time = time.time()

        return len(self.frame_times) / (current_time - self.frame_times[0])

    def _discard_old_frames(self):
        current_time = time.time()
        # Remove old history
        while len(self.frame_times) > 0 and (current_time - self.frame_times[0]) > self.history_length:
            self.frame_times.pop(0)