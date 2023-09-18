class Model():
    def __init__(self, edgetpu, min_tracking_confidence=0.7, min_detection_confidence=0.9):
        self.use_edgetpu = edgetpu
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence


    def recognize(self, image):
        raise NotImplementedError('recognize function needs to be redefined in the child class!')
    

    def draw_info(self, image, fps):
        raise NotImplementedError('draw_info function needs to be redefined in the child class!')
    