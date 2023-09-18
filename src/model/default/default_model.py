from src.model import Model
import cv2
import numpy as np
from src.utils.utils import check_hand_direction
import mediapipe as mp


class DefaultModel(Model):
    def __init__(self, edgetpu, min_tracking_confidence=0.7, min_detection_confidence=0.9):
        super().__init__(edgetpu=edgetpu, 
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        

    def recognize(self, image):
        image = cv2.flip(image, 1)
        self.results = self.hands.process(image)
        
        if self.results.multi_hand_landmarks is not None:
            h, w, _ = image.shape

            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks,
                                                  self.results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    continue

                wrist_z = hand_landmarks.landmark[0].z
                lm_list = list()

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = int((lm.z - wrist_z) * w)
                    lm_list.append([cx, cy, cz])
                
                label = handedness.classification[0].label.lower()
                lm_array = np.array(lm_list)
                direction, _ = check_hand_direction(lm_array, label)

                image = self.__draw_landmarks(hand_landmarks, image)

                direction_dict = {
                    'left': 6,
                    'right': 7,
                    'up': 2,
                    'down': 4
                } 

                return image, direction_dict[direction]
        
        return image, -1
    

    def __draw_landmarks(self, landmarks, debug_image):
        w = debug_image.shape[1]
        t = int(w / 500)
        self.mp_drawing.draw_landmarks(debug_image, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(102,255,255), thickness=3*t, circle_radius=t),
                    self.mp_drawing.DrawingSpec(color=(51,51,51), thickness=t, circle_radius=t))
        
        return debug_image


    def draw_info(self, debug_image, fps):
        cv2.putText(debug_image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return debug_image
