from src.model.model import Model
import cv2
import copy
import numpy as np
import itertools
import csv
from collections import deque
import mediapipe as mp
from pycoral.utils import edgetpu


class TelloModel(Model):
    def __init__(self, edgetpu=False, min_tracking_confidence=0.7, min_detection_confidence=0.9, history_length=16):
        super().__init__(edgetpu=edgetpu, 
                         min_detection_confidence=min_detection_confidence, 
                         min_tracking_confidence=min_tracking_confidence)

        self.hands, self.keypoint_classifier, self.keypoint_classifier_labels, \
        self.point_history_classifier, self.point_history_classifier_labels = self.load_model()

        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)

        self.history_length = history_length

    
    def load_model(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        if self.use_edgetpu and len(edgetpu.list_edge_tpus()) == 2:
            from src.model.tello.keypoint_classifier.key_point_classifier_edgetpu import KeyPointClassifierEdgeTPU
            from src.model.tello.point_history_classifier.point_history_classifier_edgetpu import PointHistoryClassifierEdgeTPU

            keypoint_classifier = KeyPointClassifierEdgeTPU(model_path='src/model/weights/tello_edgetpu_format/keypoint_classifier/keypoint_classifier_int_quantization_edgetpu.tflite')
            point_history_classifier = PointHistoryClassifierEdgeTPU(model_path='src/model/weights/tello_edgetpu_format/point_history_classifier/point_history_classifier_int_quantization_edgetpu.tflite')
        elif self.use_edgetpu and len(edgetpu.list_edge_tpus()) == 4:
            from src.model.tello.keypoint_classifier.key_point_classifier_edgetpu import OptimizedKeyPointClassifierEdgeTPU
            from src.model.tello.point_history_classifier.point_history_classifier_edgetpu import PointHistoryClassifierEdgeTPU

            keypoint_classifier = [
                'src/model/weights/tello_edgetpu_format/keypoint_classifier/keypoint_classifier_segment_0_of_2_edgetpu.tflite',
                'src/model/weights/tello_edgetpu_format/keypoint_classifier/keypoint_classifier_segment_1_of_2_edgetpu.tflite'
            ]

            keypoint_classifier = OptimizedKeyPointClassifierEdgeTPU(part_1_path=keypoint_classifier[0], part_2_path=keypoint_classifier[1])
            point_history_classifier = PointHistoryClassifierEdgeTPU(model_path='src/model/weights/tello_edgetpu_format/point_history_classifier/point_history_classifier_int_quantization_edgetpu.tflite')
        else:            
            from src.model.tello.keypoint_classifier.keypoint_classifier import KeyPointClassifier
            from src.model.tello.point_history_classifier.point_history_classifier import PointHistoryClassifier

            keypoint_classifier = KeyPointClassifier(model_path='src/model/tello/keypoint_classifier/keypoint_classifier.tflite')
            point_history_classifier = PointHistoryClassifier(model_path='src/model/tello/point_history_classifier/point_history_classifier.tflite')


        with open('src/model/tello/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'src/model/tello/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        return hands, keypoint_classifier, keypoint_classifier_labels, \
               point_history_classifier, point_history_classifier_labels


    def recognize(self, image):
        debug_image = cv2.flip(image, 1)
        gesture_id = -1

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True


        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    continue
                
                brect = self.__calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.__calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = self.__pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = self.__pre_process_point_history(
                    debug_image, self.point_history)
                
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)

                debug_image = self.__draw_bounding_rect(debug_image, brect)
                debug_image = self.__draw_landmarks(debug_image, landmark_list)
                debug_image = self.__draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id]
                )

                gesture_id = hand_sign_id
        else:
            self.point_history.append([0, 0])

        return debug_image, gesture_id


    def draw_info(self, image, fps):
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image
    

    def __calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]


    def __calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    

    def __pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    

    def __pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history


    def __draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Wrist 1
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # Wrist 2
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: Root
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: 1st joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # Index finger: Root
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: 2nd joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: 1st joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # Middle finger: Root
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: 2nd joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: 1st joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: point first
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # Ring finger: Root
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: 2nd joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: 1st joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # Little finger: base
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: 2nd joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: 1st joint
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: point first
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image


    def __draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image


    def __draw_bounding_rect(self, image, brect):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image
