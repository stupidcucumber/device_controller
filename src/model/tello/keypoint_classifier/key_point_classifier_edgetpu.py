#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycoral.utils import edgetpu


class KeyPointClassifierEdgeTPU(object):
    def __init__(
        self,
        model_path='./edgetpu_format/keypoint_classifier/keypoint_classifier_int_quantization_edgetpu.tflite',
        num_threads=1,
    ):
        self.interpreter = edgetpu.make_interpreter(model_path, device='usb:0')

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
