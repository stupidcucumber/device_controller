#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycoral.utils import edgetpu


class PointHistoryClassifierEdgeTPU(object):
    def __init__(
        self,
        model_path='edgetpu_format/point_history_classifier/point_history_classifier_int_quantization_edgetpu.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = edgetpu.make_interpreter(model_path, device='usb:1')

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
