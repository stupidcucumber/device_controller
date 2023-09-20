#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycoral.utils import edgetpu


class KeyPointClassifierEdgeTPU(object):
    def __init__(self, model_path):
        self.interpreter = edgetpu.make_interpreter(model_path, device='usb:0')

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index


class OptimizedKeyPointClassifierEdgeTPU(object):
    def __init__(self, part_1_path, part_2_path):
        self.part_1_interpreter = edgetpu.make_interpreter(part_1_path, device='usb:0')
        self.part_2_interpreter = edgetpu.make_interpreter(part_2_path, device='usb:1')

        self.part_1_interpreter.allocate_tensors()
        self.part_2_interpreter.allocate_tensors()

        self.part1_input_details = self.part_1_interpreter.get_input_details()
        self.part1_output_details = self.part_1_interpreter.get_output_details()
        self.part2_input_details = self.part_2_interpreter.get_input_details()
        self.part2_output_details = self.part_2_interpreter.get_output_details()


    def __call__(self, landmark_list):
        input_1_index = self.part1_input_details[0]['index']
        output_1_index = self.part1_output_details[0]['index']
        self.part_1_interpreter.set_tensor(
            input_1_index,
            np.array([landmark_list], dtype=np.float32)
        )
        self.part_1_interpreter.invoke()
        output_1 = self.part_1_interpreter.get_tensor(output_1_index)

        input_2_index = self.part2_input_details[0]['index']
        output_2_index = self.part2_output_details[0]['index']
        self.part_2_interpreter.set_tensor(
            input_2_index,
            output_1
        )
        self.part_2_interpreter.invoke()
        result = self.part_2_interpreter.get_tensor(output_2_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
