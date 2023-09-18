#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
from PIL import Image
import numpy as np
import time
import datetime
import serial

from src.controller import TurtleGesturesController
from src.communicator import CommunicatorUART
from src.model import TelloModel, DefaultModel
from src.utils.utils import CvFpsCalc


def get_args():
    print('Reading configuration..')
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--width", default=960, help='cap width', type=int)
    parser.add_argument("--height", default=540, help='cap height', type=int)
    parser.add_argument("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add_argument('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add_argument("--min_detection_confidence", default=0.7,
               help='min_detection_confidence',
               type=float)
    parser.add_argument("--min_tracking_confidence", default=0.5,
               help='min_tracking_confidence',
               type=float)
    parser.add_argument("--buffer_len",
               help='Length of gesture buffer',
               type=int)
    parser.add_argument("-e", "--edge_tpu", action="store_true", help="Enable EdgeTPU compyutations. At least two EdgeTPU devices needs to be connected.")
    parser.add_argument("-c", "--camera_output", action="store_true", help="Turn on camera pereview on the computer.")
    parser.add_argument("-pic", "--pi_camera", action="store_true", help="Extracting data from PiCamera. (Only if additional alterations to the rpi settings had been performed.)")
    parser.add_argument("-usbc", "--usb_camera", action="store_true", help="Extracting data from USB-camera.")
    parser.add_argument("--write_video", action="store_true", help="Write video in the AVI format. Video persists in the current folder.")
    parser.add_argument("-p", "--port", default="", help="Pass port of the turtle, so that we can operate with her.")
    parser.add_argument("--speed", default=-1, help="Defines the speed of the movements.")
    parser.add_argument("--model", default="tello", help="Use Tello model or use other model.")

    args = parser.parse_args()

    return args


def main():
    # init global vars
    global gesture_buffer
    global gesture_id

    # Argument parsing
    args = get_args()

    controller = None

    if args.port != "":
        communicator = CommunicatorUART(args.port, baudrate=115200, 
                                        parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE,)
        if args.speed != -1:
            controller = TurtleGesturesController(communicator, speed=args.speed)
        else:
            controller = TurtleGesturesController(communicator)

    if args.usb_camera:
        camera = cv2.VideoCapture(0)
    elif args.pi_camera:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera_config = camera.create_video_configuration({'size': (640, 480)})
        camera.start()
        time.sleep(1)

        camera.switch_mode(camera_config)

    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        current_date = datetime.datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        writer = cv2.VideoWriter('%s.avi' % current_date, fourcc, 30, (640, 480)) 
    if args.model == 'tello':
        gesture_detector = TelloModel(min_detection_confidence=args.min_detection_confidence, 
                                    min_tracking_confidence=args.min_tracking_confidence, 
                                    edgetpu=args.edge_tpu)
    else:
        gesture_detector = DefaultModel(min_detection_confidence=args.min_detection_confidence, 
                                    min_tracking_confidence=args.min_tracking_confidence, 
                                    edgetpu=args.edge_tpu)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cv_fps_calc.get()

        # Camera capture
        if args.usb_camera:
            _, image = camera.read()
        elif args.pi_camera:
            raw_image = camera.capture_image('main')
            image = Image.new('RGB', size=raw_image.size)
            image.paste(raw_image)

            image = np.asarray(image).astype(np.uint8)
            image = cv2.resize(image, dsize=(640, 480))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        debug_image, gesture_id = gesture_detector.recognize(image)

        if controller != None:
            controller.perform_command(gesture_id)

        if args.camera_output or args.write_video:
            debug_image = gesture_detector.draw_info(debug_image, fps)

        if args.camera_output:
            cv2.imshow('Turtle Gestures Control', debug_image)

            # Process Key (ESC: end)
            key = cv2.waitKey(1) & 0xff
            if key == 27:  # ESC
                break

        if args.write_video:
            writer.write(debug_image)

        print('FPS: %3.2f, gesture_id: %2d      ' % (fps, gesture_id), end='\r')

    if args.camera_output:
        cv2.destroyAllWindows()

    if args.pi_camera:
        camera.stop()

    if args.usb_camera:
        camera.release()

    if args.write_video:
        writer.release()


if __name__ == '__main__':
    main()
