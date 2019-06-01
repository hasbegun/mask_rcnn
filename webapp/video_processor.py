"""
Continuously capture images from a opencv cam: webcam or video files
and write to a Redis store.

Usage:
   python recorder.py [width] [height]
"""

import argparse
import logging
import os
import time

import coils
import cv2
import numpy as np

try:
    import cStringIO as StringIO
except ImportError:
    # from io import StringIO
    from io import BytesIO

# local imports
from processor import Processor
from share_args import redis_args, mask_rcnn_model_args

from mrcnn import model as modellib
from mrcnn import utils, visualize
from samples.coco import coco

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class VideoProcessor(Processor):
    def __init__(self, cam_source, redis_host, redis_port, model_name,
                 width=None, height=None):
        super(VideoProcessor, self).__init__(redis_host, redis_port)
        self.__width = width
        self.__height = height
        self.__cur_sleep = 0.1
        self.__cam_source = cam_source

        # Monitor the framerate at 1s, 5s, 10s intervals.
        self.__fps = coils.RateTicker((1, 5, 10))

        self.__mask_rcnn_model = model_name
        root_dir = os.path.abspath('../')
        model_dir = os.path.join(root_dir, 'logs')
        coco_model_path = os.path.join(root_dir, self.mask_rcnn_model)
        self.__download_coco_model(coco_model_path)

        config = InferenceConfig()
        config.display()

        self.model = modellib.MaskRCNN(
            mode='inference', model_dir=model_dir, config=config)
        self.model.load_weights(coco_model_path, by_name=True)
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']

    def __download_coco_model(self, model_path):
        # download coco model
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)

    @property
    def mask_rcnn_model(self):
        return self.__mask_rcnn_model

    @mask_rcnn_model.setter
    def mask_rcnn_model(self, model):
        self.__mask_rcnn_model = model

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def max_sleep(self):
        return 5.0

    @property
    def cur_sleep(self):
        return self.__cur_sleep

    @cur_sleep.setter
    def cur_sleep(self, t):
        self.__cur_sleep = t

    @property
    def cam_source(self):
        return self.__cam_source

    @cam_source.setter
    def cam_source(self, src):
        self.__cam_source = src

    @property
    def max_try(self):
        return 5

    def run(self):
        cur_try = 0
        # Create video capture object, retrying 5 times.
        while cur_try < self.max_try:
            cur_try += 1
            cap = cv2.VideoCapture(self.cam_source)
            if cap.isOpened():
                logging.info('Video source is opened: %s', self.cam_source)
                break
            logging.error('Video source is not opened, sleeping %ss', self.cur_sleep)
            time.sleep(self.cur_sleep)
            if self.cur_sleep < self.max_sleep:
                self.cur_sleep *= 2
                self.cur_sleep = min(self.cur_sleep, self.max_sleep)
                continue
            self.cur_sleep = 0.1

        if self.width and self.height:
            cap.set(3, self.width)
            cap.set(4, self.height)

        # Repeatedly capture current image,
        # encode, serialize and push to Redis database.
        # Then create unique ID, and push to database as well.
        while True:
            ret, image = cap.read()
            if image is None:
                time.sleep(0.5)
                continue

            results = self.model.detect([image], verbose=1)
            r = results[0]
            img =

            ret, image = cv2.imencode('.jpg', image)
            # sio = StringIO() # python2
            sio = BytesIO()
            np.save(sio, image)
            value = sio.getvalue()
            self._store.set('image', value)
            image_id = os.urandom(4)
            self._store.set('image_id', image_id)

            # Print the framerate.
            text = '{:.2f}, {:.2f}, {:.2f} fps'.format(*self.__fps.tick())
            logger.info(text)


def arg_parser():
    parser = argparse.ArgumentParser(description='Video input runs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--source', default=0,
                        help='Camera source. Default is 0 which is cam 0 which is default camera on the device.'
                             'If file name is given, the file will be loaded and play. video.mp4')
    parser.add_argument('--width', default=None,
                        help='Width of camera or video size')
    parser.add_argument('--height', default=None,
                        help='Height of camera or video size')
    redis_args(parser)
    mask_rcnn_model_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    vs = VideoProcessor(args.source,
                        args.redis_host, args.redis_port,
                        args.model_name,
                        args.width, args.height).run()
