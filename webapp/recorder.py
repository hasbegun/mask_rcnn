"""
Continuously capture images from a opencv cam: webcam or video files
and write to a Redis store.

Usage:
   python recorder.py [width] [height]
"""

import argparse
import coils
import cv2
import numpy as np
import logging
import os
import redis
import time

try:
    import cStringIO as StringIO
except ImportError:
    # from io import StringIO
    from io import BytesIO

# local imports
from share_args import redis_args

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSource(object):
    def __init__(self, cam_source, redis_host, redis_port, width=None, height=None):
        self.__width = width
        self.__height = height
        self.__cur_sleep = 0.1
        self.__store = redis.Redis()
        self.__cam_source = cam_source

        # Monitor the framerate at 1s, 5s, 10s intervals.
        self.__fps = coils.RateTicker((1, 5, 10))

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
            ret, image = cv2.imencode('.jpg', image)
            # sio = StringIO() # python2
            sio = BytesIO()
            np.save(sio, image)
            value = sio.getvalue()
            self.__store.set('image', value)
            image_id = os.urandom(4)
            self.__store.set('image_id', image_id)

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
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    vs = VideoSource(args.source,
                     args.redis_host, args.redis_port,
                     args.width, args.height).run()