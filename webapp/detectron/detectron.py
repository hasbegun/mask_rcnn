
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn import utils
from . import coco

import os
import random
import sys
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import time
import io
from PIL import Image

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Detectron(object):
    def __init__(self):
        super(Detectron, self).__init__()
        # import Mask RCNN
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # ROOT_DIR = os.path.abspath('./')
        sys.path.append(ROOT_DIR)

        # COCO config
        sys.path.append(os.path.join(ROOT_DIR, 'coco'))

        # log store
        MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

        # local path
        COCO_MODEL_PATH = os.path.join(
            ROOT_DIR, 'detectron', 'ml_models', 'mask_rcnn_coco.h5')
        # download coco model
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        self.IMAGE_DIR = os.path.join(ROOT_DIR, '../..', 'images')

        config = InferenceConfig()
        # config.display()

        self.model = modellib.MaskRCNN(
            mode='inference', model_dir=MODEL_DIR, config=config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
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

    def run(self):
        # load random image from the images dir.
        file_names = next(os.walk(self.IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(
            self.IMAGE_DIR, random.choice(file_names)))

        t = time.time()
        results = self.model.detect([image], verbose=1)
        print('detect took: ', time.time()-t)

        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])

    def web_run(self, image=None, ori=None):
        # load random image from the images dir.
        # file_names = next(os.walk(self.IMAGE_DIR))[2]

        # image = skimage.io.imread('test_small.jpg')
        t = time.time()
        results = self.model.detect([image], verbose=1)
        print('detect took: ', time.time() - t)

        # return results[0]
        r = results[0]
        marked_img = visualize.mark_instances(image, r['rois'], r['masks'], r['class_ids'],
                                            self.class_names, r['scores'])
        # import pdb; pdb.set_trace()
        print('*' * 10)
        print(ori)
        print(type(ori))
        print('*' * 10)
        # print(image)
        print('*' * 10)
        # print(marked_img)

        # import base64
        # s = base64.b64encode(marked_img)
        # print('*' * 10)
        # print(marked_img)
        # print('*' * 10)
        # print(s)
        # s = base64.b64encode(marked_img).decode()
        # print(s)

        return (r, marked_img)


if __name__ == '__main__':
    dt = Detectron()
    # dt.run()
    dt.web_run()
