
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn import utils
from samples.coco import coco

import os
import random
import sys
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNDemo(object):
    def __init__(self):
        super(MaskRCNNDemo, self).__init__()
        # import Mask RCNN
        ROOT_DIR = os.path.abspath('../')
        sys.path.append(ROOT_DIR)

        # COCO config
        sys.path.append(os.path.join(ROOT_DIR, 'sample', 'coco'))

        # log store
        MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

        # local path
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
        # download coco model
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        self.IMAGE_DIR = os.path.join(ROOT_DIR, 'images')

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
        t = time.time()
        # load random image from the images dir.
        file_names = next(os.walk(self.IMAGE_DIR))[2]
        input_file = random.choice(file_names)
        image = skimage.io.imread(os.path.join(
            self.IMAGE_DIR, input_file))

        # one image example. for debug
        # image = skimage.io.imread(os.path.join(
        #     self.IMAGE_DIR, '8433365521_9252889f9a_z.jpg'))
        # input_file ='demo'

        results = self.model.detect([image], verbose=1)
        print('detect took: ', time.time()-t)

        r = results[0]
        # print('r:', r)
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             self.class_names, r['scores'])

        res = visualize.mark_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])
        res_img = Image.fromarray(res, 'RGB')
        res_draw = ImageDraw.Draw(res_img)
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        output_file = '_'.join(input_file.split('.')[:-1])
        print('Output: %s_res.png' % output_file)

        # rois, class_ids, scores, masks
        for i in range(len(r['rois'])):
            y1, x1, y2, x2 = r['rois'][i]
            t = self.class_names[r['class_ids'][i]]
            s = r['scores'][i]
            print('box: %s obj: %s score: %s' %(r['rois'][i], t, s))
            res_draw.text((x1, y1-15), '%s %s' % (t, s), (0, 0, 0), font=font)
        res_img.save('%s_res.png' % output_file)



if __name__ == '__main__':
    demo = MaskRCNNDemo()
    demo.run()
