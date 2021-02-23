"""
Calculus math recognize.
Written by Inho Choi
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 calculus.py train --dataset=/home/datascience/Workspace/maskRcnn/Mask_RCNN-master/samples/bottle/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 calculus.py train --dataset=/path/to/bottle/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 calculus.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    # Apply color splash to an image
    python3 calculus.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 calculus.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import logging
from logging.config import fileConfig
import json
import skimage.draw
import numpy as np
import datetime

# set logging
fileConfig('logging_config.ini')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.abspath('../../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

CLASSES = {
    "\\alpha": 1,
    "\\approx": 2,
    "\\beta": 3,
    "\\cdot": 4,
    "\\delta": 5,
    "\\div": 6,
    "\\frac": 7,
    "\\gamma": 8,
    "\\geq": 9,
    "\\infty": 10,
    "\\int": 11,
    "\\left(": 12,
    "\\left[": 13,
    "\\left\\{": 14,
    "\\left|": 15,
    "\\leq": 16,
    "\\neq": 17,
    "\\pi": 18,
    "\\pm": 19,
    "\\prime": 20,
    "\\right)": 21,
    "\\right]": 22,
    "\\right|": 23,
    "\\sqrt": 24,
    "\\theta": 25,
    "+": 26,
    ",": 27,
    "-": 28,
    ".": 29,
    "/": 30,
    "0": 31,
    "1": 32,
    "2": 33,
    "3": 34,
    "4": 35,
    "5": 36,
    "6": 37,
    "7": 38,
    "8": 39,
    "9": 40,
    ";": 41,
    "<": 42,
    "=": 43,
    ">": 44,
    "A": 45,
    "C": 46,
    "F": 47,
    "G": 48,
    "H": 49,
    "L": 50,
    "a": 51,
    "b": 52,
    "c": 53,
    "d": 54,
    "e": 55,
    "f": 56,
    "g": 57,
    "h": 58,
    "k": 59,
    "n": 60,
    "p": 61,
    "r": 62,
    "s": 63,
    "t": 64,
    "u": 65,
    "v": 66,
    "w": 67,
    "x": 68,
    "y": 69,
    "z": 70,
    "\\lim_": 71,
    "\\log": 72,
    "\\cot": 73,
    "\\csc": 74,
    "\\to": 75,
    "\\cos": 76,
    "\\sec": 77,
    "\\sin": 78,
    "\\ln": 79,
    "\\tan": 80,
    "\\arcsin": 81,
    "\\arccos": 82,
    "\\arctan": 83,
    "\\arccot": 84,
    "\\arccsc": 85,
    "\\arcsec": 86,
    "\\textup{Undefined}": 87,
    "\\textup{Does not exist}": 88,
    "\\textup{True}": 89,
    "\\textup{False}": 90,
    "\\stackrel{\\textup{H}}{=}": 91,
    "O": 92
}


class CalculusConfig(Config):
    NAME = 'calculus'
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 92 + 1  # math symbols + background
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


class CalculusDateset(utils.Dataset):
    def load_calculus(self, dataset_dir, subset=None):
        for k, v in CLASSES.items():
            self.add_class('calculus', v, k)
            logger.debug('Add class %s - %s', v, k)

        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)
        logger.debug('dataset dir: %s', dataset_dir)

        annotations = json.load(open(os.path.join(dataset_dir, 'data.json')))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes']['name'] for s in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes']['name'] for s in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            num_ids = [CLASSES[a] for a in objects]
            logger.debug('Target - latex:%s img_path:%s num_id: %s', objects, image_path, num_ids)

            self.add_image('calculus', image_id=a['filename'],
                           path=image_path,
                           width=width,
                           height=height,
                           polygons=polygons,
                           num_ids=num_ids)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'calculus':
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        num_id = info['num_ids']
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)

        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_id, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'calculus':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    logger.info('Train....')
    dataset_train = CalculusDateset()
    dataset_train.load_calculus(args.dataset, 'train')
    dataset_train.prepare()

    logger.info('Val....')
    dataset_val = CalculusDateset()
    dataset_val.load_calculus(args.dataset, 'val')
    dataset_val.prepare()

    logger.info('Training network heads')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10, layers='heads')

def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray

    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    if image_path:
        logger.info('Running on %s', args.image)
        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])
        file_name = 'splash_{:Y%m%dT%H%M%S}.png'.format(datetime.datetime.now())
        skimage.io.imread(file_name, splash)
    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        file_name = 'splash_{:%Y%m%dT%H%M%S}.avi'.format(datetime.datetime.now())
        vwriter = cv2.VideoCapture(file_name, cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, (width, height))
        count = 0
        success = True
        while success:
            logger.info('frame:', count)
            success, image = vcapture.read()

            if success:
                image = image[..., ::-1]
                r = model.detect([image], verbose=0)[0]
                splash = color_splash(image, r['masks'])
                splash = splash[..., ::-1]
                vwriter.write(splash)
                count += 1
        vwriter.release()
    logger.info('Saved to ', file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CalculusConfig()
    else:
        class InferenceConfig(CalculusConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
