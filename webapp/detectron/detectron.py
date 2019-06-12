from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn import utils
try:
    from . import coco
except (SyntaxError, ImportError):
    import coco


import os
import random
import sys
import skimage.io
import time
import logging
import json
import yaml
from PIL import Image, ImageDraw, ImageDraw2, ImageFont
import numpy as np
import io
try:
    from .log import setup_custom_logger
except (SyntaxError, ImportError):
    from log import setup_custom_logger

logger = setup_custom_logger('detectron', logging.INFO)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Detectron(object):
    def __init__(self):
        super(Detectron, self).__init__()
        # import Mask RCNN
        sys.path.append(self.root_dir)

        # COCO config
        sys.path.append(os.path.join(self.root_dir, 'coco'))

        # download coco model
        if not os.path.exists(self.coco_model_path):
            utils.download_trained_weights(self.coco_model_path)

        config = InferenceConfig()
        # config.display()

        self.model = modellib.MaskRCNN(
            mode=self.detect_mode, model_dir=self.model_dir, config=config)
        self.model.load_weights(self.coco_model_path, by_name=True)

        self.__class_color_map = self.__load_class_color_map()
        logger.info('MASK RCNN init done.')

    def __load_class_color_map(self):
        filename, file_extension = os.path.splitext(
            self.class_color_map_file)
        if file_extension == '.yaml':
            color_map = yaml.load(
                open(os.path.join(self.root_dir, 'detectron',
                                  self.class_color_map_file)))
        elif file_extension == '.json':
            color_map = json.load(
                open(os.path.join(self.root_dir, 'detectron',
                                  self.class_color_map_file)))
        return color_map

    @property
    def class_color_map(self):
        # if mode == 'dict':
        #     return(self.__class_color_map)
        # elif mode == 'list':
        #     colors = []
        #     for i in self.class_names:
        #         colors.append(self.__class_color_map[i])
        #     return(colors)
        # else:
        #     return([])
        return(self.__class_color_map)

    @property
    def detect_mode(self):
        return('inference')

    @property
    def class_color_map_file(self):
        return('class_color_map.json')

    @property
    def class_names(self):
        return(['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
                'teddy bear', 'hair drier', 'toothbrush'])

    @property
    def image_dir(self):
        return(os.path.join(self.root_dir, '../..', 'images'))

    @property
    def coco_model_path(self):
        return(os.path.join(
            self.root_dir, 'detectron', 'ml_models', 'mask_rcnn_coco.h5'))

    @property
    def model_dir(self):
        return(os.path.join(self.root_dir, 'logs'))

    @property
    def root_dir(self):
        return(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def run(self):
        # load random image from the images dir.
        file_names = next(os.walk(self.image_dir))[2]
        image = skimage.io.imread(os.path.join(
            self.image_dir, random.choice(file_names)))

        t = time.time()
        results = self.model.detect([image], verbose=1)
        logger.debug('detect took: %f sec', time.time()-t)

        r = results[0]
        visualize.display_instances(image,
                                    r['rois'],
                                    r['masks'],
                                    r['class_ids'],
                                    self.class_names,
                                    r['scores'])

    def web_run(self, image=None):
        # load random image from the images dir.
        # file_names = next(os.walk(self.IMAGE_DIR))[2]

        # image = skimage.io.imread('test_small.jpg')
        t = time.time()
        results = self.model.detect([image], verbose=1)
        logger.info('detect took: %f', time.time() - t)

        r = results[0]
        marked_img = visualize.mark_instances(image,
                                              r['rois'],
                                              r['masks'],
                                              r['class_ids'],
                                              self.class_names,
                                              r['scores'],
                                              colors=self.class_color_map)
        return(r, marked_img)

    def layers(self, image=None):
        image = skimage.io.imread('test1.jpg')
        t = time.time()
        results = self.model.detect([image], verbose=1)
        logger.info('detect took: %f', time.time() - t)

        r = results[0]
        layers = layer_instances(image, r,
                                 r['rois'],
                                 r['masks'],
                                 r['class_ids'],
                                 self.class_names,
                                 r['scores'],
                                 colors=self.class_color_map)
        # print(r)

        for i, layer in enumerate(layers):
            d = Image.fromarray(layer)
            d.save('layer-%d.png' % i)


def bytes_to_png(image):
    """Convert a PIL Image to a png binary string."""
    output = io.BytesIO()
    image.save(output, 'PNG')
    return output.getvalue()


def create_layer_data(result):
    layer_map = {}
    cids = result.get('class_ids')
    rois = result.get('rois')
    masks = result.get('masks')
    scores = result.get('scores')

    for i, key in enumerate(cids):
        sub_dict = layer_map.get(key, {'rois': [], 'scores': [], 'masks': []})
        sub_dict.get('rois').append(rois[i])
        sub_dict.get('scores').append(scores[i])
        sub_dict.get('masks').append(masks[:, :, i])
        layer_map[key] = sub_dict

    return layer_map
    # classes = result.get('class_ids', 0) # [55, 55, 55, 55, 55, 55, 55, 55, 55, 44, 61]

    # r['rois'],
    # r['masks'],
    # r['class_ids'],
    # self.class_names,
    # r['scores'])


def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1+2, x1:x2] = color
    image[y2:y2+2, x1:x2] = color
    image[y1:y2, x1:x1+2] = color
    image[y1:y2, x2:x2+2] = color
    return image


def apply_mask(image, mask, color, alpha=0.5, color_strangth=255):
    """Apply the given mask to the image.
    """
    for c in range(image.shape[2]):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha *
                                  color[c] * color_strangth,
                                  image[:, :, c])

    return image


def layer_instances(image, result, boxes, masks, class_ids, class_names,
                    scores, colors, show_mask=True, show_bbox=True,
                    captions=None):
    """
    Create images by the classes.
    {class1: {rois: [[x,y,x1,y1],...,[x,y,x1,y1]],
              scores: [s1, ..., sn],
              name: class name
              }}
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    layer_map = create_layer_data(result)

    # Number of instances
    if not len(layer_map.keys()):
        print("\n*** No instances to display *** \n")
        return([])

    # Show area outside image boundaries.
    height, width, d = image.shape
    layer_images = []

    for k, v in layer_map.items():
        # add alpha channle and BG is transparent.
        tmp_image = np.zeros((height, width, d+1))
        tmp_image.astype(np.uint32)

        class_name = class_names[k]
        logger.info('Processing... %s', class_name)
        color = colors.get(class_name)
        if isinstance(color, list):
            color.append(255)
        else:
            c = []
            c.append(color[0])
            c.append(color[1])
            c.append(color[2])
            c.append(255)
            color = c

        # Bounding box
        boxes = v.get('rois')
        masks = v.get('masks')
        for i in range(len(boxes)):
            if show_bbox:
                logger.debug('BBox %s in color: %s', boxes[i], color)
                tmp_image = visualize.draw_box(tmp_image, boxes[i], color)

            # Mask
            mask = masks[i]
            if show_mask:
                tmp_image = visualize.apply_mask(tmp_image, mask, color, 150)

            # y1, x1, y2, x2 = boxes[i]
            # print(tmp_image)
            # # font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf",25)
            # text_img = Image.new("RGBA", (200, 200), (120, 20, 20))
            # draw = ImageDraw2.Draw(text_img)
            # draw.text((x1, y1+5), class_name, color)
            # draw = ImageDraw.Draw(text_img)

            # img.save("a_test.png")

            # # Label
            # if not captions:
            #     class_id = class_ids[i]
            #     score = scores[i] if scores is not None else None
            #     label = class_names[class_id]
            #     caption = "{} {:.3f}".format(label, score) if score else label
            # else:
            #     caption = captions[i]

            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")

        tmp_image = tmp_image.astype(np.uint8)
        # print('shape ', tmp_image.shape)
        # for i in tmp_image:
        # for j in i:
        # print('-->', j)
        # new_data = []
        # for item in tmp_image:
        #     if item[0] == 255 and item[1] == 200 and item[2] == 255:
        #         newData.append((255, 255, 255, 0))
        #     else:
        #         newData.append(item)

        layer_images.append(tmp_image)

    return (layer_images)


if __name__ == '__main__':
    dt = Detectron()
    # dt.run()
    # dt.web_run()
    dt.layers()
