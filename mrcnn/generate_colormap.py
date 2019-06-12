from visualize import random_colors
import json
import logging
import os
import yaml

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassColorMapGenerator(object):
    def __init__(self, classes, class_color_map_file='class_color_map.json'):
        self.__filename, self.__file_extension = os.path.splitext(
            class_color_map_file)
        self.__classes = classes

    @property
    def classes(self):
        return self.__classes

    @property
    def class_color_file_name(self):
        return self.__filename

    @property
    def class_color_file_ext(self):
        return self.__file_extension

    def generate(self):
        color_map = {}
        N = len(self.classes)
        for i, c in enumerate(random_colors(N)):
            color_map[self.classes[i]] = c
        f_name = '%s%s' % (self.class_color_file_name,
                           self.class_color_file_ext)
        with open(f_name, 'w') as f:
            if self.class_color_file_ext == '.yaml':
                logger.info('Generate yaml file.')
                yaml.dump(color_map, f, default_flow_style=False)
                logger.info('YAML %s file is generated.', f_name)
            elif self.class_color_file_ext == '.json':
                logger.info('Generate json file')
                json.dump(color_map, f, sort_keys=True, indent=4)
                logger.info('JSON %s file is generated.', f_name)

    def read_color_map(self):
        with open('%s.%s' % (self.class_color_file_name,
                             self.class_color_file_ext)) as f:
            if self.class_color_file_ext == 'yaml':
                color_map = yaml.load(f)
            elif self.class_color_file_ext == 'json':
                color_map = json.loads(f)
        return(color_map)


if __name__ == '__main__':
    classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

    ClassColorMapGenerator(classes).generate()
    ClassColorMapGenerator(classes, 'class_color_map.yaml').generate()
