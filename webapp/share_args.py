"""
Collection of arguments that need to be shared with.
"""

def redis_args(parser):
    """Redis server config args."""
    parser.add_argument('-d', '--redis-host', default='redis',
                        help='Redis server host name. Default: "redis"')
    parser.add_argument('-r', '--redis-port', default='6379',
                        help='Redis server port. Default: 6379')

def mask_rcnn_model_args(parser):
    """Mask RCNN model"""
    parser.add_argument('-m', '--model-name', default='mask_rcnn_coco.h5',
                        help='MASK RCNN model. Default: mask_rcnn_coco.h5')
