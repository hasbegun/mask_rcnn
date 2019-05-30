"""
Serve webcam images from a Redis store using Tornado.
Usage:
   python server.py
"""
import argparse
import base64
import coils
import logging
import numpy as np
import os
import redis
from tornado import websocket, web, ioloop
import time

try:
    import cStringIO as StringIO
except ImportError:
    # from io import StringIO
    from io import BytesIO

#local imports
from share_args import redis_args

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



class IndexHandler(web.RequestHandler):
    def get(self):
        self.render(os.path.join('templates', 'index.html'))


class SocketHandler(websocket.WebSocketHandler):
    """ Handler for websocket queries. """
    def __init__(self, *args, **kwargs):
        """ Initialize the Redis store and framerate monitor. """
        super(SocketHandler, self).__init__(*args, **kwargs)
        self._store = redis.Redis()
        self._fps = coils.RateTicker((1, 5, 10))
        self._prev_image_id = None
        self.__max_fps = 100

    @property
    def max_fps(self):
        return self.__max_fps

    def on_message(self, message):
        """ Retrieve image ID from database until different from last ID,
        then retrieve image, de-serialize, encode and send to client. """
        while True:
            time.sleep(1./self.max_fps)
            image_id = self._store.get('image_id')
            if image_id != self._prev_image_id:
                break

        self._prev_image_id = image_id
        image = self._store.get('image')
        # image = StringIO(image) # python2
        image = BytesIO(image)
        image = np.load(image)
        image = base64.b64encode(image)
        self.write_message(image)

        # Print object ID and the framerate.
        text = '{} {:.2f}, {:.2f}, {:.2f} fps'.format(
            id(self), *self._fps.tick())
        logger.info(text)


app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
])

def arg_parser():
    """Defin all args that are necessary.
    add_args is also called.
    """
    parser = argparse.ArgumentParser(description='Run Tornado web server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', default=9000,
                        help='Tornado web server port. Default: 9000')
    redis_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    app.listen(args.port)
    logger.info('Connect localhost: %s', args.port)
    ioloop.IOLoop.instance().start()
