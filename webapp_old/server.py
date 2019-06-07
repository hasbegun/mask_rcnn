"""
Serve webcam images from a Redis store using Tornado.

Usage:
   python server.py
"""
import argparse
import base64
import logging
import os
import time

import coils
import numpy as np
import redis
from tornado import websocket, web, ioloop

try:
    import cStringIO as StringIO
except ImportError:
    # from io import StringIO
    from io import BytesIO

# local imports
from share_args import redis_args
from uploadhandler import UploadHandler

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('index.html')


class UploadForm(web.RequestHandler):
    def get(self):
        self.render("upload_form.html")


class VideoDisplayHandler(web.RequestHandler):
    def get(self):
        self.render('video_display.html')


class SocketHandler(websocket.WebSocketHandler):
    """ Handler for websocket queries. """

    def __init__(self, *args, **kwargs):
        """ Initialize the Redis store and frame rate monitor. """
        super(SocketHandler, self).__init__(*args, **kwargs)
        # self._store = redis.Redis(host=globals().get('redis_host'),
        #                           port=globals().get('redis_port'))
        self._store = redis.Redis()
        self._fps = coils.RateTicker((1, 5, 10))
        self._prev_image_id = None
        self._max_fps = 100

    @property
    def max_fps(self):
        return self._max_fps

    def on_open(self):
        # self._connection.add(self)
        pass

    def on_close(self):
        # self._connection.remove(self)
        pass

    def on_message(self, message):
        """ Retrieve image ID from database until different from last ID,
        then retrieve image, de-serialize, encode and send to client. """
        while True:
            time.sleep(1. / self.max_fps)
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


def arg_parser():
    """ Define all args that are necessary.
    add_args is also called.
    """
    parser = argparse.ArgumentParser(description='Run Tornado web server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', default=9000,
                        help='Tornado web server port. Default: 9000')
    redis_args(parser)
    return parser.parse_args()


def make_webapp():
    """
    Pass redis_host and redis_port args to tornado web app.
    https://stackoverflow.com/questions/49627836/how-to-pass-arguments-to-tornados-websockethandler-class
    This method doesn't really work.
    According to the tornado websocket src, the websocket has positional args and need to be passed
    before key-value args.
    https://www.tornadoweb.org/en/stable/_modules/tornado/websocket.html

    In order to fix it, make this args to be global. This works, not optimal.
    :return: tornado app obj
    """
    app = web.Application([
        (r'/', IndexHandler),
        (r'/videodisplay', VideoDisplayHandler),
        (r'/ws', SocketHandler),
        (r'/uploadform', UploadForm),
        (r'/upload', UploadHandler, dict(upload_path='./uploads',
                                         naming_strategy=None))
    ],
        debug=True,
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        xsrf_cookies=True,
        cookie_secret="SOME_SECRET_GOES_HERE_MATE",
    )
    return app


if __name__ == '__main__':
    args = arg_parser()
    globals().update(args.__dict__)
    print(args)
    app = make_webapp()
    app.listen(args.port)
    logger.info('Connect localhost: %s', args.port)
    ioloop.IOLoop.instance().start()
