import logging

import redis

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Processor(object):
    """
    Abstract class for all processors such as video processor or cam processor.
    """

    def __init__(self, redis_host, redis_port):
        """
        Processor init.
        :param redis_host: redis name
        :param redis_port: redis port
        """
        super(Processor, self).__init__()
        self.__redis_host = redis_host
        self.__redis_port = int(redis_port)

        # redis connector.
        self._store = redis.Redis()
        # self.__store = redis.Redis(host=self.redis_host,
        #                            port=self.redis_port)

        logger.info('Processor is inited.')
    @property
    def redis_host(self):
        return self.__redis_host

    @property
    def redis_port(self):
        return self.__redis_port

    @redis_host.setter
    def redis_host(self, host):
        self.__redis_host = host

    @redis_port.setter
    def redis_port(self, port):
        self.__redis_port = port
