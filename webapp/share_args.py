"""
Collection of arguments that need to be shared with.
"""

def redis_args(parser):
    """Redis server config args."""
    parser.add_argument('-d', '--redis-host', default='redis',
                        help='Redis server host name. Default: "redis"')
    parser.add_argument('-r', '--redis-port', default='6379',
                        help='Redis server port. Default: 6379')
