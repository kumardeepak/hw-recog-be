import logging
import os

DEBUG = True
API_URL_PREFIX = "/api/v1"
HOST = 'localhost'
PORT = 6000
FILE_STORAGE_PATH =  '/Users/kd/Workspace/python/github/data/input/' #/tmp/nginx
ENABLE_CORS = False


logging.basicConfig(
    filename=os.getenv("SERVICE_LOG", "server.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
