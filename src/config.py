import logging
import os

DEBUG = True
API_URL_PREFIX = "/api/v1"
HOST = 'localhost'
PORT = 3000
FILE_STORAGE_PATH = '/Users/kd/Workspace/python/github/data/input/'
ENABLE_CORS = False

DATABASE_SAVE = False
DATABASE_URI = 'mongodb://localhost:27017'
DATABASE_NAME = 'facedb'

logging.basicConfig(
    filename=os.getenv("SERVICE_LOG", "server.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
