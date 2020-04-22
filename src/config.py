import logging
import os

DEBUG = True
API_URL_PREFIX = "/api/v1"
HOST = 'localhost'
PORT = 6000
FILE_STORAGE_PATH =  '/tmp/nginx' #'/home/dddhiraj/Documents/Tarento/Anuwad/Tabular_data_extraction/sample_images'# '/Users/kd/Workspace/python/github/data/input/' #
ENABLE_CORS = False


logging.basicConfig(
    filename=os.getenv("SERVICE_LOG", "server.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
