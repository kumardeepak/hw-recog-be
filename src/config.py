import logging
import os

DEBUG = True
API_URL_PREFIX = "/api/v1"
HOST = '0.0.0.0'
PORT = 5000
FILE_STORAGE_PATH = "/opt/share/nginx/upload"
ENABLE_CORS = False
#LANGUAGE = 'hin' #Specify language of document being passed

logging.basicConfig(
    filename=os.getenv("SERVICE_LOG", "server.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
