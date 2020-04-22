from flask_restful import fields, marshal_with, reqparse, Resource
import os
import config
import logging
import magic
import cv2
from flask.json import jsonify
from repositories import RectRepositories, TableRepositories

def check_image_file_id(id):
    if os.path.exists(os.path.join(config.FILE_STORAGE_PATH, id)) and os.path.isfile(os.path.join(config.FILE_STORAGE_PATH, id)):
        f           = magic.Magic(mime=True, uncompress=True)
        fileType    = f.from_file(os.path.join(config.FILE_STORAGE_PATH, id))
        if   fileType == 'image/jpeg' or fileType == 'image/jpg'  or fileType == 'image/png':
            logging.debug("file id %s is a valid %s image file" % (id, fileType))
            return id
        else:
            logging.debug("file id %s is not a valid image file" % (id))
            raise ValueError("file id {} doesn't exists".format(id))
    else:
        logging.debug("file id %s doesn't exists" % (id))
        raise ValueError("file id {} doesn't exists".format(id))

parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('Content-Type', location='headers', type=str, help='Please set Content-Type as application/json')
parser.add_argument('image_file_id', location='json', type=check_image_file_id, help='Please provide valid image_file_id in JPEG/PNG format', required=True)

class RectResource(Resource):
    def post(self):
        args            = parser.parse_args()
        Rects           = RectRepositories(os.path.join(config.FILE_STORAGE_PATH, args['image_file_id']))
        lines, _        = Rects.get_tables_and_lines ()
        table           = None
        # for table in tables:
        TableRepo       = TableRepositories (os.path.join (config.FILE_STORAGE_PATH, args ['image_file_id']), table)
        tables          = TableRepo.response ['response'] ['tables']
        return {
            'status': {
                'code' : 200,
                'message' : 'api successful'
            },
            'lines': lines,
            'tables': tables
        }