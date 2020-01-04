from flask import Blueprint, jsonify, request
from api.ocrs.helpers import get_table_structure
from api.ocrs.process_image import process_image_with_vision
import os

controllers = Blueprint('controllers', __name__)
workspace_dir       = '/home/ubuntu/workspace/output'
input_dir           = '/tmp/ngnix'

@controllers.route('/process', methods=['POST'])
def process_ocr():
    json_data           = request.get_json(force=True)
    filename            = json_data['filename']
    absolute_filepath   = os.path.join(input_dir, filename)
    
    print('received file [%s] for processing is present at [%s]' % (filename, absolute_filepath))
    return process_image_with_vision(absolute_filepath, workspace_dir)