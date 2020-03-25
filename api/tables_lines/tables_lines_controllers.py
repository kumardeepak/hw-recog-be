import os
import json
import time

from flask import Blueprint, jsonify, request
from api.tables_lines.process_tables_lines import detect_tables_and_lines

controllers         = Blueprint('tables_lines_controllers', __name__)
workspace_dir       = '/home/ubuntu/workspace/output'
input_dir           = '/tmp/nginx'

def api_response(code, message, response=None):
    rsp = {
            "status": {
                "code": code,
                "message": message
            },
            "response": [] if response == None else response
    }
    return rsp

def api_wrapper_detect_tables_and_lines(filepath):
    tables, lines = detect_tables_and_lines(filepath)
    return api_response(200, 'successfully processed the image', {'tables': tables, 'lines': lines}) 


@controllers.route('/detect', methods=['POST'])
def process_image():
    json_data           = request.get_json(force=True)
    filename            = json_data['filename']
    absolute_filepath   = os.path.join(input_dir, filename)
    
    print('received file [%s] for processing is present at [%s]' % (filename, absolute_filepath))
    return api_wrapper_detect_tables_and_lines(absolute_filepath)



