import os
import json
import time

from flask import Blueprint, jsonify, request
from api.table_rois.table_extractor import Table_extractor

controllers         = Blueprint('tables_rois_controllers', __name__)
workspace_dir       = '/home/ubuntu/workspace/output'
input_dir           = 'upload/'

def api_response(code, message, response=None):
    rsp = {
            "status": {
                "code": code,
                "message": message
            },
            "response": [] if response == None else response
    }
    return rsp

def api_wrapper_detect_tables_rosi(filepath, response=None):
    tables = Table_extractor(filepath)
    response = tables.response
    return api_response(200, 'successfully processed the image', response)


@controllers.route('/rois', methods=['POST'])
def process_image():
    json_data           = request.get_json(force=True)
    filename            = json_data['filename']
    absolute_filepath   = input_dir +  filename
    
    print('received file [%s] for processing is present at [%s]' % (filename, absolute_filepath))
    return api_wrapper_detect_tables_rosi(absolute_filepath)



