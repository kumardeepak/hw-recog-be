from flask import Blueprint, jsonify, request
from api.ocrs.helpers import get_table_structure

controllers = Blueprint('controllers', __name__)

@controllers.route('/process', methods=['POST'])
def process_ocr():
    json_data = request.get_json(force=True)
    filename = json_data['filename']
    print('received file [%s] for processing' % (filename))
    return get_table_structure()