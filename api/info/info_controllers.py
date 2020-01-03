from flask import Blueprint, jsonify, request

controllers = Blueprint('info_controllers', __name__)

@controllers.route('/info', methods=['GET', 'POST'])
def info():
    return jsonify(version=0.1, message="Welcome to OCR REST API")