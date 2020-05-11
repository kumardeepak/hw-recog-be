from flask import Blueprint
from flask_restful import Api

from resources import RectResource,OcrLineResource

RECT_BLUEPRINT = Blueprint("rect", __name__)
Api(RECT_BLUEPRINT).add_resource(
    RectResource, "/ocr/extract"
)
Api(RECT_BLUEPRINT).add_resource(OcrLineResource,"/ocr/lines")
