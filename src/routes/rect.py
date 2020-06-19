from flask import Blueprint
from flask_restful import Api

from resources import RectResource,OcrLineResourcev1,OcrLineResourcev2,OcrLineResourcev3

RECT_BLUEPRINT = Blueprint("rect", __name__)
Api(RECT_BLUEPRINT).add_resource(
    RectResource, "/api/v1/ocr/extract"
)
Api(RECT_BLUEPRINT).add_resource(OcrLineResourcev1,"/api/v1/ocr/lines")
Api(RECT_BLUEPRINT).add_resource(OcrLineResourcev2,"/api/v2/ocr/lines")
Api(RECT_BLUEPRINT).add_resource(OcrLineResourcev3,"/api/v3/ocr/lines")
