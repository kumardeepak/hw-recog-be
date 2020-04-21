from flask import Blueprint
from flask_restful import Api

from resources import RectResource

RECT_BLUEPRINT = Blueprint("rect", __name__)
Api(RECT_BLUEPRINT).add_resource(
    RectResource, "/rect/extract"
)
