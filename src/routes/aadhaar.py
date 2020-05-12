from flask import Blueprint
from flask_restful import Api

from resources import AadharResource

AADHAAR_BLUEPRINT = Blueprint("rect", __name__)
Api(AADHAAR_BLUEPRINT).add_resource(
    AadharResource, "/aadhaar/extract"
)
