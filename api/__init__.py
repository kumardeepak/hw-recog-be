from flask import Flask
from api.users.views import users
from api.ocrs.controllers import controllers as ocr_controllers

app = Flask(__name__)
app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(ocr_controllers, url_prefix='/ocr')
