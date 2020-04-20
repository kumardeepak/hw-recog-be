from flask import Flask
from api.users.views import users
from api.ocrs.controllers import controllers as ocr_controllers
from api.info.info_controllers import controllers as info_controllers
from api.tables_lines.tables_lines_controllers import controllers as tables_lines_controllers
from api.table_rois.tables_extractor_controller import controllers as tables_rois_controllers

app = Flask(__name__)

app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(ocr_controllers, url_prefix='/ocr')
app.register_blueprint(info_controllers, url_prefix='/api')
app.register_blueprint(tables_lines_controllers, url_prefix='/tables_lines')
#app.register_blueprint(tables_rois_controllers , url_prefix='/tables_extractor' )