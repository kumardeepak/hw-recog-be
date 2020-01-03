from flask import Blueprint

users = Blueprint('users', __name__)

@users.route('/me')
def me():
    return 'This is my page', 200