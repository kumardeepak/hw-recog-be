from mongoengine import *

class Ocrdata(DynamicDocument):
    created_on = StringField()
    data = ListField()
