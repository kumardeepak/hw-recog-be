from mongoengine import *

class Exams(DynamicDocument):
    exam_id = StringField(required=True)
    data = ListField()


