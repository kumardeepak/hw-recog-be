from mongoengine import *

class Student(DynamicDocument):
    student_id = StringField(required=True)
