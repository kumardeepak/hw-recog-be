from flask import Blueprint, jsonify, request
from api.ocrs.helpers import get_table_structure
from api.ocrs.process_image import process_image_with_vision
from api.ocrs.process_image_v1 import process_image_v1
from api.ocrs.process_invoice_v1 import process_invoice_v1

import json
import time

from models.exam import Exams
from models.ocr_data import Ocrdata
from models.student import Student

from models.response import CustomResponse
from models.status import Status

import os

controllers = Blueprint('controllers', __name__)
workspace_dir       = '/home/ubuntu/workspace/output'
input_dir           = '/tmp/nginx'

@controllers.route('/check-ocr-data', methods=['POST'])
def check_ocr_data():
    body = request.get_json()
    if body['student_id'] is None or body['exam_id'] is None or body['ocr_data'] is None:
        res = CustomResponse(
            Status.ERR_GLOBAL_MISSING_PARAMETERS.value, None)
        return res.getres(), Status.ERR_GLOBAL_MISSING_PARAMETERS.value['http']['status']
    student_fromdb = Student.objects(student_id=body['student_id'])
    exam_fromdb = Exams.objects(exam_id=body['exam_id'])
    if not (student_fromdb is not None and len(student_fromdb) > 0):
        res = CustomResponse(
            Status.WRONG_STUDENT_CODE.value, None)
        return res.getres()
    elif not (exam_fromdb is not None and len(exam_fromdb) > 0):
        res = CustomResponse(
            Status.WRONG_CODE.value, None)
        return res.getres()
    else:
        try:
            exam_obj = json.loads(exam_fromdb.to_json())
            student_obj = json.loads(student_fromdb.to_json())
            map_obj = conver_list_to_map(exam_obj[0]['data'])
            ocr_data = body['ocr_data']
            table_data = ocr_data['response'][1]['data']
            students_data = ocr_data['response'][0]['data']
            student_obj[0]['exam_id'] = body['exam_id']
            ocr_data['metadata'] = student_obj[0]
            if len(ocr_data['response'][0]['data']) > len(ocr_data['response'][1]['data']):
                table_data = ocr_data['response'][0]['data']
                students_data = ocr_data['response'][1]['data']
            for marks_data in table_data:
                key = str(marks_data['col'])+str(marks_data['row'])
                if key in map_obj:
                    marks_data['text'] = map_obj[key]
                elif str(marks_data['col']) == 0 or str(marks_data['col']) == 1 or str(marks_data['col']) == 2:
                    marks_data['text'] = ''
            if 'exam_date' in exam_obj[0]:
                for student_data in students_data:
                    if student_data['col'] == 1 and student_data['row'] == 3:
                        student_data['text'] = exam_obj[0]['exam_date']
                
            res = CustomResponse(Status.SUCCESS.value, ocr_data)
            return res.getres()
        except Exception as e:
            res = CustomResponse(Status.ERR_GLOBAL_SYSTEM.value, ocr_data)
            return res.getres(), Status.ERR_GLOBAL_SYSTEM.value['http']['status']


def conver_list_to_map(data):
    map_obj = {}
    for d in data:
        map_obj[str(d['col'])+str(d['row'])] = d['text']
    return map_obj

@controllers.route('/save-student-masterdata', methods=['POST'])
def save_student_masterdata():
    body = request.get_json()
    if body['student'] is None or body['student']['student_id'] is None:
        res = CustomResponse(
            Status.ERR_GLOBAL_MISSING_PARAMETERS.value, None)
        return res.getres(), Status.ERR_GLOBAL_MISSING_PARAMETERS.value['http']['status']
    student_fromdb = Student.objects(student_id=body['student']['student_id'])
    if student_fromdb is not None and len(student_fromdb) > 0:
        res = CustomResponse(
            Status.USER_ALREADY_EXISTS.value, None)
        return res.getres(), Status.USER_ALREADY_EXISTS.value['http']['status']
    else:
        student = Student(student_id=body['student']['student_id'],student_name=body['student']['student_name'])
        student.save()
    res = CustomResponse(Status.SUCCESS.value, None)
    return res.getres()


@controllers.route('/save-ocr-data', methods=['POST'])
def save_ocr_data():
    body = request.get_json()
    if body['ocr_data'] is None:
        res = CustomResponse(
            Status.ERR_GLOBAL_MISSING_PARAMETERS.value, None)
        return res.getres(), Status.ERR_GLOBAL_MISSING_PARAMETERS.value['http']['status']
    ocr_data = Ocrdata(created_on=str(int(time.time())), data=body['ocr_data']['response'])
    ocr_data.save()
    res = CustomResponse(Status.SUCCESS.value, None)
    return res.getres()


@controllers.route('/save-exam-masterdata', methods=['POST'])
def save_exam_masterdata():
    body = request.get_json()
    if body['exam'] is None or body['exam']['exam_id'] is None or body['exam']['data'] is None:
        res = CustomResponse(
            Status.ERR_GLOBAL_MISSING_PARAMETERS.value, None)
        return res.getres(), Status.ERR_GLOBAL_MISSING_PARAMETERS.value['http']['status']
    exam_fromdb = Exams.objects(exam_id=body['exam']['exam_id'])
    if exam_fromdb is not None and len(exam_fromdb) > 0:
        exam_fromdb.update(set__data=body['exam']['data'])
    else:
        exam = Exams(exam_id=body['exam']['exam_id'],data=body['exam']['data'],exam_date=body['exam']['exam_date'])
        exam.save()
    res = CustomResponse(Status.SUCCESS.value, None)
    return res.getres()



@controllers.route('/process', methods=['POST'])
def process_ocr():
    json_data           = request.get_json(force=True)
    filename            = json_data['filename']
    absolute_filepath   = os.path.join(input_dir, filename)
    
    print('received file [%s] for processing is present at [%s]' % (filename, absolute_filepath))
    return process_image_v1(absolute_filepath, workspace_dir)


@controllers.route('/invoice', methods=['POST'])
def process_invoice_ocr():
    json_data           = request.get_json(force=True)
    filename            = json_data['filename']
    absolute_filepath   = os.path.join(input_dir, filename)
    
    print('received file [%s] for processing is present at [%s]' % (filename, absolute_filepath))
    return process_invoice_v1(absolute_filepath, workspace_dir)