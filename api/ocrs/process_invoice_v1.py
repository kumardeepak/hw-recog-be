import os
import glob
from google.cloud import vision
from google.cloud.vision import types
import io
import uuid
import glob
import time
import re

client         = vision.ImageAnnotatorClient()
tin_pattern    = re.compile(r'(?i)(TIN)(\s)?(NO)?(\s)?(:)?(\s)?[a-z,A-Z,0-9]*')
gst_pattern    = re.compile(r'(?i)(GSTIN)(#)?(\s)?(NO)?(\s)?(:)?(\s)?[a-z,A-Z,0-9]*')
date_pattern   = re.compile(r'([0-9]{2})(-|/)([0-9]{2})(-|/)([0-9]{4})(\s)?([0-9]{2})?(:)?([0-9]{2})?')
bill1_pattern  = re.compile(r'(?i)(BILLNO|BILL)(#)?(\s)?(NO)?(\s)?(:)?(\s)?([a-z,A-Z,0-9]|-|/|\s)?(\s)?([a-z,A-Z,0-9]|-|\s)*')
bill2_pattern  = re.compile(r'(#)[a-z,A-Z,0-9](\s)?([a-z,A-Z,0-9]|-|\s)*')
price_pattern  = re.compile(r'(?i)(TOTAL)(\s)?(INVOICE)?(\s)?(RS|INR)?(.|\s)?(\s)?(\d+\.\d+)')

def ocr_from_google_vision(client, filepath):
    with io.open(filepath, 'rb') as image_file1:
            content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image=content_image)
    document = response.full_text_annotation
    return document.text


def extract_in_individual_detection(texts, pattern):
    for text in texts:
        found = pattern.search(text)
        if found != None:
            return found.group()
        
    return ''

def extract_in_all_detection(text, pattern):
    found = pattern.search(text)
    if found != None:
        return found.group()
    return ''

def api_response(code, message, response=None):
    rsp = {
            "status": {
                "code": code,
                "message": message
            },
            "response": {'tin': '', 'gstin' : '', 'date': '', 'bill': '', 'price': ''} if response == None else response
    }
    return rsp


def process_invoice_v1(input_filepath, workspace_dir=None):
    input_filename  = os.path.splitext(os.path.basename(input_filepath))[0]

    img_filename    = input_filepath
    print("input filename : [%s]" % (img_filename))

    ocred_text      = ocr_from_google_vision(client, input_filepath)
    texts           = ocred_text.split('\n')


    tin             = extract_in_individual_detection(texts, tin_pattern)
    gst             = extract_in_individual_detection(texts, gst_pattern)
    date            = extract_in_individual_detection(texts, date_pattern)

    bill            = extract_in_individual_detection(texts, bill1_pattern)
    if len(bill) == 0:
        bill        = extract_in_individual_detection(texts, bill2_pattern)
    
    price           = extract_in_all_detection(' '.join(texts), price_pattern)

    return api_response(200, 'successfully processed', {'tin': tin, 'gst': gst, 'date': date, 'bill': bill, 'price': price})

# input_filepath               = "/Users/kd/Desktop/invoice_1.jpg"
# proceed_rsp                  = process_invoice_v1(input_filepath)
# print(proceed_rsp)