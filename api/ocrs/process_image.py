import cv2
import numpy as np
import os
import glob
from google.cloud import vision
from google.cloud.vision import types
import io
import uuid
import glob
import time


data_dir                     = 'data'
input_data_dir               = 'input'
output_data_dir              = 'output'

output_extracted_tables_dir  = 'tables'
output_extracted_boxes_dir   = 'boxes'
output_extracted_letters_dir = 'letters'



# utility function
def create_directory(path):
    try:
        os.mkdir(path)
        return True
    except FileExistsError as fe_error:
        return True
    except OSError as error:
        print(error)
    return False

# read files present in a directory
def read_directory_files(path, pattern='*'):
    files = [f for f in glob.glob(os.path.join(path, pattern))]
    return files

def get_subdirectories(path):
    return [f.path for f in os.scandir(path) if f.is_dir() ] 

# detect horizontal and vertical lines in the cropped images and extract boxes
def smoothen_out_image(image):
    edges  = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel)
    smooth = np.copy(image)
    smooth = cv2.blur(smooth, (2, 2))
    (rows, cols) = np.where(edges != 0)
    image[rows, cols] = smooth[rows, cols]
    return image

def combine_image(vertical_img, horizontal_img):
    alpha  = 0.5
    beta   = 1.0 - alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final = cv2.addWeighted(vertical_img, alpha, horizontal_img, beta, 0.0)
    img_final = cv2.erode(~img_final, kernel, iterations=2)
    (thresh, img_final) = cv2.threshold(img_final, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_final

def extract_boxes_from_image(filepath):
    src_img      = cv2.imread(filepath, cv2.IMREAD_COLOR)
    gray_img     = src_img
    if len(src_img.shape) == 3:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.bitwise_not(gray_img)
    bw_img   = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    
    horizontal_img = np.copy(bw_img)
    vertical_img   = np.copy(bw_img)
    
    cols = horizontal_img.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_img = cv2.erode(horizontal_img, horizontalStructure)
    horizontal_img = cv2.dilate(horizontal_img, horizontalStructure)
    
    rows = vertical_img.shape[0]
    vertical_size = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_img = cv2.erode(vertical_img, verticalStructure)
    vertical_img = cv2.dilate(vertical_img, verticalStructure)
    
    horizontal_img = smoothen_out_image(horizontal_img)
    vertical_img   = smoothen_out_image(vertical_img)
    
    img_final      = combine_image(vertical_img, horizontal_img)
    
    # find mask
    masked_img     = cv2.bitwise_and(horizontal_img,vertical_img)
    
    
    return src_img, img_final

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# resize the image by joining the image
def resize_image(img, size=(28,28)):
    h, w = img.shape[:2]
    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)
    dif = h if h > w else w
    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

# return b/w images
def get_gray_and_bw_image(filepath):
    gray_img = cv2.imread(filepath, 0)
    gray_img = cv2.bitwise_not(gray_img)
    #bw_img   = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    bw_img   = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,-2)
    return gray_img, bw_img

# extract table from the image
def extract_tables(filepath, output_dir):
    image, image_processed  = extract_boxes_from_image(filepath)
    contours                = cv2.findContours(image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours                = contours[0] if len(contours) == 2 else contours[1]

    (contours, boundingBoxes) = sort_contours(contours, method='top-to-bottom')

    cont_ind = 0
    count = 0
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if (w*h > 200000) and (w*h < 800000):
            filename = os.path.join(output_dir, str(int(cont_ind/len(contours))) + "_" + str(int(cont_ind%len(contours))) + "_" + os.path.basename(filepath))
            crop_img = image[y:y+h, x:x+w]
            cv2.imwrite(filename, crop_img)
            cont_ind = cont_ind + 1
            count    = count + 1
    print("found (%d) tables in [%s]" % (count, os.path.basename(filepath)))

# extract table boxes from table
def extract_table_boxes(filepath, output_dir, num_cols=5):
    image, image_processed = extract_boxes_from_image(filepath)
    # Find contours for image, which will detect all the boxes
    contours                = cv2.findContours(image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours                = contours[0] if len(contours) == 2 else contours[1]

    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method='top-to-bottom')
    
    cont_ind = 0
    count = 0
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if (w > 80 and h > 20) and (w*h < 100000):
            filename = os.path.join(output_dir, str(int(cont_ind/num_cols))+"_"+str(int((num_cols - 1) - cont_ind%num_cols))+"_"+os.path.basename(filepath))
            crop_img = image[y:y+h, x:x+w]
            cv2.imwrite(filename, crop_img)
            cont_ind = cont_ind + 1
            count    = count + 1
    print("found (%d) boxes in [%s]" % (count, os.path.basename(filepath)))

# let's extract characters from the detected table box
def extract_box_letters(filepath, output_dir):    
    gray_img, bw_img = get_gray_and_bw_image(filepath)
    # find contours and get the external one
    contours                = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours                = contours[0] if len(contours) == 2 else contours[1]
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method='left-to-right')
        
    cont_ind = 0
    count = 0
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if h > 5 and w < 50:
            filename = os.path.join(output_dir, str(int(cont_ind/len(contours)))+"_"+str(int(cont_ind%len(contours)))+"_"+os.path.basename(filepath))
            crop_img = gray_img[y:y+h, x:x+w]
            crop_img = resize_image(crop_img)
            cv2.imwrite(filename, crop_img)
            cont_ind = cont_ind + 1
            count    = count + 1
    print("found (%d) letters in [%s]" % (count, os.path.basename(filepath)))

def ocr_from_google_vision(client, filepath):
    with io.open(filepath, 'rb') as image_file1:
            content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image=content_image)
    document = response.full_text_annotation
    return document.text

def process_image(input_filepath, workspace_dir):
    input_filename  = os.path.splitext(os.path.basename(input_filepath))[0]

    img_filename    = os.path.join(workspace_dir, data_dir, input_data_dir, input_filename)
    print("input filename : [%s]" % (img_filename))

    processing_basedir  = os.path.join(workspace_dir, data_dir, output_data_dir, os.path.splitext(input_filename)[0])
    print("processing dir: [%s]" % (processing_basedir))
    ret = create_directory(os.path.join(workspace_dir, data_dir, output_data_dir))
    ret = create_directory(processing_basedir)

    output_tables_dir = os.path.join(processing_basedir, output_extracted_tables_dir)
    print("tables dir: [%s]" % (output_tables_dir))
    ret = create_directory(output_tables_dir)

    output_boxes_dir = os.path.join(processing_basedir, output_extracted_boxes_dir)
    print("boxes dir: [%s]" % (output_boxes_dir))
    ret = create_directory(output_boxes_dir)

    output_letters_dir = os.path.join(processing_basedir, output_extracted_letters_dir)
    print("letters: [%s]" % (output_letters_dir))
    ret = create_directory(output_letters_dir)

    extract_tables(input_filepath, output_tables_dir)

    table_files = read_directory_files(output_tables_dir)
    for file in table_files:
        output_dir = os.path.join(output_boxes_dir, os.path.splitext(os.path.basename(file))[0])
        ret = create_directory(output_dir)
        filename = os.path.basename(file)
        row      = int(filename.split('_')[0])
        col      = int(filename.split('_')[1])
        if row == 0 and col == 0:
            extract_table_boxes(file, output_dir)
        if row == 0 and col == 1:
            extract_table_boxes(file, output_dir, num_cols=2)

    boxes_dirs = get_subdirectories(output_boxes_dir)
    for boxes_dir in boxes_dirs:
        boxes_files = read_directory_files(boxes_dir)
        output_dir1 = os.path.join(output_letters_dir, os.path.basename(boxes_dir))
        ret         = create_directory(output_dir1)
        
        for file in boxes_files:
            output_dir = os.path.join(output_dir1, os.path.splitext(os.path.basename(file))[0])
            ret = create_directory(output_dir)
            extract_box_letters(file, output_dir)

def detect_characters_in_image(input_filepath, workspace_dir):
    input_filename  = os.path.splitext(os.path.basename(input_filepath))[0]

    img_filename    = os.path.join(workspace_dir, data_dir, input_data_dir, input_filename)
    print("input filename : [%s]" % (img_filename))

    processing_basedir  = os.path.join(workspace_dir, data_dir, output_data_dir, os.path.splitext(input_filename)[0])
    print("processing dir: [%s]" % (processing_basedir))

    output_tables_dir = os.path.join(processing_basedir, output_extracted_tables_dir)
    print("tables dir: [%s]" % (output_tables_dir))

    output_boxes_dir = os.path.join(processing_basedir, output_extracted_boxes_dir)
    print("boxes dir: [%s]" % (output_boxes_dir))

    output_letters_dir = os.path.join(processing_basedir, output_extracted_letters_dir)
    print("letters: [%s]" % (output_letters_dir))

    client     = vision.ImageAnnotatorClient()
    boxes_dirs = get_subdirectories(output_boxes_dir)
    tables     = []

    for boxes_dir in boxes_dirs:
        boxes  = []
        boxes_files = read_directory_files(boxes_dir)
        print("table: [%s]" % (os.path.basename(boxes_dir)))
        
        for file in boxes_files:
            text = ocr_from_google_vision(client, file)
            boxes.append([os.path.basename(file), text.replace('\n', ' ')])
            print("\t\t boxes: [%s], text: [%s]" % (os.path.basename(file), text.replace('\n', ' ')))
            time.sleep(0.01)
        tables.append(boxes)

    return tables

def process_image_with_vision(input_filepath, workspace_dir):
    process_image(input_filepath, workspace_dir)
    tables = detect_characters_in_image(input_filepath, workspace_dir)
    rsp = {
            "status": {
                "code": 200,
                "message": "request successful"
            },
            "response": []
    }
    if len(tables) == 0:
        rsp['status']  = 501,
        rsp['message'] = "unable to process the given image, please check supported image format" 
        return rsp

    for table in tables:
        rows  = []
        cols  = []
        texts = []
        print('arranging entries in table {%d}' % (len(table)))
        table_data = []
        for table_detail in table:
            row, col, text = int(table_detail[0].split('_')[0]), int(table_detail[0].split('_')[1]), table_detail[1]
            table_data.append({'row':row, 'col':col, 'text': text})

            rows.append(row)
            cols.append(col)
            texts.append(text)
        table_header     = {'row': len(set(rows)), 'col': len(set(cols)), 'title': 'table'}
        table_info       = {'header': table_header, 'data': table_data}
        rsp['response'].append(table_info)
    
    return rsp



# input_filepath               = '/Users/kd/Desktop/sample_input_02.jpg'
# base_dir                     = '/Users/kd/Workspace/python/github/handwriting-recognition'
# proceed_rsp                  = process_image_with_vision(input_filepath, base_dir)
# print(proceed_rsp)