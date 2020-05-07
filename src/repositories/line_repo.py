import pytesseract
import cv2
from pdf2image import convert_from_path
import os
import glob
from repositories import TableRepositories
from lxml import html


class OCRlineRepositories:

    def __init__(self, pdf_path):
        self.pdf_path          = pdf_path
        self.response          = {'resolution': None , 'lines_data': []}
        self.language_map      = {'Malayalam' : 'mal' , 'Tamil':'tam' , 'Devanagari':'hin','Telugu':'tel','Latin':'eng'}
        self.pdf_to_image ()
        self.pdf_language_detect ()
        self.line_metadata()
        self.delete_images()

    def pdf_to_image(self):
        self.pdf_name = self.pdf_path.split('/')[-1].split('.')[0]
        self.pdf_to_image_dir  = 'tmp/images/' + self.pdf_name
        os.system('mkdir -p {0}_r'.format (self.pdf_to_image_dir))
        os.system('mkdir -p {0}_c'.format (self.pdf_to_image_dir))
        convert_from_path(self.pdf_path, output_folder=self.pdf_to_image_dir, fmt='jpeg', output_file='')
        os.system(' pdftohtml -s -c -p {0} {1}/c'.format(self.pdf_path , self.pdf_to_image_dir))
        #convert_from_path(self.pdf_path , output_folder=self.pdf_to_image_dir, fmt='jpeg', output_file='')
        
        self.num_of_pages = len(glob.glob(self.pdf_to_image_dir + '_c/*.png'))
        self.number_of_digits = len(str(self.num_of_pages))

    def pdf_language_detect(self):
        page_file         = self.pdf_to_image_dir + '/-' + self.page_num_correction (0) + '.jpg'
        osd               =  pytesseract.image_to_osd (page_file)
        language_script   =  osd.split('\nScript')[1][2:]
        self.pdf_language =  self.language_map[language_script]
        print( 'Language detected {0}'.format(self.pdf_language))

    def mask_out_tables(self, table_detect_file, page):
        tables     = TableRepositories (table_detect_file)
        table_rois = tables.response ["response"] ["tables"]

        page_image = cv2.imread (page, 0)
        if len (table_rois) != 0:
            # images extracted by pdftohtml and pdftoimage have different resolutions
            y_scale = page_image.shape [0] / float (tables.input_image.shape [0])
            x_scale = page_image.shape [1] / float (tables.input_image.shape [1])
            # if len(table_rois) != 0
            for table in table_rois:
                x = table ['x'] * x_scale
                y = table ['y'] * y_scale
                w = table ['w'] * x_scale
                h = table ['h'] * y_scale
                page_image [int (y):int (y + h), int (x):int (x + w)] = 255
        return page_image

    def line_parser(self, page_image, pdf_index):
        hocr       = pytesseract.image_to_pdf_or_hocr (page_image, lang=self.pdf_language, extension='hocr')
        tree       = html.fromstring (hocr)
        line_data  = []
        page_index = 0
        for path in tree.xpath ('//*'):
            node = path.values ()
            if 'ocr_line' in node:
                #page_index   = node [1].split ('_') [-1]
                bbox         = node [2].split (';') [0].split (' ')
                left         = int (bbox [1])
                top          = int (bbox [2])
                height       = int (bbox [4]) - top
                line_text = ''
                for child in path:
                    if (len(child.text) > 0) & (child.text != ' '):
                        line_text += ' ' + child.text
                if len (line_text) > 0:
                    pdf_index  += 1
                    page_index += 1
                    line_data.append (
                        {'x': left, 'y': top, 'height': height, 'text': str(line_text [1:]), 'page_index': page_index,
                         'pdf_index': pdf_index})
        return line_data, pdf_index


    def page_num_correction(self,page_num,num_size=None):
        padding = '0'
        if num_size == None :
            max_length = self.number_of_digits
        else :
            max_length = num_size
        corrction_factor = max_length - len(str(page_num + 1 ))
        return padding * corrction_factor + str(page_num + 1)



    def line_metadata(self):
        pdf_index=0
        for page_num in range(self.num_of_pages):
            page_file           = self.pdf_to_image_dir + '_r/-' + self.page_num_correction(page_num) + '.jpg'
            table_detect_file   = self.pdf_to_image_dir + '_c/c' + self.page_num_correction(page_num,3) + '.png'
            print(table_detect_file,page_file)
            page_image              = self.mask_out_tables(table_detect_file, page_file)
            line_data, pdf_index    = self.line_parser(page_image, pdf_index)

            if self.response['resolution'] == None:
                self.response['resolution'] = {'x':page_image.shape[1] , 'y':page_image.shape[0]}
            self.response['lines_data'].append(line_data)

    def delete_images(self):
        os.system('rm -r {0}'.format(self.pdf_to_image_dir))
