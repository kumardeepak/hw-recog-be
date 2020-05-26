import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
import os
import glob
from repositories import TableRepositories
from lxml import html
import uuid
import pandas as pd

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
        self.pdf_to_image_dir  = 'tmp/images/' + self.pdf_name + str(uuid.uuid1())
        os.system('mkdir -p {0}'.format (self.pdf_to_image_dir))
        convert_from_path(self.pdf_path, output_folder=self.pdf_to_image_dir, fmt='jpeg', output_file='')
        os.system(' pdftohtml -s -c -p {0} {1}/c'.format(self.pdf_path , self.pdf_to_image_dir))
        #convert_from_path(self.pdf_path , output_folder=self.pdf_to_image_dir, fmt='jpeg', output_file='')

        self.num_of_pages = len(glob.glob(self.pdf_to_image_dir + '/*.png'))
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



    def sorted_lines(self,group,len_groups,sorted_lines=[],page_index=1):
        while len(sorted_lines) < len_groups:
            mean_semi_height = group['height'].mean() / 2.0
            check_y          = group.iloc[0]['y']
            same_line        = group[ abs(group['y'] - check_y) < mean_semi_height]
            next_lines       = group[ abs(group['y'] - check_y) >= mean_semi_height]
            #same_line.iloc[-1]['line_change'] = 1

            same_line['page_line_index_absolute'] = page_index
            sort_line       = same_line.sort_values(by=['x'])
            page_index += 1

            for index, row in sort_line.iterrows():
                sorted_lines.append(row.to_dict())

            #if len(next_lines) < 1:
            #    break
            self.sorted_lines(next_lines,len_groups,sorted_lines)

        return sorted_lines



    def sorted_lines_data(self,group,len_groups,lines=[],page_index=1):

        while len(lines) < len_groups:
            mean_semi_height = group['height'].mean() / 2.0
            check_y          = group.iloc[0]['top']
            same_line        = group[ abs(group['top'] - check_y) < mean_semi_height]
            next_lines       = group[ abs(group['top'] - check_y) >= mean_semi_height]
            #same_line.iloc[-1]['line_change'] = 1

            same_line['page_line_index_absolute'] = page_index
            sort_line       = same_line.sort_values(by=['left'])
            page_index += 1

            for index, row in sort_line.iterrows():
                lines.append(row)
            self.sorted_lines_data(next_lines,len_groups,lines,page_index)
        return pd.DataFrame(lines)




    def line_parser_image_to_data(self, page_image, pdf_index,page_number):

        df = pytesseract.image_to_data(page_image, output_type=Output.DATAFRAME)
        df = df[df['conf'] > 10]
        df = df[df['text'] != ' ']
        df = df.sort_values(by=['top'])
        df = self.sorted_lines_data(df,len(df))
        df['line_key'] = df['block_num'].astype(str) + df['par_num'].astype(str) + df['line_num'].astype(str)
        lines_data=[]
        for uinqe_line in df['line_key'].unique():
            line= {}
            same_line          =  df[df['line_key'] == uinqe_line]
            line['text']       = ' '.join(same_line['text'].values)
            line['top']        = same_line['top'].min()
            line['left']       = same_line['left'].min()
            line['height']     = same_line['height'].max()
            line['block_num']  = same_line['block_num'].iloc[0]
            line['par_num']    = same_line['par_num'].iloc[0]
            line['line_num']   = same_line['line_num'].iloc[0]
            line['pdf_index']  = pdf_index
            line['page_no']    = page_number
            line['avrage_conf']       = same_line['conf'].mean()

            pdf_index         += 1

            line['page_line_index_absolute'] = same_line['page_line_index_absolute'].iloc[0]
            lines_data.append(line)

        return lines_data , pdf_index



    def line_parser_hocr(self, page_image, pdf_index,page_number):
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
                    #pdf_index  += 1
                    page_index += 1
                    line_data.append (
                        {'x': left, 'y': top, 'height': height, 'text': str(line_text [1:]), 'page_index': page_index,
                         'pdf_index': pdf_index , 'page_no':page_number})
        if len(line_data) > 0 :
            line_df = pd.DataFrame(line_data)
            line_df['page_line_index_absolute'] = 0
            line_df = line_df.sort_values(by=['y'])
            line_data = self.sorted_lines(line_df,len(line_data),[])

            for index,line in enumerate(line_data):
                line['page_index'] = index
                line['pdf_index']  = pdf_index
                pdf_index         += 1

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
            page_file           = self.pdf_to_image_dir + '/-' + self.page_num_correction(page_num) + '.jpg'
            table_detect_file   = self.pdf_to_image_dir + '/c' + self.page_num_correction(page_num,3) + '.png'
            print(table_detect_file,page_file)
            page_image              = self.mask_out_tables(table_detect_file, page_file)
            #line_data, pdf_index    = self.line_parser_hocr(page_image, pdf_index,page_num + 1)
            line_data, pdf_index    = self.line_parser_image_to_data(page_image, pdf_index,page_num + 1)

            if self.response['resolution'] == None:
                self.response['resolution'] = {'x':page_image.shape[1] , 'y':page_image.shape[0]}
            self.response['lines_data'].append(line_data)

    def delete_images(self):
        os.system('rm -r {0}'.format(self.pdf_to_image_dir))
        os.system('rm -r {0}'.format(self.pdf_to_image_dir))
