import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
import os
import glob
#from repositories import TableRepositories
from table import TableRepositories

import uuid
import pandas as pd
import numpy as np

class OCRlineRepositories:

    def __init__(self, pdf_path):
        self.pdf_path          = pdf_path
        self.response          = {'resolution': None , 'lines_data': []}
        self.language_map      = {'Malayalam' : 'mal' , 'Tamil':'tam' , 'Devanagari':'hin','Telugu':'tel','Latin':'eng'}
        self.margin_support    = 4
        self.tesseract_conf    = 10
        self.page_df           = None
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
        self.num_of_pages = len(glob.glob(self.pdf_to_image_dir + '/*.png'))
        self.number_of_digits = len(str(self.num_of_pages))

    def pdf_language_detect(self):
        page_file         = self.pdf_to_image_dir + '/-' + self.page_num_correction (0) + '.jpg'
        osd               =  pytesseract.image_to_osd (page_file)
        language_script   =  osd.split('\nScript')[1][2:]
        self.pdf_language =  'eng+'+self.language_map[language_script]
        self.response['resolution'] = {'x': page_image.shape[1], 'y': page_image.shape[0]}
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

    def bloat_text(self, image):
        # converitng image to binary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image > 125
        image = image.astype(np.uint8)
        # Bloating
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, self.line_spacing_median * 0.5, 255, 0)
        return sure_fg.astype(np.uint8)

    def sort_words(self,group, sorted_group=[], line_spacing=[], line=0):

        semi_height = group.iloc[0]['height'].mean() / 2.0
        check_y = group.iloc[0]['top']
        same_line = group[abs(group['top'] - check_y) < semi_height]
        next_lines = group[abs(group['top'] - check_y) >= semi_height]
        sort_lines = same_line.sort_values(by=['left'])
        sort_lines['line'] = line
        line_spacing.append(same_line['ymid'].mean())
        line += 1
        for index, row in sort_lines.iterrows():
            sorted_group.append(row)

        if len(next_lines) > 0:
            self.sort_words(next_lines, sorted_group, line_spacing, line)

        return sorted_group, line_spacing, line

    def sort_contours(self,contours_df, sorted_contours=[]):

        check_y = contours_df.iloc[0]['top']
        same_line = contours_df[abs(contours_df['top'] - check_y) < self.line_spacing_median]
        next_lines = contours_df[abs(contours_df['top'] - check_y) >= self.line_spacing_median]
        sort_lines = same_line.sort_values(by=['left'])
        for index, row in sort_lines.iterrows():
            sorted_contours.append(row)
        if len(next_lines) > 0:
            self.sort_contours(next_lines, sorted_contours)

        return sorted_contours

    def median_spacing(self,line_spacing):
        spacing = []
        for i in range(len(line_spacing) - 1):
            spacing.append(line_spacing[i + 1] - line_spacing[i])
        spacing_median = np.median(spacing)

        return spacing_median

    def find_and_sort_contours(self,bloated_image):
        contours = cv2.findContours(bloated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        image_area = bloated_image.shape[0] * bloated_image.shape[1]
        contours_list = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) < (image_area * 0.95):
                contours_list.append([x, y, w, h])
        contours_df = pd.DataFrame(contours_list, columns=['left', 'top', 'width', 'height'])
        contours_df = contours_df.sort_values(by=['top'])
        sorted_contours = self.sort_contours(contours_df, [])
        sorted_contours = pd.DataFrame(sorted_contours)

        return sorted_contours

    def extract_region(self, row):
        sub_text_df = self.page_df[
            (self.page_df['left'] >= row['left']) & (self.page_df['right'] <= (row['left'] + row['width'])) & (
                        self.page_df['top'] >= row['top']) & (self.page_df['bottom'] <= (row['top'] + row['height']))]

        return sub_text_df

    def line_start_and_end_stats(self):
        lines = self.page_df['line'].unique()
        lines_df = pd.DataFrame(lines, columns=['line'])
        lines_df['start'] = 0
        lines_df['end'] = 0
        for line in lines:
            sub_sorted_df = self.page_df[self.page_df['line'] == line]
            lines_df['start'][line] = sub_sorted_df['left'].min()
            lines_df['end'][line] = sub_sorted_df['right'].max()

        return lines_df

    def extraction_helper(self, input_image):
        text_df = pytesseract.image_to_data(input_image, lang=self.pdf_language, output_type=Output.DATAFRAME)
        text_df = text_df[text_df['conf'] > self.tesseract_conf]
        if len(text_df) > 0:
            text_df['bottom']  = text_df['top'] + text_df['height']
            text_df['right']   = text_df['left'] + text_df['width']
            text_df['ymid']    = text_df['top'] + text_df['height'] * 0.5
            text_df            = text_df.sort_values(by=['top'])
            text_df['text']    = text_df['text'].astype(str)
            text_df['line']    = None
            self.median_height      = text_df['height'].median()
            # Removing noise
            text_df            = text_df[text_df['height'] > (self.median_height / 4.0)]

            sorted_df, line_spacing, line = self.sort_words(text_df, sorted_group=[], line_spacing=[], line=0)

            self.page_df = pd.DataFrame(sorted_df).reset_index()
            self.line_df = self.line_start_and_end_stats()
            self.line_spacing_median = self.median_spacing(line_spacing)
            bloated_image = self.bloat_text(input_image)
            self.sorted_contours = self.find_and_sort_contours(bloated_image)

            return 'Found text'
        else:
            return None

    def get_left_margin(self):
        line_start_distribution = np.array(sorted(self.line_df['start']))
        threshold = self.page_df['height'].median()
        # check support

        for starting_point in line_start_distribution:
            delta = abs(line_start_distribution - starting_point)
            margin_delta = delta < threshold
            if margin_delta.sum() > self.margin_support:
                break

        return starting_point

    def get_right_margin(self):
        line_end_distribution = np.array(sorted(self.line_df['end'], reverse=True))
        threshold = self.page_df['height'].median()
        # check support

        for end_point in line_end_distribution:
            delta = abs(line_end_distribution - end_point)
            margin_delta = delta < threshold
            if margin_delta.sum() > self.margin_support:
                break

        return end_point

    def line_parser(self, page_number, pdf_index):
        lines_data = []
        # page_number = 1
        # pdf_index =   0
        for index, row in self.sorted_contours.iterrows():
            extracted_region = self.extract_region(row)
            blob_start = row['left']
            if len(extracted_region) > 0:
                # print(extracted_region)
                lines_in_blob = extracted_region['line'].unique()
                lines_count = len(lines_in_blob)
                first_line = lines_in_blob[0]
                last_line = lines_in_blob[-1]
                for line_id in lines_in_blob:
                    line = {}
                    same_line = extracted_region[extracted_region['line'] == line_id]
                    # print(same_line)
                    line['text'] = ' '.join(same_line['text'].values)
                    line['top'] = int(same_line['top'].min())
                    line['left'] = int(same_line['left'].min())
                    line['height'] = int(same_line['height'].max())
                    line['right'] = int(same_line['right'].max())
                    line['bottom'] = int(same_line['bottom'].max())

                    line['block_num'] = int(same_line['block_num'].iloc[0])
                    line['blob_id'] = index
                    line['pdf_index'] = pdf_index
                    line['page_no'] = page_number
                    line['avrage_conf'] = float(same_line['conf'].mean())
                    line['page_line_index'] = line
                    line['visual_break'] = self.break_condition( line_id, last_line, page_number, lines_count)

                    pdf_index += 1
                    lines_data.append(line)

        return lines_data, pdf_index




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
            page_image          = self.mask_out_tables(table_detect_file, page_file)
            print(table_detect_file,page_file)

            try :
                check_for_text = self.extraction_helper(page_image)
            except :
                check_for_text = None

            if check_for_text != None :
                line_data, pdf_index    = self. line_parser(page_num +1, pdf_index)
                self.response['lines_data'].append(line_data)
            else :
                self.response['lines_data'].append("### No text detected in this page")

            if self.response['resolution'] == None:

            self.response['lines_data'].append(line_data)



    # visual_break
    def break_condition(self, line_id, last_line, page_number, lines_count):
        left_margin = self.get_left_margin()
        right_margin = self.get_right_margin()
        line_start = self.line_df['start'][line_id]
        line_ending = self.line_df['end'][line_id]
        start_delta = abs(line_start - left_margin)
        end_delta = abs(right_margin - line_ending)

        if line_id == last_line:
            return 1
        else:
            # First pages uses centre alignment fo headings and titles
            if page_number == 1:
                if start_delta > 3 * self.median_height:
                    if end_delta > 2 * self.median_height:
                        return 1
                else:
                    if lines_count > 3:
                        if end_delta > 2 * self.median_height:
                            return 1
            else:
                # Supreme court uses justified text alignment
                if start_delta < 2 * self.median_height:
                    if end_delta > 2 * self.median_height:
                        return 1
                else:
                    if abs(line_start - self.line_df['start'][line_id + 1]) > 2 * self.median_height:
                        if abs(line_ending - self.line_df['end'][line_id + 1]) > 2 * self.median_height:
                            return 1

        return 0

    def delete_images(self):
        os.system('rm -r {0}'.format(self.pdf_to_image_dir))
