import cv2
import tensorflow as tf
import os
import time
import numpy as np
import imutils
import pytesseract
import pandas as pd
from pytesseract import Output
import re
import logging
import config
from repositories.east_utils import model
from repositories.east_utils import postprocess
#from repositories.bbox_tools import Box_cordinates
#from repositories.text_postprocessing import Text_to_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.basicConfig(filename='example.log',level=logging.DEBUG)
# margin
# tesseract confidence
# margin

#loadig the EAST model in memory

#Downloading weights:
#if not os.path.exists(os.path.join('/home/dddhiraj/.models/east_icdar2015_resnet_v1_50_rbox/' , 'east_icdar2015_resnet_v1_50_rbox')):
    #east_weight_url = 'https://storage.googleapis.com/drive-bulk-export-anonymous/20200521T071813Z/4133399871716478688/d30b6560-7399-436b-81e4-46dca7df8736/1/a13737a7-e9d3-4650-b146-081ae2643355?authuser'
    #os.system('mkdir -p {}'.format('/home/dddhiraj/.models/east_icdar2015_resnet_v1_50_rbox/'))
    #os.system('wget --no-check-certificate {0} -O {1}/east'.format(east_weight_url ,config.EAST_WEIGHTS))
    #os.system('unzip {0} -d {1}'.format(os.path.join(config.EAST_WEIGHTS,'east'),config.EAST_WEIGHTS))



checkpoint_path = '/home/dddhiraj/.models/east_icdar2015_resnet_v1_50_rbox'
tf.reset_default_graph ()
input_images = tf.placeholder (tf.float32, shape=[None, None, None, 3], name='input_images')
global_step = tf.get_variable ('global_step', [], initializer=tf.constant_initializer (0), trainable=False)
f_score, f_geometry = model.model (input_images, is_training=False)
variable_averages = tf.train.ExponentialMovingAverage (0.997, global_step)
saver = tf.train.Saver (variable_averages.variables_to_restore ())
sess = tf.Session (config=tf.ConfigProto (allow_soft_placement=True))
ckpt_state = tf.train.get_checkpoint_state (checkpoint_path)
model_path = os.path.join (checkpoint_path, os.path.basename (ckpt_state.model_checkpoint_path))
#print ('Restore from {}'.format (model_path))
saver.restore (sess, model_path)


class AadhaarRepositories:

    def __init__(self, image_path, session=sess,conf_threshold=50):
        self.image_path       = image_path
        self.image            = cv2.imread (image_path)[:, :, ::-1]
        self.sess             = session
        self.conf_threshold   = int(conf_threshold)
        self.timer            = {'net': 0, 'restore': 0, 'nms': 0}
        self.text             = {}
        self.extract_text()

    def east_output(self):
        out_put                        = []
        start                          = time.time ()
        im_resized, (ratio_h, ratio_w) = postprocess.resize_image (self.image)
        score, geometry                = self.sess.run ([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        self.timer ['net']             = time.time () - start
        boxes, self.timer              = postprocess.detect (score_map=score, geo_map=geometry, timer=self.timer)
        # #print (' net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format (self.timer ['net'] * 1000,
        #                                                                self.timer ['restore'] * 1000,
        #                                                                self.timer ['nms'] * 1000))

        if boxes is not None:
            boxes            = boxes [:, :8].reshape ((-1, 4, 2))
            boxes [:, :, 0] /= ratio_w
            boxes [:, :, 1] /= ratio_h
        #
        if boxes is not None:
            for box in boxes:

                # to avoid submitting errors
                box = postprocess.sort_poly (box.astype (np.int32))
                #print(box)
                if np.linalg.norm (box [0] - box [1]) < 5 or np.linalg.norm (box [3] - box [0]) < 5:
                    continue
                out_put.append (
                    [box [0, 0], box [0, 1], box [1, 0], box [1, 1], box [2, 0], box [2, 1], box [3, 0], box [3, 1]])
        return out_put


    def dump_out(self,bbc,rot):
        im = self.image.copy()
        for box in bbc:
            #print(box)
            cv2.polylines(im, [np.array(box).astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
        cv2.imwrite('tmp/' + str(rot) + '.png' , im)


    def get_rotaion_angle(self, east_coordinates):

        bboxex                = Box_cordinates (east_coordinates)
        bboxex.df ['delta_x'] = bboxex.df ['x2'] - bboxex.df ['x1']
        bboxex.df ['delta_y'] = bboxex.df ['y2'] - bboxex.df ['y1']
        box_dir               = [bboxex.df ['delta_x'].mean (), bboxex.df ['delta_y'].mean ()]
        # print(box_dir)
        x_axis                = [1, 0]
        cosine                = np.dot (box_dir, x_axis) / (np.linalg.norm (box_dir) * np.linalg.norm (x_axis))
        angle                 = np.arccos (cosine) * 180 / np.pi
        avrage_height         = bboxex.df ['height'].mean ()
        avrage_width          = bboxex.df ['width'].mean ()
        if avrage_height > avrage_width:
            angle = 90 - angle

        return angle * np.sign (box_dir [1])

    def check_orientation(self, group_cordinates, margin=5):
        upside_down = False
        orientation = []
        for index, block in enumerate (group_cordinates):
            crop = self.image [block [0] [1] - margin: block [1] [1] + margin,
                   block [0] [0] - margin: block [1] [0] + margin]
            try:
                osd   = pytesseract.image_to_osd (crop)
                angle = osd.split ('\nRotate') [0].split (': ') [-1]
                orientation.append (int (angle))
            except:
                pass
        orientation     = np.array (orientation)
        chk_orientation = orientation > 170

        # Taking vote of regions
        if chk_orientation.sum () > (len (orientation) * 0.5):
            #print ('Image is upside down')
            upside_down = True
            return upside_down

        return upside_down

    def extract_text(self):
        #try :
        east_cor = self.east_output ()
        angle = self.get_rotaion_angle (east_cor)
        rotations =  1
        #self.dump_out(east_cor,rotations)
        # Orientation correction
        while abs (angle) > 2.5:
            self.image = imutils.rotate_bound (self.image, -angle)

            if rotations > 1 :
                contours = cv2.findContours (cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours [0] if len (contours) == 2 else contours [1]
                if len(contours) > 0 :
                    x, y, w, h = cv2.boundingRect (contours[0])
                    #print('cropped area reduced ')
                    self.image = self.image[y:y+h,x:x+w,:]

            east_cor = self.east_output ()
            angle = self.get_rotaion_angle (east_cor)
            rotations += 1
            #self.dump_out(east_cor,rotations)
        bbox1 = Box_cordinates (east_cor)
        upside_down = self.check_orientation (bbox1.gr_cordinates)
        if upside_down:
            self.image = imutils.rotate_bound (self.image, 180)
            east_cor = self.east_output ()
            #self.dump_out(east_cor,rotations)
        bbox2 = Box_cordinates (east_cor,image= self.image,conf_threshold=self.conf_threshold)
        text_dic,avrage_confidence = bbox2.get_text ()
        #added text text_postprocessing
        self.text = {'metadata': Text_to_json(text_dic).metadata , 'text':[text_dic] , 'avrage_confidence':avrage_confidence}
        print(self.text)

        #except :
        #    logging.debug('Text extaction failed for {0}'.format(self.image_path))





class Box_cordinates:
    def __init__(self, bbox, conf_threshold=50, image=None):
        self.bbox             = bbox
        self.image            = image
        #if type(self.image) != None :
            #self.image = cv2.medianBlur(self.image,3)
            #self.image = cv2.GaussianBlur(median,(5,5))
            #self.image = cv2.bilateralFilter(self.image.astype(np.int8),9,75,75)
        self.conf_threshold   = conf_threshold
        self.convert_to_df()
        self.group_by_spacing()
        self.group_corodinates()


    def convert_to_df(self):
        #dic = []
        #for i,box in enumerate(self.bbox):
            #dic.append({'x1': box[0] ,'y1': box[1] ,'x2': box[2] ,'y2': box[3] ,'x3': box[4] ,'y3': box[5] ,'x4': box[6] ,'y4': box[7]})
        df = pd.DataFrame(self.bbox,columns=['x1','y1','x2','y2','x3','y3','x4','y4'])
        df['height'] = df['y4'] - df['y1']
        df['width']  = df['x2'] - df['x1']
        df['ymid']   = (df['y4'] + df['y3']) * 0.5
        df['area']   = df['width'] * df['height']
        df = df.sort_values(by=['ymid'])
        df['group']  =  None
        df['line_change'] = 0
        self.df = df


    def group_by_spacing(self):
        group = 0
        check_ymid = self.df.iloc[0]['ymid']
        for index, row in self.df.iterrows():
            height = row['height']
            ymid   = row['ymid']
            if  abs(ymid - check_ymid) < (height *2.5)  :
                self.df['group'][index] = group
            else:
                group += 1
                self.df['group'][index] = group
            check_ymid = ymid


    def group_corodinates(self):

        group_cordinates = []
        for group in self.df['group'].unique():
            gr_df = self.df[self.df['group'] == group]
            x1 = gr_df['x1'].min()
            y1 = gr_df['y1'].min()
            x2 = gr_df['x3'].max()
            y2 = gr_df['y3'].max()
            group_cordinates.append([(x1,y1),(x2,y2)])

        self.gr_cordinates = group_cordinates


    def open_minus_image(self,image,kernel,smooth_size=3):
        #kernel = 4*avrage_height
        img        = image.copy()
        open_image = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        median = cv2.medianBlur(img - open_image,smooth_size)
        return median


    def sort_group(self,group,len_groups,sorted_group=[]):

        mean_semi_height = group['height'].mean() / 2.0
        check_ymid       = group.iloc[0]['ymid']
        same_line        = group[ abs(group['ymid'] - check_ymid) < mean_semi_height]
        next_lines       = group[ abs(group['ymid'] - check_ymid) >= mean_semi_height]
        x1 = same_line ['x1'].min ()
        y1 = same_line ['y1'].min ()
        x2 = same_line ['x3'].max ()
        y2 = same_line ['y3'].max ()
        line = {'x1' : x1,'y1':y1,'x2':x2,'y4':y2,'height':same_line['height'].mean(),'fragmented':False}

        sum_area = same_line['area'].sum()
        block_area = (x2 - x1 ) * (y2 -y1)
        #print(same_line)
        if ((sum_area / block_area) < 0.5) or (len(same_line) < 3) :
            fragments = []
            sort_lines       = same_line.sort_values(by=['x1'])
            for index, row in sort_lines.iterrows():
                fragments.append(row)
            line = {'fragmented':True , 'fragments':fragments ,'height':same_line['height'].mean()}


        sorted_group.append(line)

        if len(next_lines) > 0 :
            self.sort_group (next_lines, len_groups, sorted_group)

        return sorted_group

    def crop_im(self ,row ,margin=5):
        crop = self.image[row['y1']- margin : row['y4'] + margin , row['x1'] -margin : row['x2'] + margin]
        return crop


    def get_text(self):

        block_text ={}
        conf =[]
        ignore_text = [' ' ,'']
        mean_height = self.df['height'].mean()
        smooth_image      = cv2.bilateralFilter(self.image,9,75,75)
        for group_id in self.df['group'].unique():
            group             = self.df[self.df['group'] == group_id]
            avrage_height     = int(group['height'].mean())
            sorted_grp        = self.sort_group(group,len(group),[])
            #self.image #self.open_minus_image(self.image , np.ones((avrage_height *4 , avrage_height*4)))

            group_text = ''
            text_by_line =[]
            for text_crop in sorted_grp:
                line_text = ''
                if text_crop['height'] > mean_height * 0.5 :
                    if text_crop['fragmented'] :
                        for word in text_crop['fragments']:
                            cropped_portion = self.crop_im(word,margin=5)
                            text = pytesseract.image_to_data(cropped_portion,config='--psm 7', lang='eng',output_type=Output.DATAFRAME)
                            text = text[text['conf'] >self.conf_threshold]
                            if len(text) > 0 :

                                for index, row in text.iterrows():
                                    detected_text = row['text']
                                    conf.append(row['conf'])
                                    if type(detected_text) != str:
                                        detected_text = str(int(detected_text))
                                    line_text     = line_text + ' ' + detected_text

                        #text = text['text'].astype(str)
                        #line_text = ' '.join(text.values)
                        if len(line_text)>0:
                            text_by_line.append(line_text)
                        #group_text = group_text + ' ' + line_text

                    else :
                        cropped_portion = self.crop_im(text_crop,margin=5)
                        text = pytesseract.image_to_data(cropped_portion,config='--psm 7', lang='eng',output_type=Output.DATAFRAME)
                        text = text[text['conf'] >self.conf_threshold]
                        if len(text) > 0 :

                            for index, row in text.iterrows():
                                detected_text = row['text']
                                conf.append(row['conf'])
                                if type(detected_text) != str:
                                    detected_text = str(int(detected_text))
                                line_text     = line_text + ' ' + detected_text

                            #text = text['text'].astype(str)
                            #line_text = ' '.join(text.values)
                            if len(line_text) > 0:
                                text_by_line.append(line_text)
                            #group_text = group_text + ' ' + line_text
                                #print(row['text'] ,row['conf'])

            block_text[group_id] = {'text' : ' '.join(text_by_line) , 'text_by_line':text_by_line}
        avrage_conf = np.mean(conf)
        return block_text , avrage_conf




class Text_to_json:

    def __init__(self,dic):
        self.text_df = pd.DataFrame.from_dict(dic,orient='index')
        self.metadata = {}
        self.get_name_dob_gender()
        self.get_aadhar_no()
        self.get_address()





    def get_dob_by_keyword(self,keyword,name,DOB,gender):
        keyword_found = False
        for index,row in self.text_df.iterrows():
            if keyword in row['text'] :
                #print(row['text_by_line'])
                index = 0
                keyword_found = True
                for index ,text in enumerate(row['text_by_line']):
                    print(text)
                    if keyword in text:
                        print(text)
                        break
                try:        
                    name = row['text_by_line'][index-1]
                    if 'Husband' in name :
                        name = ' '.join(row['text_by_line'][:index-2])
                except :
                    logging.debug('Failed to  extact name')
                try:
                    DOB  = row['text_by_line'][index].split(keyword)[-1]
                except :
                    logging.debug('Failed to  extact DOB')
                try:
                    g_detect = ' '.join(row['text_by_line'][index + 1 :])
                    #print('gdette' ,g_detect)
                    if 'Mal' in g_detect:
                        gender = 'Male'
                    if 'Fem' in g_detect:
                        gender = 'Female'
                except :
                    logging.debug('Failed to  extact gender')

                break
        return name,DOB,gender,keyword_found
        
        

    def get_name_dob_gender(self):
        name = 'unable to detect'
        DOB  = 'unable to detect'
        gender = 'unable to detect'
        
        name,DOB,gender,keyword_found = self.get_dob_by_keyword('DOB',name,DOB,gender)
        print('DOB',keyword_found)
        if not keyword_found :
            name,DOB,gender,keyword_found = self.get_dob_by_keyword('Year of Birth',name,DOB,gender)
            print('Year ',keyword_found)
        self.metadata['name'] = name
        self.metadata['DOB']  = DOB
        self.metadata['gender'] = gender
        #return name ,DOB,gender


    def get_aadhar_no(self):

        addhaar_no ='unable to detect'
        addhar_list = []
        for text in self.text_df['text'].values:
            try :
                find_a_no = re.search(r'(\d\d\d\d \d\d\d\d \d\d\d\d)', text)
                #detected_text = find_a_no.group(1)
                #if len(detected_text) < 18:
                #addhar_list.append(find_a_no.group(1))
                addhaar_no = find_a_no.group(1)
                break
            except:
                logging.debug('Failed to  extact aadhaar_no')

        self.metadata['aadhaar_no'] = addhaar_no
        #return addhaar_no


    def get_address(self):
        Address = 'unable to detect'
        pin     = 'unable to detect'
        
        #Assuming largest text cluster to be of Address
        self.text_df['text_length'] = self.text_df['text'].str.len()
        address_index = self.text_df["text_length"].idxmax() 
        address_text = self.text_df['text'][address_index]
        
        if 'Address' in address_text:
            Address = address_text.split('Address')[-1]
        else :
            if 'To' in address_text :
                Address = ' '.join(address_text.split('To')[1:])
            else:
                Address = address_text
        try :
            pin = re.search(r'(\d\d\d\d\d\d)', address_text)
            pin = pin.group(1)

        except:
            logging.debug('Failed to  detect pin ')
        self.metadata['Address'] = Address
        self.metadata['pincode']= pin

