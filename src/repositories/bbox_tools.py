import cv2
import pandas as pd
import pytesseract
from pytesseract import Output
import numpy as np


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
            if  abs(ymid - check_ymid) < (height *1.75)  :
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
                                    conf.append[row['conf']]
                                    if type(detected_text) != str:
                                        detected_text = str(int(detected_text))
                                    line_text     = line_text + ' ' + detected_text
                              
                        #text = text['text'].astype(str)
                        #line_text = ' '.join(text.values)
                            
                        text_by_line.append(line_text)    
                        #group_text = group_text + ' ' + line_text
                            
                    else :
                        cropped_portion = self.crop_im(text_crop,margin=5)
                        text = pytesseract.image_to_data(cropped_portion,config='--psm 7', lang='eng',output_type=Output.DATAFRAME)
                        text = text[text['conf'] >self.conf_threshold]
                        if len(text) > 0 :
                            
                            for index, row in text.iterrows():
                                detected_text = row['text']
                                conf.append[row['conf']]
                                if type(detected_text) != str:
                                    detected_text = str(int(detected_text))
                                line_text     = line_text + ' ' + detected_text
                            
                            #text = text['text'].astype(str)
                            #line_text = ' '.join(text.values)
                            text_by_line.append(line_text)    
                            #group_text = group_text + ' ' + line_text
                                #print(row['text'] ,row['conf'])

            block_text[group_id] = {'text' : ' '.join(text_by_line) , 'text_by_line':text_by_line}
        block_text['avrage_confidence'] = mean(conf)
        return block_text




    
