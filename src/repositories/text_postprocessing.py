import pandas as pd
import re




class Text_to_json:
    
    def __init__(self,dic):
        self.text_df = pd.DataFrame.from_dict(imgs[3],orient='index')
        self.metadata = {}
        self.get_name_dob_gender()
        self.get_aadhar_no()
        self.get_address()


    def get_name_dob_gender(self):
        name = 'unable to detect'
        DOb  = 'unable to detect'
        gender = 'unable to detect'
        for index,row in self.text_df.iterrows():
            if 'DOB' in row['text'] :
                print(row['text_by_line'])
                index = 0
                for index ,text in enumerate(row['text_by_line']):
                    if 'DOB' in text:
                        break
                try:
                    name = row['text_by_line'][index-1]
                except :
                    pass
                try:
                    DOB  = row['text_by_line'][index].split('DOB')[-1]
                except :
                    pass
                try:
                    g_detect = row['text_by_line'][index + 1]
                    if 'Male' in g_detect:
                        gender = 'Male'
                    if 'Female' in g_detet:
                        gender = 'Female'
                except :
                    pass
                
                break
                
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
                pass
            
        self.metadata['aadhaar_no'] = addhaar_no
        #return addhaar_no
                
                
    def get_address(self):
        Address = 'unable to detect'
        pin  = 'unable to detect'
        for index,row in self.text_df.iterrows():
            if 'Address' in row['text'] :
                Address = row['text'].split('Address')[-1]
                try :
                    pin = re.search(r'(\d\d\d\d\d\d)', row['text'])
                    pin = pin.group(1)
                    
                except:
                    pass
        self.metadata['Address'] = Address
        self.metadata['pincode']= pin
        #return Address , pin
