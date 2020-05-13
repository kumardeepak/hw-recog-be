from repositories import Aadhaar_exract
#import json

image_file =  '/home/dddhiraj/Documents/Tarento/Anuwad/EAST/input_images/0.png'
#/home/ubuntu/apps/aadhar/backup/inputs
#path_to_pdf= pdf_path_tel
adhar_text = Aadhaar_exract(image_file)

print(adhar_text.text)




'''
line_parser = OCRlineRepositories(path_to_pdf)
response = line_parser.response


file_name = path_to_pdf.split('/')[-1].split('.')[0]
output_file = output_dir + '/' + file_name + '.json'

with open(output_file, "w", encoding='utf8') as write_file:
    json.dump(response, write_file,ensure_ascii=False )

'''
