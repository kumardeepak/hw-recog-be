from repositories import OCRlineRepositories
import json


path_to_pdf  = '/home/dddhiraj/Documents/Tarento/data/rajyasabha/v250_hindi.pdf'
output_dir   = '/home/dddhiraj/Documents/Tarento/data/rajyasabha/ocr_line_output' 

line_parser = OCRlineRepositories(path_to_pdf,language='hin')
response = line_parser.response


file_name = path_to_pdf.split('/')[-1].split('.')[0]
output_file = output_dir + '/' + file_name + '.json'

with open(output_file, "w", encoding='utf8') as write_file:
    json.dump(response, write_file,ensure_ascii=False )


