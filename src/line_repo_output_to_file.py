from repositories import OCRlineRepositories
import json

path_to_pdf = '/opt/share/nginx/upload/4.pdf'
#path_to_pdf  = '/home/dddhiraj/Documents/Tarento/data/rajyasabha/v250_hindi.pdf'
#pdf_path_mal = '/home/dddhiraj/Documents/Tarento/data/SC_JUDGMENTS_2010_2020/SC_JUDGMENTS_2010_2020/2019/12609_2009_14_101_16975_Judgement_19-Sep-2019_MAL.pdf'
#pdf_path_tam = '/home/dddhiraj/Documents/Tarento/data/SC_JUDGMENTS_2010_2020/SC_JUDGMENTS_2010_2020/2019/2314_2008_13_1501_13210_Judgement_15-Mar-2019_TAM.pdf'
#pdf_path_hin = '/home/dddhiraj/Documents/Tarento/data/SC_JUDGMENTS_2010_2020/SC_JUDGMENTS_2010_2020/2019/15941_2016_11_1501_13004_Judgement_06-Mar-2019_HIN.pdf'
#pdf_path_tel = '/home/dddhiraj/Documents/Tarento/data/SC_JUDGMENTS_2010_2020/SC_JUDGMENTS_2010_2020/2019/5775_2007_14_1501_17049_Judgement_25-Sep-2019_TEL.pdf'
output_dir   = '/home/ubuntu' #'/home/dddhiraj/Documents/Tarento/data/rajyasabha/ocr_line_output' 


#path_to_pdf= pdf_path_tel

line_parser = OCRlineRepositories(path_to_pdf)
response = line_parser.response


file_name = path_to_pdf.split('/')[-1].split('.')[0]
output_file = output_dir + '/' + file_name + '.json'

with open(output_file, "w", encoding='utf8') as write_file:
    json.dump(response, write_file,ensure_ascii=False )


