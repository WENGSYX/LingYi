import torch
from transformers import AutoModelForSequenceClassification,BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
from Triage import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FENZHEN_MODEL_NAME = 'fenzhen_BERT'
CMDD_MODEL_NAME = 'CPT'
BL_MODEL_NAME = 'BLCPT'

cmdd_model = CPTForConditionalGeneration.from_pretrained(CMDD_MODEL_NAME)
cmdd_model = cmdd_model.to(device)
berttokenizer = BertTokenizer.from_pretrained(CMDD_MODEL_NAME)


def Dagnosis(parser):

    if parser.mode == 'interactive':
        while True:
            message = input('医生：\033[0;34m您好，有什么我能帮您？\033[0m(退出请输入\033[0;35m退出\033[0m)')
            if message == '退出':
                return ''
            text = berttokenizer('患者:'+message,padding='max_length', truncation=True, max_length=512,return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            token_type_ids = text['token_type_ids'].to(device)

            out = berttokenizer.batch_decode(
                cmdd_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
            out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：',
                                                                                                              '')
            print('医生: \033[0;34m{}\033[0m'.format(out))
    elif parser.mode == 'batch':
        with open(parser.file_name,'r',encoding='utf-8') as f:
            data = f.readlines()
        result = []
        for message in data:
            text = berttokenizer('患者:'+message,padding='max_length', truncation=True, max_length=512,return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            token_type_ids = text['token_type_ids'].to(device)

            out = berttokenizer.batch_decode(
                cmdd_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
            out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：',
                                                                                                              '')
            result.append(out)
        with open(parser.result_file_name,'w',encoding='utf-8') as f:
            for i in result:
                f.write(i+'\n')

    else:
        assert parser.message != None,print('请传入文本')
        text = berttokenizer('患者:' + parser.message, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = text['input_ids'].to(device)
        attention_mask = text['attention_mask'].to(device)
        token_type_ids = text['token_type_ids'].to(device)

        out = berttokenizer.batch_decode(
            cmdd_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
        out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：',
                                                                                                          '')
        return out