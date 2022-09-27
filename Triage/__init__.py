import torch
from transformers import AutoModelForSequenceClassification,BertTokenizer
from .modeling_cpt import CPTForConditionalGeneration
from Triage import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FENZHEN_MODEL_NAME = 'WENGSYX/Dagnosis_Chinese_BERT'
CMDD_MODEL_NAME = 'WENGSYX/Dagnosis_Chinese_CPT'
BL_MODEL_NAME = 'WENGSYX/Dagnosis_Chinese_CPT'
fenzhen_model = AutoModelForSequenceClassification.from_pretrained(FENZHEN_MODEL_NAME, num_labels=6).to(device)  # 模型
fenzhen_tokenizer = BertTokenizer.from_pretrained(FENZHEN_MODEL_NAME)

def Triage(parser):

    if parser.mode == 'interactive':
        print('我们将为您分配科室')
        while True:
            message = input('请输入您想询问的症状(退出请输入\033[0;35m退出\033[0m)')
            if message == '退出':
                return ''
            text = fenzhen_tokenizer(message, max_length=512, return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            fenzhen_prodict = fenzhen_model(input_ids)[0]
            ks = {0:'男科',1:'内科',2:'妇科',3:'肿瘤科',4:'儿科',5:'外科'}[int(fenzhen_prodict.argmax())]
            print('您可能需要前往: \033[0;32m{}\033[0m'.format(ks))
    elif parser.mode == 'batch':
        with open(parser.file_name,'r',encoding='utf-8') as f:
            data = f.readlines()
        result = []
        for message in data:
            text = fenzhen_tokenizer(message.replace('\n',''), max_length=512, return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            fenzhen_prodict = fenzhen_model(input_ids)[0]
            ks = {0:'男科',1:'内科',2:'妇科',3:'肿瘤科',4:'儿科',5:'外科'}[int(fenzhen_prodict.argmax())]
            result.append(ks)
        with open(parser.result_file_name,'w',encoding='utf-8') as f:
            for i in result:
                f.write(i+'\n')

    else:
        assert parser.message != None,print('请传入文本')
        text = fenzhen_tokenizer(parser.message, max_length=512, return_tensors='pt')
        input_ids = text['input_ids'].to(device)
        fenzhen_prodict = fenzhen_model(input_ids)[0]
        ks = {0: '男科', 1: '内科', 2: '妇科', 3: '肿瘤科', 4: '儿科', 5: '外科'}[int(fenzhen_prodict.argmax())]
        return ks
