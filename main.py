import torch
from transformers import AutoModelForSequenceClassification,BertTokenizer
from modeling_cpt import CPTForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FENZHEN_MODEL_NAME = 'WENGSYX/Dagnosis_Chinese_BERT'
CMDD_MODEL_NAME = 'WENGSYX/Dagnosis_Chinese_CPT'
BL_MODEL_NAME = 'WENGSYX/Medical_Report_Chinese_CPT'
fenzhen_model = AutoModelForSequenceClassification.from_pretrained(FENZHEN_MODEL_NAME, num_labels=6).to(device)  # 模型
fenzhen_tokenizer = BertTokenizer.from_pretrained(FENZHEN_MODEL_NAME)


cmdd_model = CPTForConditionalGeneration.from_pretrained(CMDD_MODEL_NAME)
cmdd_model = cmdd_model.to(device)
berttokenizer = BertTokenizer.from_pretrained(CMDD_MODEL_NAME)

bl_model = CPTForConditionalGeneration.from_pretrained(BL_MODEL_NAME)
bl_model = bl_model.to(device)


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


def Summary(parser):

    if parser.mode == 'interactive':
        while True:
            message = input('请输入对话历史，让我来帮您记录病例信息：(退出请输入\033[0;35m退出\033[0m)')
            if message == '退出':
                return ''
            text = berttokenizer(message,padding='max_length', truncation=True, max_length=512,return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            token_type_ids = text['token_type_ids'].to(device)

            out = berttokenizer.batch_decode(
                bl_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
            out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：',
                                                                                                              '')
            print('医生: \033[0;34m{}\033[0m'.format(out))
    elif parser.mode == 'batch':
        with open(parser.file_name,'r',encoding='utf-8') as f:
            data = f.readlines()
        result = []
        for message in data:
            text = berttokenizer(message,padding='max_length', truncation=True, max_length=512,return_tensors='pt')
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            token_type_ids = text['token_type_ids'].to(device)

            out = berttokenizer.batch_decode(
                bl_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
            out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：',
                                                                                                              '')
            result.append(out)
        with open(parser.result_file_name,'w',encoding='utf-8') as f:
            for i in result:
                f.write(i+'\n')

    else:
        assert parser.message != None,print('请传入文本')
        text = berttokenizer(parser.message, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = text['input_ids'].to(device)
        attention_mask = text['attention_mask'].to(device)
        token_type_ids = text['token_type_ids'].to(device)

        out = berttokenizer.batch_decode(
            bl_model.generate(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, max_length=256))[0]
        out = out.replace('[SEP]', '').replace('[CLS]', '').replace(' ', '').replace('指导意见：', '').replace('病情分析：','')
        return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='Dagnosis',type=str)
    parser.add_argument("--mode", default='interactive', type=str)
    parser.add_argument("--file_name",default=None,type=str)
    parser.add_argument("--result_file_name", default='result.csv', type=str)
    parser.add_argument("--message", default=None, type=str)
    parser = parser.parse_args()

    exec('RSA = {}(parser)'.format(parser.type))
    print(RSA)