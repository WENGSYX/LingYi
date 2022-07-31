import torch
from transformers import AutoModelForSequenceClassification,BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
from Triage import *
from Summary import *
from Dagnosis import *



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
