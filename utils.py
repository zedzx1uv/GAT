import torch 
import re 
import os 
import numpy as np 
import random 
import csv 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

def get_sst_data(type):
    '''SST-2 GLUE version
    '''
    print('Getting SST-2 Data...')
    text = [] 
    label = [] 
    path = './data/SST-2/' + type + '.tsv'
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin.readlines()[1:]:
            line = line.strip().split('\t')
            text.append(line[0])
            label.append(1. if line[1]=='1' else 0.)

    print('Done...')
    return text, label 




def get_imdb_data(type):
    """
    type: 'train' or 'test'
    """

    # [0,1] means positive，[1,0] means negative
    all_labels = []
    for _ in range(12500):
        all_labels.append(1.)
    for _ in range(12500):
        all_labels.append(0.)

    all_texts = []
    file_list = []
    path = r'./data/aclImdb/'
    pos_path = path + type + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + type + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts, all_labels

def get_imdb_unsup_data():

    all_texts = [] 
    file_list = [] 
    path = r'./data/aclImdb/'
    unsup_path = path + 'train/unsup/'
    for file in os.listdir(unsup_path):
        file_list.append(unsup_path + file) 
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts

def get_agnews_data(type):
    """
    type: 'train' or 'test'
    """
    print('Getting Agnews Data...')
    texts = [] 
    labels = [] 
    path = './data/agnews/' + type + '.csv'
    with open(path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for id_, row in enumerate(csv_reader):
            label, title, description = row
            # Original labels are [1, 2, 3, 4] ->
            #                   ['World', 'Sports', 'Business', 'Sci/Tech']
            # Re-map to [0, 1, 2, 3].
            label = int(label) - 1
            text = " ".join((title, description))
            labels.append(label)
            texts.append(text)
    return texts, labels

# def split_imdb_files():
#     print('Processing IMDB dataset')
#     train_texts, train_labels = read_imdb_files('train')
#     test_texts, test_labels = read_imdb_files('test')
#     return train_texts, train_labels, test_texts, test_labels

def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc = sum(int(t) for t in pred_flat==labels_flat) / len(pred_flat)
    # return accuracy_score(labels_flat, pred_flat)
    return acc 

def encode_fn(tokenizer, text_list, dataset='sst'):
    length = {'sst':40, 'agnews':40, 'imdb':200}
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,
                        truncation=True,                       
                        add_special_tokens = True,  # special tokens， CLS SEP
                        max_length = length[dataset],           # 
                        # pad_to_max_length = True,   #   
                        padding = 'max_length',
                        return_tensors = 'pt'       # 
                   )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

def read_atkre(path):
    oa, ra = [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if "Original accuracy:" in line:
                oa.append(float(line[-8:-3].replace('%','')))
            if 'Accuracy under attack:' in line:
                ra.append(float(line[-8:-3].replace('%','')))
    return oa, ra 
            


if __name__ == '__main__':
    # unsup_data = get_imdb_unsup_data()
    # print(len(unsup_data))
    # oa, ra = read_atkre('./sst_model/bert-base-uncased-advk-10/atkre-bae.txt')
    texts, labels = get_agnews_data("train")
    l = 0
    for text in texts:
        l += len(text.split(' '))
    print(l/len(texts))
