import torch
# torch.cuda.set_device(1)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig 
from torch.utils.data import TensorDataset, DataLoader, random_split 
import random 

dataset_name = 'agnews' # or agnews
model_name = 'bert' # or deberta
if model_name == 'bert':
    model_type = "bert-base-uncased"
    base_model = "bert"
else:
    model_type = "microsoft/deberta-v3-base" 
    base_model = 'deberta'

if dataset_name == 'sst':
    num_labels = 2 
else:
    num_labels = 4 

tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)
config = AutoConfig.from_pretrained(model_type, num_labels=num_labels, output_attentions=False, output_hidden_states=False)
model = AutoModelForSequenceClassification.from_pretrained(model_type, config=config)

from utils import get_sst_data, get_imdb_data, get_imdb_unsup_data, get_agnews_data, set_seed, flat_accuracy, encode_fn

# sst 
if dataset_name == 'sst':
    train_texts, train_labels = get_sst_data('train')
    all_train_ids = encode_fn(tokenizer, train_texts, dataset='sst')
    labels = torch.tensor(train_labels)
    test_texts, test_labels = get_sst_data('dev')
    all_test_ids = encode_fn(tokenizer, test_texts, dataset='sst')
    test_labels = torch.tensor(test_labels)
else:
    # agnews 
    train_texts, train_labels = get_agnews_data('train')
    all_train_ids = encode_fn(tokenizer, train_texts, dataset='agnews')
    labels = torch.tensor(train_labels)
    test_texts, test_labels = get_agnews_data('test')
    all_test_ids = encode_fn(tokenizer, test_texts, dataset='agnews')
    test_labels = torch.tensor(test_labels)


set_seed(2021)

device = torch.device('cuda:0')

def build_inputs(batch):
    '''
    Sent all model inputs to the appropriate device (GPU on CPU)
    rreturn:
     The inputs are in a dictionary format
    '''
    input_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    batch = (batch[0].to(device), (batch[0]>0).to(device), None, batch[1].long().to(device))
    # batch = tuple(t.to(device) for t in batch)
    inputs = {key: value for key, value in zip(input_keys, batch)}
    return inputs

from adversarial_train import FreeLB
from tqdm import tqdm 

def loss_acc_of_epochs(path, k=10, testdata=True):
    loss_list, acc_list = [], []
    for i in tqdm(range(10)):
        # model_path = './sst_model/bert-base-uncased-advk-10/'+str(i)+'.pt'
        model_path = path + str(i) + '.pt' 
        # /mnt/rao/home/zb/Flooding-X/saved_models_tavat_flooding0.2/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs100_test
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert/infobert_glue-sst2_mag0.02_adv-steps10_adv-lr0.01_epochs100/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert_flooding0.2/infobert_glue-sst2_mag0.02_adv-steps10_adv-lr0.01_epochs100/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat_flooding0.2/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs100_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = path+str(i)+'/pytorch_model.bin' 
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert_cali/infobert_glue-sst2_mag0.02_adv-steps10_adv-lr0.01_epochs100/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat_cali/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs100_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat/TAVAT_microsoft-deberta-v3-base_ag_news_None_adv10_epochs10_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert/infobert_microsoft-deberta-v3-base_ag_news_mag0.02_adv-steps10_adv-lr0.01_epochs10/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat_flooding0.1/TAVAT_microsoft-deberta-v3-base_ag_news_None_adv10_epochs10_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert_flooding0.1/infobert_microsoft-deberta-v3-base_ag_news_mag0.02_adv-steps10_adv-lr0.01_epochs10/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat/TAVAT_microsoft-deberta-v3-base_glue_sst2_adv10_epochs10_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat/TAVAT_bert-base-uncased_ag_news_None_adv10_epochs10_test/epoch'+str(i)+'/pytorch_model.bin'
        # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert/infobert_bert-base-uncased_ag_news_mag0.02_adv-steps10_adv-lr0.01_epochs10/epoch'+str(i)+'/pytorch_model.bin'
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        tmp_acc = 0
        tmp_loss = 0
        all = 768
        _iter = 6 
        bs = int(all/_iter)
        tt = random.randint(0,400)
        for ii in range(_iter):
            model.zero_grad()
            if testdata:
                batch = (all_test_ids[ii*bs:(ii+1)*bs], test_labels[ii*bs:(ii+1)*bs])
            else:
                
                batch = (all_train_ids[(ii+tt)*bs:(ii+tt+1)*bs], labels[(ii+tt)*bs:(ii+tt+1)*bs])
                # batch = next(train_dataloader)
            # batch = (all_train_ids[0:128], labels[0:128])
            inputs = build_inputs(batch)
            adv_trainer = FreeLB(adv_K=k,adv_lr=1e-2,adv_init_mag=2e-2,adv_max_norm=1,adv_norm_type='l2',base_model=base_model,f_b=0,flooding=False)
            loss, logits, loss_item = adv_trainer.attack(model, inputs, gradient_accumulation_steps=1)

            logits_de = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            acc = flat_accuracy(logits_de, label_ids)
            tmp_acc += acc / _iter
            tmp_loss += loss_item / _iter
        loss_list.append(tmp_loss)
        acc_list.append(tmp_acc)
        print('loss:{}'.format(tmp_loss))
        print('acc:{}'.format(tmp_acc))
        print('=============================================')
    return loss_list, acc_list



# freelb_10_new_test_loss_list, freelb_10_new_test_acc_list = \
#     loss_acc_of_epochs(path='./sst_model/bert-base-uncased-advk-10-new/', k=10, testdata=True)
# freelb_10_cali_new_test_loss_list, freelb_10_cali_new_test_acc_list = \
#     loss_acc_of_epochs(path='./sst_model/bert-base-uncased-advk-10-cali-new/', k=10, testdata=True)
# tavat10_train_loss_list, tavat10_train_acc_list = \
#         loss_acc_of_epochs(path='/mnt/rao/home/zb/Flooding-X/saved_models_tavat/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs100_test/epoch', k=10, testdata=False)

# infobert10_train_loss_list, infobert10_train_acc_list = \
#         loss_acc_of_epochs(path='/mnt/rao/home/zb/Flooding-X/saved_models_infobert/infobert_glue-sst2_mag0.02_adv-steps10_adv-lr0.01_epochs100/epoch', k=10, testdata=False)

# freelb10_test_loss_list, freelb10_test_acc_list = \
#     loss_acc_of_epochs(path='./sst_model/bert-base-uncased-advk-10/', k=10, testdata=True)

# freelb10_flooding_test_loss_list, freelb10_flooding_test_acc_list = \
#     loss_acc_of_epochs(path='./sst_model/bert-base-uncased-advk-10-flooding0.2/', k=10, testdata=True)

# infobert10_test_loss_list, infobert10_test_acc_list = \
#       loss_acc_of_epochs(path='', k=10, testdata=True)

# infobert10_flooding_test_loss_list, infobert10_flooding_test_acc_list = \
#       loss_acc_of_epochs(path='', k=10, testdata=True)

# tavat10_flooding_test_loss_list, tavat10_flooding_test_acc_list = \
#       loss_acc_of_epochs(path='', k=10, testdata=True)

# freelb10_cali_test_loss_list, freelb10_cali_test_acc_list = \
#         loss_acc_of_epochs('./sst_model/bert-base-uncased-advk-10-cali/', k=10, testdata=True)

# tavat10_cali_test_loss_list, tavat10_cali_test_acc_list = \
#         loss_acc_of_epochs('', k=10, testdata=True)

# infobert10_cali_train_loss_list, infobert10_cali_train_acc_list = \
#         loss_acc_of_epochs('', k=10, testdata=False)
# freelb10_cali_train_loss_list, freelb10_cali_train_acc_list = \
#         loss_acc_of_epochs('/mnt/rao/home/zb/GAT_code/sst_model/bert-base-uncased-advk-10-cali/', k=10, testdata=False)
# tavat10_cali_train_loss_list, tavat10_cali_train_acc_list = \
#         loss_acc_of_epochs('', k=10, testdata=False)

# freelb10_test_loss_list, freelb10_test_acc_list = \
#     loss_acc_of_epochs(path='./agnews_model/deberta-v3-base-advk-10/', k=10, testdata=True)

# freelb10_train_loss_list, freelb10_train_acc_list = \
#     loss_acc_of_epochs(path='./agnews_model/deberta-v3-base-advk-10/', k=10, testdata=False)

# tavat10_test_loss_list, tavat10_test_acc_list = \
#     loss_acc_of_epochs(path='', k=10, testdata=True)

# tavat10_train_loss_list, tavat10_train_acc_list = \
#     loss_acc_of_epochs(path='', k=10, testdata=False)

# infobert10_test_loss_list, infobert10_test_acc_list = \
#     loss_acc_of_epochs(path='', k=10, testdata=True)

# infobert10_train_loss_list, infobert10_train_acc_list = \
#     loss_acc_of_epochs(path='', k=10, testdata=False)

# weight decay
# freelb10_test_loss_list, freelb10_test_acc_list = \
#     loss_acc_of_epochs(path='./sst_model/dropout/bert-base-uncased-advk-10-dropout0.9/', k=10, testdata=True)
# f_b = 0 
# freelb_10_flooding_test_loss_list, freelb_10_flooding_test_acc_list = \
#       loss_acc_of_epochs(path='./sst_model/bert-base-uncased-advk-10/',k=10, testdata=True)

# sst deberta
freelb_10_test_loss_list, freelb_10_test_acc_list = \
        loss_acc_of_epochs(path='./agnews_model/bert-base-uncased-advk-10/',k=10,testdata=True)
freelb_10_train_loss_list, freelb_10_train_acc_list = \
        loss_acc_of_epochs(path='./agnews_model/bert-base-uncased-advk-10/',k=10,testdata=False)
# tavat_10_test_loss_list, tavat_10_test_acc_list = \
#         loss_acc_of_epochs(path='',k=10,testdata=True)
# tavat_10_train_loss_list, tavat_10_train_acc_list = \
#         loss_acc_of_epochs(path='',k=10,testdata=False)
# infobert_10_test_loss_list, infobert_10_test_acc_list = \
#         loss_acc_of_epochs(path='',k=10,testdata=True)
# infobert_10_train_loss_list, infobert_10_train_acc_list = \
#         loss_acc_of_epochs(path='',k=10,testdata=False)
import numpy as np 

np.save('./fig/advk10/bert_agnews/freelb_10_test_loss_list.npy',freelb_10_test_loss_list)
np.save('./fig/advk10/bert_agnews/freelb_10_test_acc_list.npy',freelb_10_test_acc_list)
np.save('./fig/advk10/bert_agnews/freelb_10_train_loss_list.npy',freelb_10_train_loss_list)
np.save('./fig/advk10/bert_agnews/freelb_10_train_acc_list.npy',freelb_10_train_acc_list)
# np.save('./fig/advk10/flooding/freelb_10_flooding_'+str(f_b)+'_test_loss_list.npy', freelb_10_flooding_test_loss_list)
# np.save('./fig/advk10/flooding/freelb_10_flooding_'+str(f_b)+'_test_acc_list.npy', freelb_10_flooding_test_acc_list)
# np.save('./fig/advk10/dropout/freelb10_test_loss_list_0.9.npy',freelb10_test_loss_list)
# np.save('./fig/advk10/dropout/freelb10_test_acc_list_0.9.npy',freelb10_test_acc_list)

# np.save('./fig/advk10/deberta/infobert_10_test_loss_list.npy', infobert10_test_loss_list)
# np.save('./fig/advk10/deberta/infobert_10_test_acc_list.npy', infobert10_test_acc_list)
# np.save('./fig/advk10/deberta/infobert_10_train_loss_list.npy', infobert10_train_loss_list)
# np.save('./fig/advk10/deberta/infobert_10_train_acc_list.npy', infobert10_train_acc_list)

# re-attack
# np.save('./fig/advk10/freelb_10_new_test_loss_list.npy', freelb_10_new_test_loss_list)
# np.save('./fig/advk10/freelb_10_new_test_acc_list.npy', freelb_10_new_test_acc_list)

# np.save('./fig/advk10/freelb_10_cali_new_test_loss_list.npy', freelb_10_cali_new_test_loss_list)
# np.save('./fig/advk10/freelb_10_cali_new_test_acc_list.npy', freelb_10_cali_new_test_acc_list)
# np.save('./fig/advk10/freelb10_cali_test_loss_list.npy', freelb10_cali_test_loss_list)
# np.save('./fig/advk10/freelb10_cali_test_acc_list.npy', freelb10_cali_test_acc_list)

# np.save('./fig/advk10/tavat10_train_loss_list.npy', tavat10_train_loss_list)
# np.save('./fig/advk10/tavat10_train_acc_list.npy', tavat10_train_acc_list)

# np.save('./fig/advk10/infobert10_train_loss_list.npy', infobert10_train_loss_list)
# np.save('./fig/advk10/infobert10_train_acc_list.npy', infobert10_train_acc_list)

# np.save('./fig/advk10/freelb10_test_loss_list.npy', freelb10_test_loss_list)
# np.save('./fig/advk10/freelb10_test_acc_list.npy', freelb10_test_acc_list)

# np.save('./fig/advk10/freelb10_flooding_test_loss_list.npy', freelb10_flooding_test_loss_list)
# np.save('./fig/advk10/freelb10_flooding_test_acc_list.npy', freelb10_flooding_test_acc_list)

# np.save('./fig/advk10/infobert10_flooding_test_loss_list.npy', infobert10_flooding_test_loss_list)
# np.save('./fig/advk10/infobert10_flooding_test_acc_list.npy', infobert10_flooding_test_acc_list)

# ================================================================================

def loss_acc_of_strength(path, epoch=9, testdata=True):
    loss_list, acc_list = [], []
    
    # model_path = path+str(epoch)+'.pt'
    model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_tavat/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs100_test/epoch'+str(epoch)+'/pytorch_model.bin'
    # model_path = '/mnt/rao/home/zb/Flooding-X/saved_models_infobert/infobert_bert-base-uncased_glue-sst2_mag0.02_adv-steps10_adv-lr0.01_epochs100/epoch'+str(epoch)+'/pytorch_model.bin'
    model.load_state_dict(torch.load(model_path))
    model.cuda()  
    model.eval()
    
    for k in tqdm(range(30)):
        tmp_acc = 0
        tmp_loss = 0
        # batch = (all_train_ids[0:128], labels[0:128])
        all = 768
        _iter = 6 
        bs = int(all/_iter)
        tt = random.randint(0,400)
        for ii in range(_iter):
            if testdata:
                batch = (all_test_ids[ii*bs:(ii+1)*bs], test_labels[ii*bs:(ii+1)*bs])
            else:
                # batch = (all_train_ids[ii*bs:(ii+1)*bs], labels[ii*bs:(ii+1)*bs])
                batch = (all_train_ids[(ii+tt)*bs:(ii+tt+1)*bs], labels[(ii+tt)*bs:(ii+tt+1)*bs])
            inputs = build_inputs(batch)
            adv_trainer = FreeLB(adv_K=k+1,adv_lr=1e-2,adv_init_mag=2e-2,adv_max_norm=1,adv_norm_type='l2',base_model='bert',f_b=0,flooding=False)
            loss, logits, loss_item = adv_trainer.attack(model, inputs, gradient_accumulation_steps=1)
            logits_de = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            acc = flat_accuracy(logits_de, label_ids)
            tmp_acc += acc / _iter
            tmp_loss += loss_item / _iter
        
        loss_list.append(tmp_loss)
        acc_list.append(tmp_acc)
        print('Loss:{}\nAcc:{}'.format(tmp_loss, tmp_acc))

    return loss_list, acc_list

# freelb10_epoch49_train_loss_list, freelb10_epoch49_train_acc_list = \
#         loss_acc_of_strength('./sst_model/bert-base-uncased-advk-10/', epoch=49, testdata=False)

# tavat10_epoch49_train_loss_list, tavat10_epoch49_train_acc_list = \
#         loss_acc_of_strength('', epoch=49, testdata=False)

# infobert10_epoch49_train_loss_list, infobert10_epoch49_train_acc_list = \
#         loss_acc_of_strength('', epoch=49, testdata=False)
        
# import numpy as np

# np.save('./fig/advk10/tavat10_epoch49_train_loss_list.npy', tavat10_epoch49_train_loss_list)
# np.save('./fig/advk10/tavat10_epoch49_train_acc_list.npy', tavat10_epoch49_train_acc_list)