import torch 
import random 
import numpy as np 
import time 
import os 
import csv 
import argparse 
import sys 
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader, random_split 
from transformers import BertTokenizer, BertConfig, BertModel 
from transformers import BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig 
from transformers import get_linear_schedule_with_warmup 
from adversarial_train import FreeLB, PGD, FGM, KL, CalibrateFreeLB 
from ChildTuningOptimizer import ChildTuningAdamW 
from torch.optim import SGD, AdamW
from utils import get_sst_data, get_imdb_data, get_imdb_unsup_data, get_agnews_data, set_seed, flat_accuracy, encode_fn   
from modeling import BertForCL, Similarity 

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

def build_unsup_inputs(batch):
    input_keys = ['input_ids', 'attention_mask', 'token_type_ids',]
    batch = (batch[0].to(device), (batch[0]>0).to(device), None,)

    inputs = {key: value for key, value in zip(input_keys, batch)}
    return inputs



def train(args):
    # _cls = '[CLS]'
    # _sep = '[SEP]'
    # _pad = '[PAD]'
    fada = args.fada 
    fada_path = args.fada_path 
    ada = args.ada 
    ada_path = args.ada_path 
    freelb = args.freelb 
    kl = args.kl 
    cl = args.cl 
    fgm = args.fgm 
    tokenizer, config, model = None, None, None 
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, do_lower_case=True)
    config = AutoConfig.from_pretrained(args.model_type, num_labels=args.num_labels, output_attentions=False, output_hidden_states=False,\
        hidden_dropout_prob=args.dropout_r, attention_probs_dropout_prob=args.dropout_r)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type, config=config)
    # print(model)
    # model.load_state_dict(torch.load('./imdb_model/bert-base-uncased/9.pt'))
    model.cuda()
    # print(model)

    if args.sst:
        train_texts, train_labels = get_sst_data('train')
    elif args.agnews:
        train_texts, train_labels = get_agnews_data('train')
        # print('Encoding Data...')
        # all_train_ids = encode_fn(tokenizer, train_texts, sst=args.sst)
        # labels = torch.tensor(train_labels)
    # else:
        # train_texts, train_labels = get_imdb_data('train')
        # all_train_ids = torch.load('./data/aclImdb/train_ids.pt')
        # labels = torch.load('./data/aclImdb/labels_ids.pt')
    # # GLUE version no test label
    # test_texts, test_labels = get_sst_data('dev')
    # if args.aug_only:
    #     train_texts, train_labels = [], [] 
    if args.fada:
        with open(fada_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            reader = list(reader)[1:]
            for line in reader:
                train_texts.append(str(line[7]))
                train_labels.append(int(float(line[5])))
    elif args.ada:
        with open(ada_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            reader = list(reader)[1:]
            for line in reader:
                if line[8] == 'Successful':
                    train_texts.append(str(line[1]))
                    train_labels.append(int(float(line[5])))
    print('Encoding Data...')
    if args.sst:
        all_train_ids = encode_fn(tokenizer, train_texts, dataset='sst')
        labels = torch.tensor(train_labels)
    elif args.agnews:
        all_train_ids = encode_fn(tokenizer, train_texts, dataset='agnews')
        labels = torch.tensor(train_labels)
    # torch.save(all_train_ids, './data/aclImdb/train_ids.pt')
    # torch.save(labels, './data/aclImdb/labels_ids.pt')
    print('Done...')
    # print(len(all_train_ids))
    # print(len(labels))
    epochs = args.epochs
    batch_size = args.batch_size

    # Split data into train and validation
    dataset = TensorDataset(all_train_ids, labels)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    
    # if args.unsup:
    #     print('Using Unsup Data')
    #     all_unsup_ids = torch.load('data/aclImdb/unsup_ids.pt')[:train_size]
    #     unsup_labels = torch.load('data/aclImdb/unsup_labels.pt')[:train_size]
    #     unsup_dataset = TensorDataset(all_unsup_ids, unsup_labels)
    #     unsup_dataloader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if args.unsup:
        print('Using Unsup Data')
        all_unsup_ids = torch.load('data/aclImdb/unsup_ids.pt')
        unsup_labels = torch.load('data/aclImdb/unsup_labels.pt')
        tmp = len(unsup_labels) / train_size
        lambda_bs = tmp if tmp > 1. else 1./tmp
        # all_train_ids = torch.cat((all_train_ids,all_unsup_ids),dim=0)
        # labels = torch.cat((labels, unsup_labels), dim=0)
        unsup_dataset = TensorDataset(all_unsup_ids, unsup_labels)
        unsup_dataloader = DataLoader(unsup_dataset, batch_size=int(lambda_bs*batch_size), shuffle=True)


    # print(len(train_dataloader))
    # create optimizer and learning rate schedule
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'bert' in n]}, 
                                    {'params': [p for n, p in model.named_parameters() if 'bert' not in n],'lr':2e-3}]
    if args.child_tuning:
        print('Using child tuning...')
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        optimizer = ChildTuningAdamW(optimizer_grouped_parameters, lr=2e-5, reserve_p=args.reserve_p, mode="ChildTuning-F") # acc 很低 -_-b
        # optimizer = AdamW(model.classifier.parameters(),lr=1e-3)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-6,)
    else:
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * epochs
    # if args.unsup:
    #     total_steps += len(unsup_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # CELoss = torch.nn.CrossEntropyLoss()

    save_path = args.save_path 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_epoch = 0
    best_val = 0.0

    if args.cali:
        adv_trainer = CalibrateFreeLB(adv_K=args.adv_K, adv_lr=args.adv_lr,
                                adv_init_mag=args.adv_init_mag,
                                adv_norm_type=args.adv_norm_type,
                                base_model=args.base_model,
                                adv_max_norm=args.adv_max_norm,
                                f_b=args.f_b,
                                flooding=args.flooding)
        cali_model = AutoModelForSequenceClassification.from_pretrained(args.model_type, config=config)
        cali_model.load_state_dict(torch.load(args.cali_model_path))
        cali_model.cuda()
    elif freelb:
        adv_trainer = FreeLB(adv_K=args.adv_K, adv_lr=args.adv_lr,
                                adv_init_mag=args.adv_init_mag,
                                adv_norm_type=args.adv_norm_type,
                                base_model=args.base_model,
                                adv_max_norm=args.adv_max_norm,
                                f_b=args.f_b,
                                flooding=args.flooding)
    elif fgm:
        adv_trainer = FGM(model, emb_name='word_embeddings', epsilon=args.fgm_epsilon)
    if kl:
        kl_trainer = KL(adv_K=args.adv_K, adv_lr=args.adv_lr,
                                adv_init_mag=args.adv_init_mag,
                                adv_norm_type=args.adv_norm_type,
                                base_model='roberta' if args.base_model == 'vascl' else args.base_model,
                                adv_max_norm=args.adv_max_norm)
    if cl:
        bert_cl = BertForCL(model, base_model=args.base_model) 
    # keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    print(save_path)
    print('Start Trainging...')
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_cali_loss = 0 
        total_eval_accuracy = 0
        # for step, batch_mix in tqdm(enumerate(zip(train_dataloader, unsup_dataloader))):
        if args.unsup:
            unsup_dataloader_iterator = iter(unsup_dataloader)
        # if args.freelb:
        #     adv_trainer = FreeLB(adv_K=epoch+1, adv_lr=args.adv_lr,
        #                             adv_init_mag=args.adv_init_mag,
        #                             adv_norm_type=args.adv_norm_type,
        #                             base_model=args.base_model,
        #                             adv_max_norm=args.adv_max_norm,
        #                             f_b=args.f_b,
        #                             flooding=args.flooding)
        for step, batch in enumerate(train_dataloader): 
            if args.unsup:
                try:
                    unsup_batch = next(unsup_dataloader_iterator)
                except StopIteration:
                    unsup_dataloader_iterator = iter(unsup_dataloader)
                    unsup_batch = next(unsup_dataloader_iterator) 
                # cat sup and unsup 
                # batch = (torch.cat((batch[0], unsup_batch[0]), dim=0),torch.cat((batch[1], unsup_batch[1]), dim=0))
                unsup_inputs = build_inputs(unsup_batch)

            model.zero_grad()
            # print(batch)
            # print('------------------')
            # print(unsup_batch)
            # inputs = build_inputs(batch)
            # unsup_inputs = build_inputs(unsup_batch)
                
            inputs = build_inputs(batch)
            # print(inputs['input_ids'].size(0))
            # print(inputs['input_ids'].size(1))

            # unsup_inputs = build_inputs(unsup_batch)
            # bert_cl.cl_forward(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
            if args.cali:
                loss, logits, cali_loss, loss_item = adv_trainer.attack(model, inputs, cali_model=cali_model, temp=args.temp, gradient_accumulation_steps=args.gradient_accumulation_steps)
            elif freelb:
                loss, logits, loss_item = adv_trainer.attack(model, inputs, gradient_accumulation_steps=args.gradient_accumulation_steps)
                
            elif fgm:
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                loss.backward(retain_graph=True)

                adv_trainer.attack()
                adv_outputs = model(**inputs)
                loss_adv, logits_adv = adv_outputs[:2]
                loss_adv.backward()
                adv_trainer.restore()
            else:
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                loss_item = loss.item() 
                # if args.unsup:
                #     unsup_outputs = model(**unsup_inputs)
                #     unsup_loss, unsup_logits = unsup_outputs[:2]
                # flooding 
                if args.flooding:
                    loss = (loss - args.f_b).abs() + args.f_b
                    # if args.unsup:
                    #     unsup_loss = (unsup_loss - 0.2).abs() + 0.2 
                    #     unsup_loss.backward()
                # unsup_loss, unsup_logits = adv_trainer.attack(model, unsup_inputs, gradient_accumulation_steps=args.gradient_accumulation_steps)
            
                # todo kl散度 自监督对抗训练

                loss.backward()
            # if (step % 10 == 0) and args.kl:
            #     kl_loss, kl_logits, embeds_init, delta = kl_trainer.attack(model, unsup_inputs)
            #     if cl:
            #         sim = Similarity(temp=5e-2)

            #         embeds_perturb = embeds_init + delta 
            #         unsup_inputs['input_ids'] = None
            #         # init 
            #         unsup_inputs['inputs_embeds'] = embeds_init
            #         cls_init = bert_cl.cl_forward(**unsup_inputs)
            #         # attention_mask=unsup_inputs['attention_mask'], inputs_embeds=unsup_inputs['inputs_embeds']
            #         # perturb
            #         unsup_inputs['inputs_embeds'] = embeds_perturb
            #         cls_perturb = bert_cl.cl_forward(**unsup_inputs)
            #         # embeds_cat = torch.cat((embeds_init, embeds_perturb), dim=1)
            #         # print('-------------------------')
            #         # print(embeds_cat.shape)
            #         # print('-------------------------')
            #         cos_sim = sim(cls_init.unsqueeze(1), cls_perturb.unsqueeze(0))
            #         labels = torch.arange(cos_sim.size(0)).long().to(device)
            #         cl_loss = CELoss(cos_sim, labels)
            #         # cl_loss *= 0.5
            #         cl_loss.backward()
            

            # loss = CELoss(logits, batch[1].long().to(device))

            # total_loss += loss.item()
             
            total_loss += loss_item

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step() 
            scheduler.step()
            if args.cali:
                print('\033[0;31;40m{}\033[0m,step:{},training_loss:{},cali_loss:{}'\
                    .format(time.asctime(time.localtime(time.time())), step, loss_item, cali_loss.item()))
            if (step+1) % 20 == 0:
                logits = logits.detach().cpu().numpy()
                # unsup_logits = kl_logits.detach().cpu().numpy() 
                label_ids = batch[1].to('cpu').numpy()
                # unsup_label_ids = batch_mix[1][1].to('cpu').numpy()
                training_acc = flat_accuracy(logits, label_ids)
                # unsup_training_acc = flat_accuracy(unsup_logits, unsup_label_ids)
                print('\033[0;31;40m{}\033[0m,step:{},training_loss:{},training_acc:{}'\
                    .format(time.asctime(time.localtime(time.time())), step, loss_item, training_acc))
                # print('\033[0;31;40m{}\033[0m,step:{},training_loss:{},training_acc:{},kl_loss:{},cl_loss:{}'.format(time.asctime(time.localtime(time.time())), step, loss.item(), training_acc, kl_loss.item(), cl_loss.item()))

        # for step, batch in tqdm(enumerate(unsup_dataloader)): 
        #     model.zero_grad()
        #     inputs = build_inputs(batch)
        #     loss, logits = adv_trainer.attack(model, inputs, gradient_accumulation_steps=args.gradient_accumulation_steps)

        #     total_loss += loss.item()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #     optimizer.step() 
        #     scheduler.step()

        #     if (step+1) % 100 == 0:
        #         logits = logits.detach().cpu().numpy()
        #         label_ids = batch[1].to('cpu').numpy()
        #         training_acc = flat_accuracy(logits, label_ids)
        #         print('\033[0;31;40m{}\033[0m,step:{},training_loss:{},training_acc:{}'.format(time.asctime(time.localtime(time.time())), step, loss.item(), training_acc))

        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                inputs = build_inputs(batch)
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                # loss = CELoss(logits, batch[1].long().to(device))    
                total_val_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        if best_val < avg_val_accuracy:
            best_val = avg_val_accuracy
            best_epoch = epoch
        
        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Val Accuracy: {avg_val_accuracy}')
        print(f'Best Val Accuracy: {best_val}')
        print('Best Epoch:', best_epoch)

        print('Save model...')
        # torch.save(model.state_dict(), save_path+a_type+f_type+str(args.reserve_p)+'_'+str(epoch)+'.pt')
        torch.save(model.state_dict(), save_path+str(epoch)+'.pt')
        print('Done...')

def test(args):
    # model_type = None
    # if args.base_model == 'bert':
    #     model_type = 'bert-base-uncased' 
    # elif args.base_model == 'roberta':
    #     model_type = 'roberta-base' 
    # elif args.base_model == 'simcse':
    #     model_type = 'roberta-base' 

    save_path = args.save_path
    print('------------------------------------------------------------')
    print(save_path)
    print('------------------------------------------------------------')
    tokenizer, config, model = None, None, None  
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, do_lower_case=True)
    config = AutoConfig.from_pretrained(args.model_type, num_labels=args.num_labels, output_attentions=False, output_hidden_states=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type, config=config)
    batch_size = 128
    # tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
    # train_texts, train_labels, test_texts, test_labels = get_sst_data()
    if args.sst:
        test_texts, test_labels = get_sst_data('dev')
    elif args.agnews:
        test_texts, test_labels = get_agnews_data('test')
    all_test_ids = encode_fn(tokenizer, test_texts, dataset='sst' if args.sst else 'agnews')
    test_labels = torch.tensor(test_labels)
    pred_data = TensorDataset(all_test_ids, test_labels)
    pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)

    model.cuda()

    model.load_state_dict(torch.load(save_path))
    model.eval()
    total_test_accuracy = 0
    for i, batch in tqdm(enumerate(pred_dataloader)):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
            logits = outputs[0]
            
            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_test_accuracy += flat_accuracy(logits, label_ids)
    avg_test_accuracy = total_test_accuracy / len(pred_dataloader)

    print('test_acc:{}'.format(avg_test_accuracy))

    # for i in range(10):
    #     model.load_state_dict(torch.load(save_path+str(i)+'.pt'))
        
    #     model.eval()
    #     total_test_accuracy = 0
    #     for i, batch in tqdm(enumerate(pred_dataloader)):
    #         with torch.no_grad():
    #             outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
    #             logits = outputs[0]
                
    #             logits = logits.detach().cpu().numpy()
    #             label_ids = batch[1].to('cpu').numpy()
    #             total_test_accuracy += flat_accuracy(logits, label_ids)
    #     avg_test_accuracy = total_test_accuracy / len(pred_dataloader)

    #     print('test_acc:{}'.format(avg_test_accuracy))

def gen_pseudo_labels(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, do_lower_case=True)
    config = AutoConfig.from_pretrained(args.model_type, num_labels=args.num_labels, output_attentions=False, output_hidden_states=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type, config=config)
    model.cuda()

    print('Getting Unsup Data...')
    # unsup_data = get_imdb_unsup_data()
    # print('Encoding Unsup Data...')
    # all_unsup_ids = encode_fn(tokenizer, unsup_data, sst=args.sst)
    # torch.save(all_unsup_ids, 'data/aclImdb/unsup_ids.pt')
    all_unsup_ids = torch.load('data/aclImdb/unsup_ids.pt')
    unsup_dataset = TensorDataset(all_unsup_ids)
    unsup_dataloader = DataLoader(unsup_dataset, batch_size=args.batch_size, shuffle=True)

    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    all_logits = []
    for i, batch in tqdm(enumerate(unsup_dataloader)):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
            logits = outputs[0]
            logits = logits.detach().cpu()
            # logits = np.argmax(logits, axis=1).flatten()
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_logits = np.argmax(all_logits, axis=1).flatten()
    labels = torch.tensor(all_logits)
    torch.save(labels, 'data/aclImdb/unsup_labels.pt')
    return all_logits


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--fada", action='store_true', help="whether to use fada")
    argparser.add_argument("--fada_path", type=str, default='./data/FADA_sst.csv')
    argparser.add_argument("--ada", action='store_true', help="whether to use ada")
    argparser.add_argument("--ada_path", type=str, default='/media/rao/Disk-1/home/zb/GAT/TextAttack/fae/2022-09-23-20-56-log.csv')
    argparser.add_argument("--freelb", action='store_true', help="whether to use freelb")
    argparser.add_argument("--fgm", action='store_true', help="whether to use fgm")
    argparser.add_argument("--fgm_epsilon", type=float, default=1.0, help="fgm epsilon")
    argparser.add_argument('--adv_lr', type=float, default=1e-2)
    argparser.add_argument('--adv_K', type=int, default=3, help="should be at least 1")
    argparser.add_argument('--adv_init_mag', type=float, default=2e-2)
    argparser.add_argument('--adv_norm_type', type=str, default="l2", choices=["l2", "linf"])
    argparser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    argparser.add_argument('--base_model', type=str, default='bert', help="adv train model type")
    argparser.add_argument('--model_type', type=str, default='bert-base-uncased', help="model full name ")
    argparser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    argparser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    argparser.add_argument('--best_epoch', type=int, default=0)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--num_labels', type=int, default=2)
    argparser.add_argument('--do_train', action='store_true')
    argparser.add_argument('--do_test', action='store_true')
    argparser.add_argument('--aug_only', action='store_true')
    argparser.add_argument('--freelb_o', action='store_true')
    argparser.add_argument('--unsup', action='store_true')
    argparser.add_argument('--save_path', type=str, default="")
    argparser.add_argument('--only_cls', action='store_true')
    argparser.add_argument('--reserve_p', type=float, default='0.3')
    argparser.add_argument('--sst', action='store_true')
    argparser.add_argument('--agnews', action='store_true')
    argparser.add_argument('--kl', action='store_true')
    argparser.add_argument('--cl', action='store_true')
    argparser.add_argument('--flooding', action='store_true')
    argparser.add_argument('--child_tuning', action='store_true')
    argparser.add_argument('--f_b', type=float, default=0.)
    argparser.add_argument('--cali', action='store_true')
    argparser.add_argument('--cali_model_path', type=str, default="./sst_model/bert-base-uncased/0.pt")
    argparser.add_argument('--temp', type=float, default=1.)
    argparser.add_argument('--kl_alpha', type=float, default=1.)
    argparser.add_argument('--weight_decay', type=float, default=1e-2)
    argparser.add_argument('--dropout_r', type=float, default=0.1)
    args = argparser.parse_args()
    set_seed(2021)
    device = torch.device('cuda')

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    # print(gen_pseudo_labels(args))

