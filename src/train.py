import os
import re
import gc
import sys
import glob
import time
import copy
import random
import shutil
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from tensorboardX  import SummaryWriter
from utils.address_dataset import pckl,upkl
from models.baseline_model  import str2bool
from utils.dataset import EmotionDataset as Corpus
from models.model  import SentimentConfig, SentimentClassifier

module_path = os.path.abspath(os.path.join('../'))
module_path = os.path.abspath(os.path.join('../../'))

if module_path not in sys.path:
    sys.path.append(module_path)

parser = argparse.ArgumentParser(description='Train question embeddings with plain model')

# dataset params
parser.add_argument('--seed',             default=228,   type=int)
parser.add_argument('--df_path',          default='../data/fin_tweets_train.feather',         type=str)
parser.add_argument('--holdout_share',    default=0.25,  type=float)
parser.add_argument('--language',         default='en',  type=str)

# general model params
parser.add_argument('--dropout',              default=0.1,     type=float)
parser.add_argument('--context_network_type', default='rnn',   type=str)
parser.add_argument('--classifier_model',     default='mlp',   type=str)
parser.add_argument('--classifier_mlp_sizes', default=[100,50,11], nargs='+', type=int)
parser.add_argument('--max_len',              default=50,      type=int)
parser.add_argument('--pooling_type',         default=0,       type=int)
parser.add_argument('--embedding_type',       default='bag',   type=str)
parser.add_argument('--embedding_dict',       default='',      type=str)
parser.add_argument('--embedding_matrix',     default='',      type=str)
# sentence piece model
parser.add_argument('--tokenizer_path',       default='',      type=str)
parser.add_argument('--cat_sizes',            default=[],      nargs='+', type=int)

# shared context model params
parser.add_argument('--input_embedding_size',   default=100,   type=int)
parser.add_argument('--output_embedding_size',  default=100,   type=int)

# rnn only params
parser.add_argument('--hidden_size',            default=100,   type=int)
parser.add_argument('--bidirectional',          default=False, type=str2bool)
parser.add_argument('--gru',                    default=False, type=str2bool)
parser.add_argument('--num_layers',             default=2,     type=int)

# cnn only params
parser.add_argument('--num_channels',    default=[500, 500, 500, 500, 300], nargs='+', type=int)
parser.add_argument('--kernel_size',     default=2, type=int)

# optimization params
parser.add_argument('--batch_size',  default=512,    type=int)
parser.add_argument('--num_workers', default=0,      type=int)
parser.add_argument('--predict',     default=False,  type=str2bool)
parser.add_argument('--optimizer',   default='adam', type=str)
parser.add_argument('--lr',          default=1e-3,   type=float)
parser.add_argument('--epochs',      default=50,     type=int)
parser.add_argument('--resume',      default=None)

# logging
parser.add_argument('--print_freq',  default=1,    type=int)
parser.add_argument('--tensorboard', default=False, type=str2bool)
parser.add_argument('--tb_name',     default=None)

# global
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {} to train on'.format(torch.cuda.get_device_name(0)))

print(args)
best_score = 100000
train_minib_counter = 0
valid_minib_counter = 0

tb_name = args.tb_name
start_epoch = 0

# Set the Tensorboard logger
if args.tensorboard:
    writer = SummaryWriter('runs/{}'.format(args.tb_name))

def main():
    global args, best_score, device, tb_name, start_epoch, writer
    
    emb_dict = upkl(args.embedding_dict)
    emb_matrix = None # for warm start
    emb_shape = (len(emb_dict),
                 args.input_embedding_size)
    
    if args.embedding_type in ['dict','bag','wbag']:
        assert ' ' in emb_dict
        assert 'UNK' in emb_dict
    
    if args.embedding_type in ['bpe']:
        assert os.path.isfile(args.tokenizer_path)
        
    print('Embedding matrix shape {}'.format(emb_shape.shape))
      
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_score = checkpoint['best_score']
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        checkpoint = None

    corpus_kwargs = {
        'mode':'train',
        'max_input_len':args.max_len,
        'holdout_share':args.holdout_share,
        'df_path':args.df_path,
        'language':args.language,
        'random_state':args.seed,
        'embedding_type':args.embedding_type,
        'tokenizer_path':args.tokenizer_path,
        'pad_token_id':0,
        'w2i':emb_dict
    }
        
    train_dataset = Corpus(**corpus_kwargs)
    val_dataset = Corpus(**{**corpus_kwargs,
                            **{'mode': 'val'}})

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True)    

    # cnn or rnn encoder
    rnn_kwargs = {
        'hidden_size':args.hidden_size,
        'device':device,
        'bidirectional':args.bidirectional,
        'gru':args.gru,
        'num_layers':args.num_layers,
        'dropout':args.dropout,
        'input_embedding_size':args.input_embedding_size,
        'output_embedding_size':args.output_embedding_size
    }
    cnn_kwargs = {
        'num_inputs':args.input_embedding_size,
        'num_channels':args.num_channels,
        'kernel_size':args.kernel_size,
        'dropout':args.dropout, 
    }
    
    if args.context_network_type=='rnn':
        context_config = rnn_kwargs
    else:
        context_config = cnn_kwargs       
        
    if args.classifier_model=='mlp':
        classifier_kwargs = None
    else
        raise NotImplementedError()
    
    if args.cat_sizes == []:
        cat_sizes
    elif type(args.cat_sizes)==list:
        cat_sizes = args.cat_sizes
    else:
        raise ValueError('Wrong cat sizes param')
        
    config_dict = {
        'device':device,
        'dropout':args.dropout,
        'max_len':args.max_len,
        'cat_sizes':cat_sizes,
        'embedding_type':args.embedding_type,
        'emb_matrix':emb_matrix,
        'emb_dict':emb_dict,
        'emb_shape':emb_shape,
        'context_network_type':args.context_network_type,
        'context_config':context_config,
        'classifier_mlp_sizes':args.classifier_mlp_sizes,
        'classifier_model':args.classifier_model,
        'classifier_model_config':classifier_kwargs,
        'pooling_type':args.pooling_type
    }
    
    config = SentimentConfig.from_dict(copy.deepcopy(config_dict))
    model = SentimentClassifier(config)
    param_count = model.get_param_size(model)  
    model = model.to(device)
    params = model.parameters()
    print("Model params {:,}".format(param_count))
    print(model)
    
    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                     lr=args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, params),
                                        lr=args.lr)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params),
                                    lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
    
    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=10,
                                  factor=0.5,
                                  verbose=True)
    
    # scheduler = StepLR(optimizer, 37, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()
    
    # model = nn.DataParallel(model)

    print('Training starts...')
    
    for epoch in range(args.epochs)[start_epoch:]:

        (train_loss, train_top1, train_top3) = train(train_loader,
                                                     model,
                                                     criterion,
                                                     optimizer,
                                                     device,
                                                     epoch,
                                                     loop_type='Train')

        (val_loss, val_top1, val_top3) = train(val_loader,
                                               model,
                                               criterion,
                                               optimizer,
                                               device,
                                               epoch,
                                               loop_type='Val')

        scheduler.step(val_loss)
        
        if args.tensorboard:
            writer.add_scalars('epoch/total_losses', {'train_loss': train_loss,
                                                      'val_loss': val_loss},
                               epoch+1)
            writer.add_scalars('epoch/top1', {'train_top1': train_top1,
                                              'val_top1': val_top1},
                               epoch+1)
            writer.add_scalars('epoch/top3', {'train_top3': train_top3,
                                              'val_top3': val_top3},
                               epoch+1)              
   
        is_best = val_loss < best_score
        best_score = min(val_loss, best_score)

        if not os.path.exists('model_saves/'):
            os.makedirs('model_saves/')
            
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_score': best_score,
            },
            is_best,
            'model_saves/{}_checkpoint.pth.tar'.format(str(tb_name)),
            'model_saves/{}_best.pth.tar'.format(str(tb_name))
        )
        gc.collect()
                
def train(train_loader,
          model,
          criterion,
          optimizer,
          device,
          epoch,
          loop_type='Train'):
    
    global train_minib_counter
    global valid_minib_counter
    global writer

    context = torch.no_grad() if loop_type=='Val' else torch.enable_grad()
    
    if loop_type=='Train':
        model.train()
    elif loop_type=='Val':
        model.eval()
    else:
        raise ValueError('Wrong loop type')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    top1_accs = AverageMeter()
    top3_accs = AverageMeter()

    # switch to train mode
    end = time.time()

    print('Starting {} epoch {}'.format(loop_type,
                                        epoch))
    
    with context:
        for i, (tup) in enumerate(tqdm(train_loader)):
            
            (texts,
             lens,
             target) = tup
            
            # measure data loading time
            data_time.update(time.time() - end)

            if args.embedding_type in ['bag','wbag']:
                # in case of embedding bag
                # we pass padded texts in numpy arrays
                texts = np.array(texts).T.reshape(-1)
            else:
                # in case of dicts we just pass tensors
                texts = texts.long().to(device)
                
            # embeddings = embeddings.long().to(device)
            lens = lens.long().to(device)

            clf_logits = model(c_seq=texts,
                               c_cat=None,
                               c_lengths=lens

            loss = criterion(clf_logits, target)
            
            # compute gradient and do SGD step
            if loop_type=='Train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # record losses
            losses.update(loss.item(), lens.size(0))

            top1_acc, top3_acc = accuracy(clf_logits.detach(),
                                          target,
                                          topk=(1, 3))  
                               
            top1_accs.update(top1_acc.item(), lens.size(0))
            top3_accs.update(top3_acc.item(), lens.size(0))
                               
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()        

            # log the current lr
            if args.optimizer == 'adamw':
                current_lr = optimizer.lr_log
            else:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']

            #============ TensorBoard logging ============#
            if loop_type=='Train':      
                if args.tensorboard:
                    writer.add_scalar('train/train_loss', losses.val, train_minib_counter)
                    writer.add_scalar('train/train_lr', current_lr, train_minib_counter)  
                    train_minib_counter += 1
                
            elif loop_type=='Val':     
                if args.tensorboard:
                    writer.add_scalar('val/val_loss', losses.val, valid_minib_counter)    
                    valid_minib_counter += 1                

            if i % args.print_freq == 0:
                tqdm.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time   {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      'Data   {data_time.val:.2f} ({data_time.avg:.2f})\t'
                      'Loss   {losses.val:.3f} ({losses.avg:.3f})\t'                     
                      'Top 1  {top1_accs.val:.1f} ({top1_accs.avg:.1f})\t'
                      'Top 3  {top3_acc.val:.1f} ({top3_acc.avg:.1f})\t'.format(
                          epoch, i, len(train_loader),
                          batch_time=batch_time,data_time=data_time,
                          losses=losses,
                          top1_accs=top1_accs,top3_acc=top3_acc))


    return_tuple = (losses.avg,
                    top1_accs.avg,
                    top3_acc.avg)
    
    gc.collect()
    return return_tuple

def predict(val_loader, model, device):
    pass
    
def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res            
        
if __name__ == '__main__':
    main()