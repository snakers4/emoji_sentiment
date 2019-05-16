import re
import os,sys
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import EmbeddingBag
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 


class SentimentConfig(object):
    def __init__(self,
                 
                 device=torch.device('cpu'),
                 dropout=0.1,
                 max_len=30,
                 
                 cat_sizes=None, # categorical embeddings
                 
                 embedding_type='dict', # dict, bag, wbag, char
                 emb_matrix=None, # set to None
                 emb_dict=None,
                 emb_shape=(1,50),
                 
                 # configs for each network type
                 context_network_type='rnn',
                 context_config={},
                 
                 classifier_mlp_sizes=[300,128,2],
                 classifier_model='mlp',
                 classifier_model_config={},
                 
                 pooling_type=0, # 0 mean pooling, 1 - one head, 2+ several heads
                ):
        
        self.device = device
        self.dropout = dropout
        self.max_len = max_len
        
        self.cat_sizes = cat_sizes
        
        self.embedding_type = embedding_type
        self.emb_matrix = emb_matrix
        self.emb_dict = emb_dict
        
        self.context_network_type = context_network_type
        self.context_config = context_config
        self.classifier_mlp_sizes = classifier_mlp_sizes
        self.classifier_model = classifier_model
        self.classifier_model_config = classifier_model_config
        self.pooling_type = pooling_type


    @classmethod
    def from_dict(cls, json_object):
        config = SentimentConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        config.check_params()
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
    
    
    def check_params(self):
        assert len(self.emb_shape)==2
        assert type(self.emb_shape)==tuple
        assert type(self.device)==torch.device
        assert type(self.context_config)==dict
        assert (self.emb_matrix is None or type(self.emb_matrix) == np.ndarray) # set matrix with a np array
        assert self.context_network_type in ['rnn','cnn']
        
        rnn_param_assert_dict = {
            'hidden_size':int,
            'device':torch.device,
            'bidirectional':bool,
            'gru':bool,
            'num_layers':int,
            'dropout':float,
            'input_embedding_size':int,
            'output_embedding_size':int        
        }
        cnn_param_assert_dict = {
            'num_inputs':int,
            'num_channels':list,
            'kernel_size':int,
            'dropout':float           
        }
        # check that all params have been passed to the constructor
        if self.context_network_type == 'rnn':
            for k,v in rnn_param_assert_dict.items():
                assert k in self.context_config
                assert type(self.context_config[k])==v
        elif self.context_network_type == 'cnn':
             for k,v in cnn_param_assert_dict.items():
                assert k in self.context_config
                assert type(self.context_config[k])==v
        
        if self.classifier_model == 'mlp':
            assert type(self.classifier_mlp_sizes)==list
        # classifier can also be a complex model in some cases
        elif self.classifier_model == 'cnn':
            assert type(self.classifier_model_config)==dict
            for k,v in cnn_param_assert_dict.items():
                assert k in self.classifier_model_config
                assert type(self.classifier_model_config[k])==v           
        elif self.classifier_model == 'rnn':
            assert type(self.classifier_model_config)==dict
            for k,v in rnn_param_assert_dict.items():
                assert k in self.classifier_model_config
                assert type(self.classifier_model_config[k])==v            
        
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"    
    

class SentimentClassifier(nn.Module):
    def __init__(self,
                 config):
        
        super(SentimentClassifier, self).__init__()
        
        self.config = config

        if self.config.embedding_type=='bag':
            self.emb = FastTextEmbeddingBag(ngram_dict=self.config.emb_dict,
                                            input_matrix=self.config.emb_matrix,
                                            device=self.config.device,
                                            input_shape=self.config.emb_shape,
                                            att_module=None)
        elif self.config.embedding_type=='wbag':
            self.emb = FastTextEmbeddingBag(ngram_dict=self.config.emb_dict,
                                            input_matrix=self.config.emb_matrix,
                                            device=self.config.device,
                                            input_shape=self.config.emb_shape,
                                            att_module=SelfAttentionEmb(self.config.emb_shape[1]))            
        elif self.config.embedding_type=='dict':
            self.emb = EmbeddingVocabulary(ngram_dict=self.config.emb_dict,
                                           input_matrix=self.config.emb_matrix,
                                           device=self.config.device,
                                           input_shape=self.config.emb_shape)     

        embedding_sizes = []
        if self.config.cat_sizes:
            print('Categorical embeddings initialized!')
            for i, cat_size in enumerate(self.config.cat_sizes):
                emb,emb_size = self.create_emb(cat_size=cat_size,max_emb_size=50)
                setattr(self, 'emb_{}'.format(i), emb)
                embedding_sizes.append(emb_size)
        self.embedding_sizes = embedding_sizes
        
        if self.config.context_network_type == 'rnn':
            # before adding the categorical embeddings
            self.first_c_size = self.config.context_config['input_embedding_size'] 
            self.config.context_config['input_embedding_size'] += sum(embedding_sizes)
            self.c_net = TemporalRecurrentNet(**self.config.context_config)
            self.last_c_size = self.config.context_config['output_embedding_size']
        else:
            # before adding the categorical embeddings
            self.first_c_size = self.config.context_config['num_inputs']
            self.config.context_config['num_inputs'] += sum(embedding_sizes)
            self.c_net = TemporalConvNet(**self.config.context_config)
            self.last_c_size = self.config.context_config['num_channels'][-1]
        
        self.attention = None
        
        if self.config.pooling_type==0:
            # plain classifier in case of mean pooling
            self.clf = self.get_classifier(self.config.classifier_model,
                                                       self.config.classifier_mlp_sizes,
                                                       self.config.classifier_model_config)             
            
        elif self.config.pooling_type==1:
            # plain classifier in case of self attention
            self.clf = self.get_classifier(self.config.classifier_model,
                                                       self.config.classifier_mlp_sizes,
                                                       self.config.classifier_model_config)                
            
            self.attention = SelfAttention(self.last_c_size,
                                           batch_first=False,
                                           non_linearity="tanh")
        elif self.config.pooling_type>1:
            # multi-head attention
            self.attention = SelfAttentionMulti(heads=self.config.pooling_type,
                                                attention_size=self.last_c_size,
                                                batch_first=False,
                                                non_linearity="tanh")
            # multi-head classifier
            # one classifier for each attention head
            self.clf = nn.ModuleList([self.get_classifier(self.config.classifier_model,
                                                          self.config.classifier_mlp_sizes,
                                                          self.config.classifier_model_config)  for i in range(self.config.pooling_type)])
            
            
    def get_classifier(self,
                       model_type,
                       mlp_sizes,
                       config):
        if model_type == 'mlp':
            # concatenate activations
            tdd_conf = [self.last_c_size] + mlp_sizes        

            modules = []
            for i in range(0,len(tdd_conf)-1):
                # last value does not need the RELU activation
                if i==(len(tdd_conf)-2):
                    is_last=True
                else:
                    is_last=False
                modules = self.linear_block(modules,
                                            tdd_conf[i],
                                            tdd_conf[i+1],
                                            is_last=is_last)

            classifier = nn.Sequential(*modules)
        elif model_type == 'rnn':
            classifier = TemporalRecurrentNet(**config)
        elif model_type == 'cnn':
            classifier = TemporalConvNet(**config)        
        return classifier
    
    
    def linear_block(self,
                     modules,
                     neurons_in,
                     neurons_out,
                     is_last=False):
        modules.append(nn.Dropout(self.config.dropout))        
        modules.append(nn.Linear(neurons_in, neurons_out))
        if not is_last:
            modules.append(nn.ReLU(True))        
        return modules  
    
    
    def create_emb(self,
                   cat_size = 7,
                   max_emb_size = 50,
                   embedding_factor = 3):

        emb_size = min([(cat_size+2)//embedding_factor, max_emb_size])    
        emb = nn.Embedding(num_embeddings = cat_size,
                           embedding_dim = emb_size)

        return emb,emb_size     
    
    
    def forward(self,
                c_seq,
                c_cat,
                c_lengths=None,
                ):
        
        batch_size = c_cat.size(0)
        max_len = self.config.max_len
        
        # assume basic lstm (batch,sequence,channels)
        # this should work with all embedding types
        c_embeds = self.emb(c_seq).view((-1,
                                         max_len,
                                         self.first_c_size))
        assert c_embeds.size(0) == batch_size
        
        # if we use categorical embeddings
        if self.config.cat_sizes:
            c_cat_embeds = [getattr(self,'emb_{}'.format(i))(c_cat[:,i,:]) 
                            for i,cat_size in enumerate(self.config.cat_sizes)]            
            # categorical embeddings should be (batch,sequence,embedding_channels)
            c_cat_embeds = torch.cat(c_cat_embeds,dim=2)
        
            # merge layers together
            c_embeds = torch.cat([c_embeds,c_cat_embeds],dim=2)
        
            assert c_embeds.size() == (batch_size,
                                       max_len,
                                       self.first_c_size +sum(self.embedding_sizes))
        
        if self.attention is not None:
            assert c_lengths is not None
            
        if self.config.context_network_type == 'rnn':
            if self.config.pooling_type==-1:
                c_state = self.c_net(c_embeds)
                c_state = c_state.mean(dim=1)
                # max pooling
                # c_state, _ = c_state.max(dim=1)                
            elif self.config.pooling_type==0:
                c_state = self.c_net(c_embeds)
                # max pooling
                c_state, _ = c_state.max(dim=1)
            elif self.config.pooling_type==1:
                c_state = self.c_net(c_embeds)
                # just attention scores are discarded, only representations are kept
                c_state, _ = self.attention(c_state, c_lengths)
            elif self.config.pooling_type>1:
                c_state = self.c_net(c_embeds)
                all_representations,_ = self.attention(c_state, c_lengths)
        else:
            if self.config.pooling_type==0:
                # note - tcn requires cnn-based format (batch,channels,sequence) as input
                c_state = self.c_net(c_embeds.permute(dims=(0,2,1))).permute(dims=(0,2,1))
                c_state = c_state.mean(dim=1)
                # max pooling
                # c_state, _ = c_state.max(dim=1)                
            elif self.config.pooling_type==1:
                c_state = self.c_net(c_embeds.permute(dims=(0,2,1))).permute(dims=(0,2,1))
                # just attention scores are discarded, only representations are kept
                c_state = self.attention(c_state, c_lengths)
            elif self.config.pooling_type>1:
                c_state = self.c_net(c_embeds.permute(dims=(0,2,1))).permute(dims=(0,2,1))
                all_representations,_ = self.attention(c_state, c_lengths)

        if self.config.pooling_type in [0,1]:
            return self.clf(c_state)
        elif self.config.pooling_type>1:
            outs = []
            # return a set of binary classifiers
            assert len(self.clf)==len(all_representations)
            for clf, rep in zip(self.clf,all_representations):
                outs.append(clf(rep))
            # a list of logits
            return torch.cat(outs, dim=1)
    
    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
    
    
    @staticmethod
    def is_parallel(model):
        return isinstance(model, torch.nn.parallel.DataParallel) or \
               isinstance(model, torch.nn.parallel.DistributedDataParallel)    
    
    
class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self,
                 ngram_dict=None,
                 device=None,
                 input_shape=None,
                 input_matrix=None,
                 att_module=None,
                 ):
        
        self.device = device
        self.ngram_dict = ngram_dict
        if att_module is None:
            mode = 'mean'
        else:
            mode = 'sum'
        
        if input_matrix is None:
            # standard init
            assert input_shape[0] == len(self.ngram_dict) 
            super().__init__(input_shape[0], input_shape[1],
                             mode=mode) 
        else:
            # init with pre-calculated matrix
            assert input_matrix.shape == input_shape
            assert len(self.ngram_dict) == input_matrix.shape[0]
            super().__init__(input_matrix.shape[0], input_matrix.shape[1],
                             mode=mode)       
            self.weight.data.copy_(torch.FloatTensor(input_matrix))
        self.att_module = att_module            

    def forward(self, words):
        # replace np arrays with list for speed
        # word_subinds = np.empty([0], dtype=np.int64)
        word_subinds = []
        word_offsets = [0]
        for word in words:
            # standard
            subinds = [self.ngram_dict[gram] for gram
                       in word_ngrams(word)
                       if gram in self.ngram_dict]
                
            if subinds == []:
                print(word)
                subinds.append(self.ngram_dict['UNK'])
                
            # word_subinds = np.concatenate((word_subinds, subinds))
            word_subinds.extend(subinds)
            word_offsets.append(word_offsets[-1] + len(subinds))
        
        _word_offsets = word_offsets[:-1]
        ind = torch.LongTensor(word_subinds).to(self.device)
        offsets = torch.LongTensor(_word_offsets).to(self.device)
        
        if self.att_module is None:
            return super().forward(ind, offsets)
        else:
            # very slow
            # calculate attention weights for each word
            per_sample_weights = []
            for i,offset in enumerate(word_offsets):
                if i==0:
                    pass
                else:
                    current_idx = word_subinds[word_offsets[i-1]:offset]
                    current_vectors = self.weight.data[current_idx,:]
                    weights = self.att_module(current_vectors)
                    per_sample_weights.append(weights)
            per_sample_weights = torch.cat(per_sample_weights,dim=0)
            return super().forward(ind, offsets, per_sample_weights)

        
class EmbeddingVocabulary(nn.Module):
    def __init__(self, 
                 ngram_dict=None,
                 input_matrix=None,
                 device=None,
                 input_shape=None):
        super(EmbeddingVocabulary, self).__init__()
        assert ' ' in ngram_dict
        self.ngram_dict = ngram_dict
        if input_matrix is None:
            assert input_shape[0] == len(self.ngram_dict)  
            self.word_embeddings = nn.Embedding(input_shape[0],
                                                input_shape[1])             
        else:
            assert input_matrix.shape == input_shape
            assert len(self.ngram_dict) == input_matrix.shape[0]
            self.word_embeddings.from_pretrained(torch.FloatTensor(input_matrix))            

        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids):
        seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return words_embeddings
    

# TODO
class EmbeddingCharCNN(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

    def forward(self, input_ids, token_type_ids=None):
        return embeddings          
        

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,
                 num_inputs=320,
                 num_channels=[500, 500, 500, 500, 300],
                 kernel_size=2,
                 dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)    


class TemporalRecurrentNet(nn.Module):
    def __init__(self,
                 hidden_size=300,
                 device=torch.device('cpu'),
                 bidirectional=False,
                 gru=True,
                 num_layers=1,
                 dropout=0.2,
                 input_embedding_size=300,
                 output_embedding_size=300):
        
        super(TemporalRecurrentNet, self).__init__()
          
        self.gru = gru
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.bidirectional = bidirectional
        
        self.input_embedding_size=input_embedding_size
        self.output_embedding_size=output_embedding_size
        
        if not self.gru:
            self.lstm = nn.LSTM(self.input_embedding_size,
                                self.hidden_size,
                                bidirectional=self.bidirectional,
                                batch_first=True,
                                num_layers=self.num_layers,
                                dropout=dropout )
        else:
            self.lstm = nn.GRU(self.input_embedding_size,
                               self.hidden_size,
                               bidirectional=self.bidirectional,
                               batch_first=True,
                               num_layers=self.num_layers,
                               dropout=dropout)
         
        # compress the biLSTM state 
        self.linears = nn.Sequential((nn.Linear((1+int(self.bidirectional)) * self.hidden_size,
                                                self.output_embedding_size)),
                                      nn.Dropout(p=dropout),
                                      nn.LeakyReLU())
    
    def init_hidden(self,
                    batch_size):
        if not self.gru:
            return (torch.zeros(self.num_layers * (1+int(self.bidirectional)),
                                batch_size,
                                self.hidden_size).to(self.device),
                    torch.zeros(self.num_layers * (1+int(self.bidirectional)),
                                batch_size,
                                self.hidden_size).to(self.device))
        else:
            return torch.zeros(self.num_layers * (1+int(self.bidirectional)),
                               batch_size,
                               self.hidden_size).to(self.device)
    
    def forward(self,
                embeds):
        # assuming that embedding is handled in the top model
        batch_size = embeds.size(0)
        self.hidden = self.init_hidden(batch_size)
        assert embeds.size(2) == self.input_embedding_size
        
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        linears = self.linears(lstm_out)
        return linears  


def find_ngrams(string, n):
    ngrams = zip(*[string[i:] for i in range(n)])
    ngrams = [''.join(_) for _ in ngrams]
    return ngrams


def word_ngrams(string,
                min_len=1,
                max_len=6):
    ngrams = []
    for i in range(min_len,max_len+1):
        ngrams.extend(find_ngrams('<'+string+'>',i))
    if len(string) < 3:
        ngrams.append(string)
    return ngrams    


class SelfAttention(nn.Module):
    def __init__(self,
                 attention_size,
                 batch_first=False,
                 non_linearity="tanh"):
        
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = (torch.ones(attentions.size())).detach()

        _device = attentions.device
        mask = mask.to(_device)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores
    

class SelfAttentionMulti(nn.Module):
    def __init__(self,
                 heads=10,
                 attention_size=256,
                 batch_first=False,
                 non_linearity="tanh"):
        
        super(SelfAttentionMulti, self).__init__()

        self.attentions = nn.ModuleList([SelfAttention(attention_size,
                                                       batch_first=False,
                                                       non_linearity="tanh") for i in range(heads)])


    def forward(self, inputs, lengths):
        all_representations = []
        all_scores = []
                 
        for i, l in enumerate(self.attentions):
            representations, scores = l(inputs, lengths)
            all_representations.append(representations)
            all_scores.append(scores)
        
        # (batch, attention_head, length, dims)
        # all_representations = torch.stack(all_representations, dim=1)
        # (batch, attention_head, length)
        # all_scores = torch.stack(all_scores, dim=1)
        return all_representations,all_scores

    
class SelfAttentionEmb(nn.Module):
    def __init__(self,
                 attention_size,
                 non_linearity="tanh"):
        
        super(SelfAttentionEmb, self).__init__()

        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)

        
    def forward(self, inputs):
        # inputs is a 2D Tensor: len, hidden_size
        # scores is a 1D Tensor: len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)
        return scores    


def tokenize(text):
    return list(text)


class Tokenizer(object):
    def __init__(self,
                 model_path,
                 models_dir='bpe_models/'):
        sp = spm.SentencePieceProcessor()
        sp.Load(os.path.join(models_dir,model_path))
        self.sp = sp
        
    def tokenize(self,text):
        return self.sp.EncodeAsPieces(text)
    
    def enc_text2seq(self,text):
        return self.sp.EncodeAsIds(text)

    def enc_text2pieces(self,text):
        return self.sp.EncodeAsPieces(text)

    def dec_pieces2text(self,piece_list):
        return self.sp.DecodePieces(piece_list)    
    
    
def main():
    import gc
    import copy
    from string import printable    
    from string import punctuation
    
    device = torch.device('cpu')

    context_config = {
        'hidden_size':300,
        'device':device,
        'bidirectional':True,
        'gru':False,
        'num_layers':2,
        'dropout':0.1,
        'input_embedding_size':50,
        'output_embedding_size':300
    }

    config_dict = {
        'device':device,
        'dropout':0.1,
        'max_len':6,

        'cat_sizes':[5],

        'embedding_type':'bag',
        'emb_matrix':None,
        'emb_dict':{char:i for i,char in enumerate(printable)},
        'emb_shape':(len(printable),50),

        # configs for each network type
        'context_network_type':'rnn',
        'context_config':context_config,

        'classifier_mlp_sizes':[300,128,2],
        'classifier_model':'mlp',
        'classifier_model_config':None,

        'pooling_type':0
    }
    
    print('Testing plain embedding bag')
    # plain embedding bag test case
    config = SentimentConfig.from_dict(copy.deepcopy(config_dict))
    model = SentimentClassifier(config)
    c_seq = np.asarray(['The cat is on the mat'.split(' '), 'And it is quite well fed'.split(' ')]).reshape(-1)
    c_cat = torch.zeros(2,1,6).long()
    c_lengths = (torch.ones(2,1)*6).long()
    out = model(c_seq,c_cat,c_lengths)
    print(out.size())
    print()

    print('Testing attention')
    # test attention
    config = SentimentConfig.from_dict({**copy.deepcopy(config_dict),**{'pooling_type':1}})
    model = SentimentClassifier(config)
    c_seq = np.asarray(['The cat is on the mat'.split(' '), 'And it is quite well fed'.split(' ')]).reshape(-1)
    c_cat = torch.zeros(2,1,6).long()
    c_lengths = (torch.ones(2,1)*6).long()
    out = model(c_seq,c_cat,c_lengths)
    print(out.size())
    print()

    print('Testing multi-head attention')
    # test multi-head attention
    config = SentimentConfig.from_dict({**copy.deepcopy(config_dict),**{'pooling_type':11,
                                                                        'classifier_mlp_sizes':[300,128,1]}})
    model = SentimentClassifier(config)
    c_seq = np.asarray(['The cat is on the mat'.split(' '), 'And it is quite well fed'.split(' ')]).reshape(-1)
    c_cat = torch.zeros(2,1,6).long()
    c_lengths = (torch.ones(2,1)*6).long()
    out = model(c_seq,c_cat,c_lengths)
    print(out.size())
    print()

    print('Testing bag with attention')
    # test bag with attention
    config = SentimentConfig.from_dict({**copy.deepcopy(config_dict),**{'pooling_type':1,
                                                                        'embedding_type':'wbag'}})
    model = SentimentClassifier(config)
    c_seq = np.asarray(['The cat is on the mat'.split(' '), 'And it is quite well fed'.split(' ')]).reshape(-1)
    c_cat = torch.zeros(2,1,6).long()
    c_lengths = (torch.ones(2,1)*6).long()
    out = model(c_seq,c_cat,c_lengths)
    print(out.size())
    print()

    print('Testing plain embedding dictionary')
    # plain embedding bag test case
    config = SentimentConfig.from_dict({**copy.deepcopy(config_dict),
                                        **{'embedding_type':'dict'}})
    model = SentimentClassifier(config)
    c_seq = torch.LongTensor([[0,0,0,0,0,0],
                              [0,0,0,0,0,0]])
    c_cat = torch.zeros(2,1,6).long()
    c_lengths = (torch.ones(2,1)*6).long()
    out = model(c_seq,c_cat,c_lengths)
    print(out.size())
    print()

if __name__ == '__main__':
    main()    