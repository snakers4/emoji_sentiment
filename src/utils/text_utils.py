import re
import bz2
import json
import string
import pickle
import unicodedata
import fastText as ft
from tqdm import tqdm
from itertools import groupby
from string import punctuation
from collections import Counter
from multiprocessing import Pool
from utils.emoji_reader import get_emoji_codes


ft_model = ft.load_model('../models/lid.176.bin')
emoji_dict, emoji_64_dict = get_emoji_codes('utils/emoji-test.txt')
emojis = sorted(list(emoji_dict.keys()))
emojis_64 = sorted(list(emoji_64_dict.keys()))
emojis_non64 = sorted(set(emoji_dict.keys()).difference(set(emoji_64_dict)))


all_chars = (chr(i) for i in range(0x110000))
# control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')
# or equivalently and much more efficiently
control_chars = ''.join(map(chr, list(range(0,32)) + list(range(127,160))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

special_commands = ['&amp;',
                     '&lt;',
                     '&br;',
                     '&gt;']


flatten_list = lambda l: [item for sublist in l for item in sublist]
str_contains = lambda x,y: x in y 

FILTER_CONFIG = {
    'min_len':5, # chars or words due to Chinese
    'only_one_type':True,
    'exclude_non64':True,
    'only_one64_block':False
}

def list_multiprocessing(param_lst,
                         func,
                         **kwargs):
    
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)


def remove_non_printed_chars(text):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ0-9{}{}]'.format(punctuation,
                                                       string.printable))
    return reg.sub(' ', text)


def trim_string(text):
    # remove extra spaces, remove trailing spaces
    return re.sub('\s+',' ',text).strip()


# all of the plain pre-processing except for lowering the case
# case will be embedded separately
def process_text(text):
    return trim_string(remove_non_printed_chars(text)).replace('. ','').replace(', ', '')


def read_line(fname, line_count=-1):
    lines = []
    with open(fname, encoding='utf-8') as f:
        for i, l in enumerate(f):
            lines.append(l)
            if i>line_count and line_count>0:
                break
    return lines


def find_ngrams(string, n):
    ngrams = zip(*[string[i:] for i in range(n)])
    ngrams = [''.join(_) for _ in ngrams]
    return ngrams


def word_ngrams(string, max_len=6):
    ngrams = []
    for i in range(3,max_len+1):
        ngrams.extend(find_ngrams('<'+string+'>',i))
    if len(string) < 3:
        ngrams.append(string)
    return ngrams


def space_punctuation(text):
    for p in punctuation:
        text = text.replace(p, ' '+p+' ')
    return text    


def process_text_dataset(text):
    return trim_string(space_punctuation(text.lower()))


def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _   


def has_punctuation(text):
    for _ in punctuation:
        if _ in text:
            return True
    return False


def remove_control_chars(s):
    return control_char_re.sub('', s)


def remove_non_printed_chars(text):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ0-9{}{}]'.format(string.punctuation,
                                                       string.printable))
    return reg.sub(' ', text)


def trim_string(text):
    # remove extra spaces, remove trailing spaces
    return re.sub('\s+',' ',text).strip()


def process_tweets(text):
    return  space_punctuation(
        remove_tags(
            remove_links(
                remove_control_chars(
                    text
                )
            )
        )
    )


def space_punctuation(text,
                      remove=True,
                      max_smile_len=4):
    
    p_mask = p_mask = [1 if _ in punctuation else 0 for _ in text]
    
    if remove:
        txt_spaces = []
        for i in range(0,len(p_mask)):
            if text[i] in punctuation:
                txt_spaces.append(' ')
            else:
                txt_spaces.append(text[i])
    else:
        # space them
        
        txt_spaces = []
        txt_spaces.append(text[0])

        smile_len = 0
        for i in range(1,len(p_mask)):

            # smile start
            if p_mask[i]==1 and p_mask[i-1]==0:
                smile_len = 0

            if (p_mask[i]==0 and p_mask[i-1]==1) or (p_mask[i]==1 and p_mask[i-1]==0):
                txt_spaces.append(' ')
                txt_spaces.append(text[i])   
            else:
                smile_len += 1
                if text[i] in punctuation:
                    if smile_len<max_smile_len:
                        txt_spaces.append(text[i]) 
                else:
                    txt_spaces.append(text[i])
        
    return trim_string(''.join(txt_spaces))


def remove_links(text):
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def remove_tags(text):
    # twitter specific pre-processing
    text = text.replace('\n',' ')
    text = text.replace('RT ','')
    
    for _ in special_commands:
        text = text.replace(_,' ')
    
    text = trim_string(text)
    
    words = text.split(' ')
    
    cleaned_words = []
    for word in words:
        if not word.startswith('@') and not word.startswith('#'):
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words)


def remove64_emojis(text):
    global emojis_64
    for emoji in emojis_64:
        text = text.replace(emoji,'')
    return text
    

def emoji64_count(text):
    global emojis_64
    counter_64 = Counter()
    for emoji in emojis_64:
        counter_64[emoji]=text.count(emoji)
    return counter_64


def detailed_emoji_count(text):
    global emojis_64,emojis_non64
    counter_64 = Counter()
    counter_non64 = Counter()
    for emoji in emojis_64:
        counter_64[emoji]=text.count(emoji)
    for emoji in emojis_non64:
        counter_non64[emoji]=text.count(emoji)        
    return counter_64,counter_non64


def filter_emoji(text):
    global FILTER_CONFIG
    counter_64,counter_non64 = detailed_emoji_count(text)
    text_wo_emoji = remove64_emojis(text)
    # filter - any necessary emojis
    if sum(counter_64.values())==0:
        return False
    # filter - no other emojis       
    elif sum(counter_non64.values())>0 and FILTER_CONFIG['exclude_non64']:
        return False
    # filter - one emoji type
    elif sum([1 for k,v in counter_64.items() if v>0])>1 and FILTER_CONFIG['only_one_type']:
        return False
    # filter - minimum words except for an emoji
    elif len(text_wo_emoji.split(' '))<FILTER_CONFIG['min_len'] and len(text_wo_emoji)<FILTER_CONFIG['min_len']:
        return False
    
    if  FILTER_CONFIG['only_one64_block']:
        # filter - one consequtive block of emojis
        # at this moment we already have only one emoji type
        emoji_tuple = counter_64.most_common(1)[0]
        grby = [(label, sum(1 for _ in group)) for label,
                group in groupby(text.replace(emoji_tuple[0],
                                              '⁂'))]
        if len([1 for _ in grby if _[0]=='⁂'])>1:
            return False
    return counter_64.most_common(1)[0]


def process_twitter_log_file(filename):
    global ft_model
    with bz2.open(filename, "rt") as bzinput:
        lines = []
        for i, line in enumerate(bzinput):
            # if i == 10: break
            tweets = json.loads(line)
            lines.append(tweets)    
    
    texts = []
    langs = []
    probs = []
    
    for tweet in lines:
        emoji_count = 0
        if 'text' in tweet:
            tweet_text = tweet['text'] 
            tweet_text = process_tweets(tweet_text)
            single_emoji = filter_emoji(tweet_text)
        if single_emoji:
            texts.append(tweet_text)
            
            
    # naive deduplicate
    texts = list(set(texts))    
    
    for text in texts:
        language_pred = ft_model.predict(text, k=1)
        try:
            langs.append(language_pred[0][0].replace('__label__',''))
        except:
            langs.append('none')
        try:
            probs.append(language_pred[1][0])
        except:
            probs.append(0)                    
                
    return texts,langs,probs