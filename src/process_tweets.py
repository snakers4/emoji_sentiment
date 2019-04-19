import pandas as pd
from glob import glob
from tqdm import tqdm
from utils.text_utils import (flatten_list,
                              emoji64_count,
                              split_into_batches,                              
                              list_multiprocessing,
                              process_twitter_log_file)

WORKERS = 8
GLOB = '../../../dump/10_TWEETS/2018/*/*/*/*.json.bz2'
OUTPUT = 'tweet_df_batch_{}.feather'

archive_files = sorted(glob(GLOB))
batches = split_into_batches(archive_files,200)
print('Processing {} batches'.format(len(batches)))

for i,batch in enumerate(batches):
    try:
        results = list_multiprocessing(batch,
                                       process_twitter_log_file,
                                       workers=WORKERS)

        texts = flatten_list([_[0] for _ in results])
        langs = flatten_list([_[1] for _ in results])
        probs = flatten_list([_[2] for _ in results])

        assert len(texts)==len(langs)
        assert len(langs)==len(probs)

        df = pd.DataFrame({
            'text':texts,
            'lang':langs,
            'prob':probs,
        })
        df['class'] = ''
        df['count'] = 0

        df['class'] = df['text'].apply(lambda x: emoji64_count(x).most_common(1)[0][0])
        df['count'] = df['text'].apply(lambda x: emoji64_count(x).most_common(1)[0][1])

        df.to_feather(OUTPUT.format(str(i).zfill(6)))
    except Exception as e: 
        print('Problem with batch {}'.format(i))
        print(e)        