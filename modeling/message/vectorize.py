import random

import numpy as np
import pandas as pd
import multiprocessing
from pandarallel import pandarallel
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument

tqdm.pandas(desc="progress-bar")
cores = multiprocessing.cpu_count()
pandarallel.initialize()
msg_data = pd.read_parquet("/Users/saransh/Desktop/practicum/data/Messaging.parquet")
msg_data.reset_index(inplace=True)


def str_to_tokens(token_str):
    return token_str.split()


msg_data['tokenized_msg'] = msg_data.tokenized_msg.parallel_apply(str_to_tokens)
msg_types = dict(
    user_id=np.int64,
    stylist_id=np.int64
)

for col, type_ in msg_types.items():
    msg_data[col] = msg_data[col].astype(type_)

random.seed(60616)
msg_data_ind = list(msg_data.index)
random.shuffle(msg_data_ind)  # shuffle so as to allow a better optimum/ fit via stochastic gradient in neural network

train_doc_list = msg_data.loc[msg_data_ind]
train_tagged = train_doc_list.apply(lambda r: TaggedDocument(words=r['tokenized_msg'], tags=[r['index']]), axis=1)

skip_gram_model = Doc2Vec(dm=0, vector_size=100, workers=cores, epochs=50, min_count=2, negative=5, hs=0, sample=0)
skip_gram_model.build_vocab(train_tagged.tolist())
skip_gram_model.train(train_tagged, total_examples=len(train_tagged), epochs=skip_gram_model.epochs)
# skip_gram_model.infer_vector()
dm_model = Doc2Vec(dm=1, vector_size=100, workers=cores, epochs=50, min_count=2, negative=5, hs=0,
                   sample=0)
dm_model.build_vocab(train_tagged.values)
dm_model.train(train_tagged, total_examples=len(train_tagged), epochs=dm_model.epochs)

dm_model_concat = Doc2Vec(dm=1, dm_concat=1, vector_size=100, workers=cores, epochs=50, min_count=2, negative=5, hs=0,
                          sample=0)
dm_model_concat.build_vocab(train_tagged.tolist())
dm_model_concat.train(train_tagged, total_examples=len(train_tagged), epochs=dm_model_concat.epochs)
mixed_model = ConcatenatedDoc2Vec([skip_gram_model, dm_model])
mixed_model_with_concat = ConcatenatedDoc2Vec([skip_gram_model, dm_model_concat])


def populate_doc_vectors_to_dataframe(series_data):
    for i, vec in enumerate(mixed_model_with_concat.docvecs[series_data['index']]):
        series_data[f"docvec_{i}"] = vec
    return series_data


msg_data = msg_data.parallel_apply(populate_doc_vectors_to_dataframe, axis=1)
doc_vectors = []
for ind_ in msg_data.index:
    doc_vectors.append(
        mixed_model_with_concat.docvecs[ind_]
    )

doc_vec_df = pd.DataFrame(doc_vectors)

doc_vec_df.columns = [f"docvec_{col}" for col in range(200)]
msg_data = pd.concat([msg_data, doc_vec_df], axis=1)
msg_data.drop('index', axis=1, inplace=True)
