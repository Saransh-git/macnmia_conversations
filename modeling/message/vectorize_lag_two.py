import multiprocessing
import random

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from pandarallel import pandarallel


cores = multiprocessing.cpu_count()
pandarallel.initialize()


order_level_two_month = pd.read_parquet("/Users/saransh/Desktop/practicum/data/lagtwo.parquet")
order_level_two_month.reset_index(inplace=True)


def str_to_tokens(token_str):
    return token_str.split()


order_level_two_month['customer_tokenized_msg'] = order_level_two_month.customer_tokenized_msg.parallel_apply(
    str_to_tokens)
order_level_two_month['stylist_tokenized_msg'] = order_level_two_month.stylist_tokenized_msg.parallel_apply(
    str_to_tokens)


'''
msg_types = dict(
    user_id=np.int64,
    stylist_id=np.int64
)

for col, type_ in msg_types.items():
    msg_data[col] = msg_data[col].astype(type_)
'''

random.seed(60616)
msg_data_ind = list(order_level_two_month.index)
random.shuffle(msg_data_ind)  # shuffle so as to allow a better optimum/ fit via stochastic gradient in neural network

train_doc_list = order_level_two_month.loc[msg_data_ind]
train_tagged_c = train_doc_list.apply(lambda r: TaggedDocument(words=r['customer_tokenized_msg'], tags=[r['index']]),
                                    axis=1)

train_tagged_s = train_doc_list.apply(lambda r: TaggedDocument(words=r['stylist_tokenized_msg'], tags=[r['index']]),
                                    axis=1)

skip_gram_model = Doc2Vec(dm=0, vector_size=50, workers=cores, epochs=100, min_count=2, negative=5, hs=0, sample=0)
skip_gram_model.build_vocab(train_tagged_c.tolist())
skip_gram_model.train(train_tagged_c, total_examples=len(train_tagged_c), epochs=skip_gram_model.epochs)


dm_model_concat = Doc2Vec(dm=1, dm_concat=1, vector_size=50, workers=cores, epochs=100, min_count=2, negative=5, hs=0,
                          sample=0)
dm_model_concat.build_vocab(train_tagged_c.tolist())
dm_model_concat.train(train_tagged_c, total_examples=len(train_tagged_c), epochs=dm_model_concat.epochs)

# mixed_model = ConcatenatedDoc2Vec([skip_gram_model, dm_model])
mixed_model_with_concat = ConcatenatedDoc2Vec([skip_gram_model, dm_model_concat])

doc_vectors = []
for ind_ in order_level_two_month.index:
    doc_vectors.append(
        mixed_model_with_concat.docvecs[ind_]
    )

doc_vec_df = pd.DataFrame(doc_vectors)

doc_vec_df.columns = [f"custvec_{col}" for col in range(100)]
order_level_two_month = pd.concat([order_level_two_month, doc_vec_df], axis=1)
skip_gram_model.delete_temporary_training_data(False, False)  # free up memory
dm_model_concat.delete_temporary_training_data(False, False)  # free up memory
##########################


skip_gram_model = Doc2Vec(dm=0, vector_size=50, workers=cores, epochs=100, min_count=2, negative=5, hs=0, sample=0)
skip_gram_model.build_vocab(train_tagged_s.tolist())
skip_gram_model.train(train_tagged_s, total_examples=len(train_tagged_s), epochs=skip_gram_model.epochs)

dm_model_concat = Doc2Vec(dm=1, dm_concat=1, vector_size=50, workers=cores, epochs=100, min_count=2, negative=5, hs=0,
                          sample=0)
dm_model_concat.build_vocab(train_tagged_s.tolist())
dm_model_concat.train(train_tagged_s, total_examples=len(train_tagged_s), epochs=dm_model_concat.epochs)

# mixed_model = ConcatenatedDoc2Vec([skip_gram_model, dm_model])
mixed_model_with_concat = ConcatenatedDoc2Vec([skip_gram_model, dm_model_concat])

doc_vectors = []
for ind_ in order_level_two_month.index:
    doc_vectors.append(
        mixed_model_with_concat.docvecs[ind_]
    )

doc_vec_df = pd.DataFrame(doc_vectors)

doc_vec_df.columns = [f"stvec_{col}" for col in range(100)]
order_level_two_month = pd.concat([order_level_two_month, doc_vec_df], axis=1)

skip_gram_model.delete_temporary_training_data(False, False)  # free up memory
dm_model_concat.delete_temporary_training_data(False, False)  # free up memory
order_level_two_month.drop('index', axis=1, inplace=True)
order_level_two_month.to_parquet('/Users/saransh/Desktop/practicum/data/lagtwotrain.parquet', index=False)
