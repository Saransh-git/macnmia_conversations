from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel, CoherenceModel
from pandarallel import pandarallel
import nltk
import pyLDAvis.gensim

pandarallel.initialize()
order_lag_one = pd.read_parquet('/Users/saransh/Desktop/practicum/data/lagonetrain.parquet')


def str_to_tokens(str):
    return str.split()


order_lag_one['customer_processed_msg'] = order_lag_one.customer_processed_msg.parallel_apply(str_to_tokens)
order_lag_one['stylist_processed_msg'] = order_lag_one.stylist_processed_msg.parallel_apply(str_to_tokens)

stopword_dict = {}  # indexing for faster query
for word in nltk.corpus.stopwords.words('english'):
    stopword_dict[word] = 1


def is_stop_word(token: str) -> bool:
    try:
        stopword_dict[token]
    except KeyError:
        return False
    else:
        return True


def remove_stop_words(tokens):
    finalized_tokens = []
    for token in tokens:
        if is_stop_word(token):
            continue
        else:
            finalized_tokens.append(token)
    return finalized_tokens


order_lag_one['customer_processed_msg'] = order_lag_one.customer_processed_msg.parallel_apply(remove_stop_words)
order_lag_one['stylist_processed_msg'] = order_lag_one.stylist_processed_msg.parallel_apply(remove_stop_words)

dict_corp_c = corpora.Dictionary(order_lag_one['customer_processed_msg'].tolist())
corpus = [dict_corp_c.doc2bow(text) for text in order_lag_one['customer_processed_msg']]


dict_corp_s = corpora.Dictionary(order_lag_one['stylist_processed_msg'].tolist())
corpus_s = [dict_corp_s.doc2bow(text) for text in order_lag_one['stylist_processed_msg']]


def compute_coherence_c(num_topic):
    print(f"Computing for num topic {num_topic}")
    ldamodel_ = LdaModel(corpus, num_topics=num_topic, id2word=dict_corp_c, passes=15, random_state=60616)
    return CoherenceModel(model=ldamodel_, texts=order_lag_one['customer_processed_msg'].tolist(),
                          dictionary=dict_corp_c,
                          coherence='c_v').get_coherence()


executor = ThreadPoolExecutor(max_workers=4)
fts = []
coherence_score = []
for num_topic in range(2,10):
    fts.append(executor.submit(compute_coherence_c, num_topic))

executor.shutdown()
for ft in fts:
    coherence_score.append(ft.result())


def compute_coherence_s(num_topic):
    print(f"Computing for num topic {num_topic}")
    ldamodel_s = LdaModel(corpus_s, num_topics=num_topic, id2word=dict_corp_s, passes=15)
    return CoherenceModel(model=ldamodel_s, texts=order_lag_one['stylist_processed_msg'].tolist(),
                          dictionary=dict_corp_s,
                          coherence='c_v').get_coherence()


executor = ThreadPoolExecutor(max_workers=4)
coherence_score_s = []
fts = []
for num_topic in range(2,10):
    fts.append(executor.submit(compute_coherence_s, num_topic))

executor.shutdown()
for ft in fts:
    coherence_score_s.append(ft.result())


ldamodel_ = LdaModel(corpus, num_topics=4, id2word=dict_corp_c, passes=15, random_state=60616)
ldamodel_s = LdaModel(corpus_s, num_topics=5, id2word=dict_corp_s, passes=15)


def assign_topic_from_lda(tokens, dict_corpus, lda_model):
    #
    def sort_by_max_topic(item):
        return item[1]
    topics_ = lda_model.get_document_topics(
        dict_corpus.doc2bow(tokens)
    )
    return sorted(topics_, key=sort_by_max_topic, reverse=True)[0][0]


order_lag_one['customer_topic'] = order_lag_one['customer_processed_msg'].parallel_apply(
    partial(assign_topic_from_lda, dict_corpus=dict_corp_c, lda_model=ldamodel_))
order_lag_one['stylist_topic'] = order_lag_one['stylist_processed_msg'].parallel_apply(
    partial(assign_topic_from_lda, dict_corpus=dict_corp_s, lda_model=ldamodel_s)
)

order_lag_one.to_parquet("/Users/saransh/Desktop/practicum/data/lagonetrain_l.parquet", index=False)
lda_display = pyLDAvis.gensim.prepare(ldamodel_, corpus, dict_corp_c, sort_topics=False)
lda_cust = pyLDAvis.display(lda_display)

lda_display_s = pyLDAvis.gensim.prepare(ldamodel_s, corpus_s, dict_corp_s, sort_topics=False)
lda_stylist = pyLDAvis.display(lda_display_s)

with open("/Users/saransh/Desktop/practicum/data/lda_cust.html") as f:
    f.write(lda_cust.data)

with open("/Users/saransh/Desktop/practicum/data/lda_stylist.html") as f:
    f.write(lda_stylist.data)
