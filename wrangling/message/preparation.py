import re
import string
from pathlib import Path

import enchant
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk import word_tokenize, Text, BigramCollocationFinder, RegexpTokenizer
from nltk.corpus.reader.wordnet import POS_LIST
from pandarallel import pandarallel
from pandas import DataFrame
from typing import Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.corpora.dictionary import Dictionary


pandarallel.initialize()
sentiment_analyzer = SentimentIntensityAnalyzer()
stopword_dict = {}  # indexing for faster query
english_us_dict = enchant.Dict("en_US")
for word in nltk.corpus.stopwords.words('english'):
    stopword_dict[word] = 1

regex_tokenizer = RegexpTokenizer(r'[0-9a-zA-Z]+')

field_name_mapping = {
    'Users User ID': 'user_id',
    ' CHATting Stylists Stylist ID': 'stylist_id',
    'CHAT Messages Created Time': 'created_at',
    'CHAT Messages Sender': 'sender',
    'CHAT Messages Message Body': 'msg'
}

file_path = Path('/Users/saransh/Desktop/practicum/data/Messaging Data all time.csv')
file_path = file_path.expanduser()
if not file_path.exists():
    raise FileNotFoundError

msg_data: DataFrame = pd.read_csv(
    file_path.as_posix()
)
msg_data.rename(field_name_mapping, inplace=True, axis=1)
msg_data = msg_data[~msg_data['user_id'].isna()]  # 663 rows
msg_data = msg_data[~msg_data['msg'].isna()]  # 201117 rows
msg_data = msg_data[~msg_data['created_at'].isna()]
msg_data = msg_data[~msg_data['stylist_id'].isna()]
msg_data.sender[msg_data.sender == 'unknown'] = 'stylist'  # from data inspection
punctuations = string.punctuation


date_pattern = r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{1,4}'
date_regex = re.compile(date_pattern)


def is_date(token: str) -> bool:
    # 09/04/2018
    if date_regex.fullmatch(token):
        return True
    return False


# phone_nums = {}
# phone_nums = {key: phone_nums[key]  for key in sorted(phone_nums, key=phone_nums.get, reverse=True)}

def is_phone_number(phone_str: str) -> bool:
    phone_regex_1 = r'(?:\+1(?P<delim>(\.|-)?))([0-9]{3})(?P=delim)([0-9]{3})(?P=delim)([0-9]{4})'
    phone_regex_2 = r'([0-9]{3})(?P<delim>(\.|-)?)([0-9]{3})(?P=delim)([0-9]{4})'
    if phone_str.startswith('+1'):
        match = re.fullmatch(phone_regex_1, phone_str)
        if not match:
            return False
        num = '-'.join(match.groups()[2:])
        # phone_nums[num] = phone_nums.get(num, 0) + 1
    else:
        match = re.fullmatch(phone_regex_2, phone_str)
        if not match:
            return False
        num = f"{match.groups()[0]}-" + '-'.join(match.groups()[3:])
        # phone_nums[num] = phone_nums.get(num, 0) + 1
    return True


url_pattern = r'/{0,2}(?:www\.)?' \
              r'(((?:[^(?:\s)|(?:www)]+?\.)*[@-_a-zA-Z0-9]+' \
              r'(?:\.com|\.me|\.it|\.org|\.net|\.co|\.be|\.gl|\.in))(?:/[^(\s)/]+)*)/?'  # let's use this
# stupid url pattern
url_regex = re.compile(url_pattern)
short_age_pattern = re.compile("\d{1,2}([ty]|mo)")


def analyze_sentiment(msg_data):
    sent_dict = sentiment_analyzer.polarity_scores(msg_data['msg'])
    msg_data['pos_sent'] = sent_dict['pos']
    msg_data['neg_sent'] = sent_dict['neg']
    msg_data['neu_sent'] = sent_dict['neu']
    msg_data['sent_score'] = sent_dict['compound']
    return msg_data


def polarize_sentiments(sentiment_score: float):
    if sentiment_score >= 0.05:
        return 'pos'
    elif sentiment_score <= -0.05:
        return 'neg'
    else:
        return 'neu'


def is_url(token: str):
    # macmia.me
    # /pricing
    # app.macandmia.com
    # .com
    # /pin.it/bqs46f52mvgal3o4
    # hire.lever.co
    # macmia.co/kylebstylist
    return url_regex.fullmatch(token)


def retrieve_url_tokens(token: str) -> str:
    urls_to_keep = {
        '/add-billing': '<BILL_URL>'
    }
    if is_url(token):
        for u in urls_to_keep:
            if u in token:
                return urls_to_keep[u]
        return "<URL>"
    return token


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


def is_english_word(token: str) -> bool:
    return english_us_dict.check(token)


def strip_and_tokenize(msg):
    msg = msg.strip()
    msg = msg.lower()
    msg = re.sub(b'\xe2\x80\x99'.decode(), "'", msg)
    # remove stop words
    # remove non-ascii characters
    tokens = word_tokenize(msg)
    # tokens_ = tokens.copy()
    ret_tokens = []
    token_bigrams = BigramCollocationFinder.from_words(tokens)
    for token in tokens:
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            # tokens.remove(token)  # remove non-ascii tokens
            continue
        #
        if token in punctuations:  # remove punctuations
            # tokens.remove(token)
            continue
        #
        if is_date(token):
            ret_tokens.append("<DATE>")
            continue
        if is_phone_number(token):
            # tokens.remove(token)
            ret_tokens.append("<PHONE>")
            continue
        #
        if bool(is_url(token)):
            ret_tokens.append(retrieve_url_tokens(token))
            continue
        new_token, is_resolved = resolve_clitics(token)
        if is_resolved:
            ret_tokens.append(new_token)
            continue
        p_token, is_tokenized = tokenize_price_numbers(token, token_bigrams)
        if is_tokenized:
            ret_tokens.append(p_token)
            continue
        ret_tokens.extend(regex_tokenizer.tokenize(token))
        #
    return ret_tokens


def tokenize_price_numbers(token, token_bigrams):
    """
    Remove numbers which are not prices
    """
    finalized_tokens = []
    num_pattern = r"\d+(?:\.\d+)?"
    pattern = re.compile(f"{num_pattern}(-{num_pattern})?")  # include tokens like ranges of numbers/ numbers
    if not pattern.fullmatch(token):
        return token, False
    if (token_bigrams.ngram_fd.get(('$', token)) or token_bigrams.ngram_fd.get((token, '$'))) \
            or (token_bigrams.ngram_fd.get(('usd', token)) or token_bigrams.ngram_fd.get((token, 'usd'))):
        return "<PRICE>", True
    else:
        return token, False


def tokenize_age_tokens(tokens) -> bool:
    age_pattern = re.compile(r"\d{1,2}")
    finalized_tokens = []
    token_bigrams = BigramCollocationFinder.from_words(tokens)
    for token in tokens:
        if short_age_pattern.fullmatch(token):
            finalized_tokens.append("<AGE>")
            continue
        if not age_pattern.fullmatch(token):
            finalized_tokens.append(token)
            continue
        if token_bigrams.ngram_fd.get((token, 'year')) or token_bigrams.ngram_fd.get((token, 'years')):
            finalized_tokens.append("<AGE>")
            continue
        elif token_bigrams.ngram_fd.get((token, 'month')) or token_bigrams.ngram_fd.get((token, 'months')):
            finalized_tokens.append("<AGE>")
            continue
        else:
            finalized_tokens.append(token)
    return finalized_tokens


def tokenize_numbers(tokens):
    finalized_tokens = []
    number_pattern = re.compile(r"\d+")
    for token in tokens:
        if number_pattern.fullmatch(token):
            finalized_tokens.append("<NUM>")
        else:
            finalized_tokens.append(token)
    return finalized_tokens


def resolve_clitics(token) -> Tuple[str, bool]:
    clitic_map = {
        "n't": "not",
        "'d": "would",
        "'s": "is",
        "'re": "are",
        "'m": "am"
    }
    if token in clitic_map:
        return clitic_map[token], True
    return token, False


def remove_non_english_tokens(tokens):
    finalized_tokens = []
    for token in tokens:
        if token in ["<AGE>", "<PRICE>", "<NUM>", "<URL>", "<DATE>", "<PHONE>", "<BILL_URL>"]:
            finalized_tokens.append(token)
            continue
        if not is_english_word(token):
            finalized_tokens.append("<UNK>")
            continue
        finalized_tokens.append(token)
    #
    return finalized_tokens


lemmatizer = WordNetLemmatizer()


def lemmatize(word):
    for pos in POS_LIST:
        lemmatized_token = lemmatizer.lemmatize(word, pos)
        if lemmatized_token != word:
            return lemmatized_token
    return word


def lematize_tokens(tokens):
    tokens_ = []
    for token in tokens:
        tokens_.append(lemmatize(token))
    return tokens_


def remove_non_informatic_pos(tokens):
    pos_tags = nltk.pos_tag(tokens)
    for token, tag in pos_tags:
        if tag in ['DT', 'NNP', 'NNPS', 'CC', 'IN', 'TO', 'WDT', 'WP', 'WP$', 'WRB']:
            tokens.remove(token)
    return tokens


msg_data = msg_data.parallel_apply(analyze_sentiment, axis=1)
msg_data['sentiments'] = msg_data['sent_score'].parallel_apply(polarize_sentiments)
msg_data['tokenized_msg'] = msg_data['msg'].parallel_apply(strip_and_tokenize)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(tokenize_age_tokens)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(tokenize_numbers)


corpus = []


def construct_msg_corpus(tokenized_msg):
    corpus.extend(tokenized_msg)


msg_data['tokenized_msg'].apply(construct_msg_corpus)
messaging_text = Text(corpus, name="whole corpus")
vocab = messaging_text.vocab()


def remove_frequent_and_rare_tokens(tokens):
    for token in tokens:
        if vocab.get(token) >= 50000 or vocab.get(token) < 5:
            tokens.remove(token)
    return tokens


msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_english_tokens)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].apply(lambda tokens: " ".join(tokens))
msg_data['processed_msg'] = msg_data['tokenized_msg'].apply(lambda msg: msg.split())  # This one for topic modeling
# msg_data['processed_msg'] = msg_data['processed_msg'].parallel_apply(lematize_tokens)
dict_corpus = Dictionary(msg_data['processed_msg'].tolist())
dict_corpus.filter_extremes(no_above=0.7)


def reconstruct_processed_msgs_from_corpora(msg_tokens):
    finalized_tokens = []
    ret_val = dict_corpus.doc2idx(msg_tokens)
    for token_id in ret_val:
        try:
            finalized_tokens.append(
                dict_corpus[token_id]
            )
        except KeyError:
            continue
    #
    return finalized_tokens


msg_data['processed_msg'] = msg_data['processed_msg'].parallel_apply(reconstruct_processed_msgs_from_corpora)
# msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_frequent_and_rare_tokens)
# msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_informatic_pos)
# obs_to_keep = msg_data['tokenized_msg'].apply(lambda tokens: len(tokens) > 0)
# msg_data = msg_data[obs_to_keep]
msg_data['processed_msg'] = msg_data['processed_msg'].apply(lambda tokens: " ".join(tokens))
msg_data.index = range(msg_data.shape[0])
msg_data.reset_index(drop=True).to_parquet('/Users/saransh/Desktop/practicum/data/Messaging.parquet', index=False)
