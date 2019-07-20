import re
import string
from pathlib import Path

import enchant
import pandas as pd
from pandarallel import pandarallel
import nltk
from nltk import word_tokenize, Text, BigramCollocationFinder, RegexpTokenizer
from nltk.corpus.reader.wordnet import POS_LIST
from pandas import DataFrame
from nltk import WordNetLemmatizer
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

pandarallel.initialize()
stopword_dict = {}  # indexing for faster query
english_us_dict = enchant.Dict("en_US")
for word in nltk.corpus.stopwords.words('english'):
    stopword_dict[word] = 1

regex_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

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


# msg_data = msg_data[~msg_data['stylist_id'].isna()] TODO: clean it afterwards

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


def is_url(token: str):
    # macmia.me
    # /pricing
    # app.macandmia.com
    # .com
    # /pin.it/bqs46f52mvgal3o4
    # hire.lever.co
    # macmia.co/kylebstylist
    return url_regex.fullmatch(token)


def is_remove_url(token: str) -> bool:
    urls_to_keep = ['/add-billing']
    if is_url(token):
        for u in urls_to_keep:
            if u in token:
                return False
            return True
    return False


def is_stop_word(token: str) -> bool:
    try:
        stopword_dict[token]
    except KeyError:
        return False
    else:
        return True


def is_english_word(token: str) -> bool:
    return english_us_dict.check(token)


def strip_and_tokenize(msg):
    msg = msg.strip()
    msg = msg.lower()
    # remove stop words
    # remove non-ascii characters
    tokens = word_tokenize(msg)
    tokens_ = tokens.copy()
    for token in tokens_:
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            tokens.remove(token)  # remove non-ascii tokens
            continue
        #
        if is_stop_word(token):  # remove stopwords
            tokens.remove(token)
            continue
        if token in punctuations:  # remove punctuations
            tokens.remove(token)
            continue
        if len(token) <= 2:  # drop the words below length of 2
            tokens.remove(token)
            continue
        if is_date(token) or is_phone_number(token):
            tokens.remove(token)
            continue
        remove_url = is_remove_url(token)
        if remove_url:
            tokens.remove(token)
            continue
        if len(token) >= 30:  # drop the words above length of 30, retain to be kept urls
            if bool(is_url(token)):
                continue
            else:
                tokens.remove(token)
                continue
        elif 20 <= len(token) < 30:
            if bool(is_url(token)):
                continue
            else:
                tokens.extend(regex_tokenizer.tokenize(token))
    return tokens


def remove_non_price_numbers(tokens):
    """
    Remove numbers which are not prices
    """
    finalized_tokens = []
    token_bigrams = BigramCollocationFinder.from_words(tokens)
    num_pattern = r"\d+(?:\.\d+)?"
    pattern = re.compile(f"{num_pattern}(-{num_pattern})?")  # include tokens like ranges of numbers/ numbers
    for token in tokens:
        if not pattern.fullmatch(token):
            finalized_tokens.append(token)
            continue
        if (token_bigrams.ngram_fd.get(('$', token)) or token_bigrams.ngram_fd.get((token, '$'))) \
                or (token_bigrams.ngram_fd.get(('usd', token)) or token_bigrams.ngram_fd.get((token, 'usd'))):
            finalized_tokens.append(token)
    return finalized_tokens


def remove_non_english_tokens(tokens):
    tokens_ = tokens.copy()
    tokens_removed = 0
    for index, token in enumerate(tokens_):
        if token == "n't":
            tokens.insert(index - tokens_removed, 'not')
        if not is_english_word(token):
            tokens.remove(token)
            tokens_removed += 1
    return tokens


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


msg_data['tokenized_msg'] = msg_data['msg'].parallel_apply(strip_and_tokenize)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_price_numbers)

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


msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_frequent_and_rare_tokens)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_english_tokens)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(lematize_tokens)
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_informatic_pos)
obs_to_keep = msg_data['tokenized_msg'].apply(lambda tokens: len(tokens) > 0)
msg_data = msg_data[obs_to_keep]
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].apply(lambda tokens: " ".join(tokens))
msg_data.reset_index(drop=True).to_parquet('/Users/saransh/Desktop/practicum/data/Messaging.parquet', index=False)
