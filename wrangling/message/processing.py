import re
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
import numpy as np
import nltk
from matplotlib.axes import Axes
from nltk import word_tokenize, Text, BigramCollocationFinder, TrigramCollocationFinder, FreqDist
from pandas import DataFrame
from matplotlib import pyplot as plt


pandarallel.initialize()
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


# msg_data = msg_data[~msg_data['stylist_id'].isna()] TODO: clean it afterwards


def strip_and_tokenize(msg):
    msg = msg.strip()
    msg = msg.lower()
    # remove stop words
    # remove non-ascii characters
    tokens = word_tokenize(msg)
    finalized_tokens = []
    for token in tokens:
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            pass  # reomove non-ascii tokens
        else:
            finalized_tokens.append(token)
    return finalized_tokens


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


print("Strip/tokenize/remove non-ascii tokens...")
msg_data['tokenized_msg'] = msg_data['msg'].parallel_apply(strip_and_tokenize)
print("remove non price numbers...")
msg_data['tokenized_msg'] = msg_data['tokenized_msg'].parallel_apply(remove_non_price_numbers)
corpus = []
customer_corpus = []
stylist_corpus = []


def construct_msg_corpus(tokenized_msg):
    corpus.extend(tokenized_msg)


msg_data['tokenized_msg'].apply(construct_msg_corpus)
messaging_text = Text(corpus, name="whole corpus")
vocab = messaging_text.vocab()
sorted_vocab_keys = sorted(vocab, key=vocab.get, reverse=True)
vocab.plot(50)  # demonstrates that the stop words are heavily frequent and needs to be removed.
sorted_vocab = FreqDist()
for key in sorted_vocab_keys:
    sorted_vocab[key] = vocab[key]



def remove_hapaxes(tokens):
    hapax_dict = {hapax: 1 for hapax in vocab.hapaxes()}
    finalized_tokens = []
    for token in tokens:
        if vocab.get(token) >= 50000 or vocab.get(token) < 5:
            tokens.remove(token)



print(f"Total number of tokens: {len(corpus)}")
print(f"Total number of types: {len(vocab)}")
print(f"Lexical density: {len(messaging_text)/ len(vocab)}")
print(vocab.most_common(50))  # most common are the stop words and need to be removed.
print(vocab.most_common(200))
print(len(vocab.hapaxes()))  # around 30228 out of 67533 tokens just occur once


frequency_num_words_df = pd.DataFrame.from_records(list(vocab.r_Nr().items()), columns=['frequency', 'num_words'])
frequency_num_words_df.sort_values("frequency", inplace=True)
frequency_num_words_df['cumulative_num_words'] = frequency_num_words_df['num_words'].cumsum()

# Overall plot for frequency vs num words in samples
plt.figure()
axs: Axes = plt.gca()

axs.plot(frequency_num_words_df['frequency'], frequency_num_words_df['num_words'], 'bo--')
axs.set_xlabel("Frequency")
axs.set_ylabel("Number of words having that frequency")
axs.set_axisbelow(True)
axs.grid(True)
plt.show()

# Plot for frequency vs num words with frequency ranging from 1 to 15
plt.figure()
axs: Axes = plt.gca()
axs.plot(frequency_num_words_df['frequency'][1:], frequency_num_words_df['num_words'][1:], 'bo--')
axs.set_xlim(0.5, 15.5)
axs.set_xlabel("Frequency")
axs.set_ylabel("Number of words having that frequency")
axs.set_xticks(range(1, 16))
axs.set_yticks(range(0, 31000, 1000))
axs.set_axisbelow(True)
axs.grid(True)
plt.show()

# Plot for frequency vs cumulative num words with frequency ranging from 1 to 15
plt.figure()
axs: Axes = plt.gca()
axs.plot(frequency_num_words_df['frequency'], frequency_num_words_df['cumulative_num_words'], 'bo--')
axs.set_xlim(0.5, 15.5)
axs.set_xlabel("Frequency")
axs.set_ylabel("Cumulative number of words")
axs.set_xticks(range(1, 16))
axs.set_ylim(29000, 60000)
axs.set_axisbelow(True)
axs.grid(True)
plt.show()   # Almost 49962 words out of 67533 just have a frequency less than or equal to 4.


# plot r vs nr, decide on a cutoff based on the graph
# plot hapaxes

bins = [0,5,11,100,1000,5000,20000, 50000, 1000000, 2000000]
frequency_num_words_df['freq_bins'] = pd.cut(
    frequency_num_words_df['frequency'], bins, right=False, include_lowest=True
)


# Bin wise plot for inspecting frequency to number of word samples in that frequency
frequency_num_words_df['freq_bins'] = frequency_num_words_df['freq_bins'].astype(np.object)
pivot_by_freq = pd.pivot_table(frequency_num_words_df, values='num_words', index='freq_bins', aggfunc=np.sum)
# pivot_by_freq = pivot_by_freq.reset_index()
ret_axes: Axes = pivot_by_freq.plot(kind='barh')
ret_axes.plot(pivot_by_freq['num_words'], list(ret_axes.get_yticks()))
ret_axes.get_legend().remove()
ret_axes.set_xlabel("Bin wise cumulative word numbers")
ret_axes.set_title("Whole corpus")
ret_axes.set_axisbelow(True)
ret_axes.grid(True)
plt.show()

########## Inspect by word length ############
len_counter = FreqDist()
len_to_freq_dict = {}
for key in vocab:
    len_counter[len(key)] = len_counter.get(len(key), 0) + 1
    len_to_freq_dict[len(key)] = len_to_freq_dict.get(len(key), 0) + vocab[key]


len_df = pd.DataFrame.from_records(list(len_counter.items()), columns=['word_length', 'num_words'])
len_df.sort_values('word_length', inplace=True)
len_df = len_df.merge(
    pd.DataFrame.from_records(
        list(len_to_freq_dict.items()), columns=['word_length', 'frequency']
    ), on='word_length', how='inner', sort=True
)

ret_axes: Axes = len_df.plot(x='word_length', y='num_words', marker='o', color='blue', linestyle='--')
ret_axes.set_xlabel("Word length")
ret_axes.set_ylabel("Number of words")
ret_axes.set_title("Whole corpus")
ret_axes.set_axisbelow(True)
ret_axes.grid(True)
plt.show()

word_len_bins = [0, 3, 9, 13, 20, 30, 50, 100, 600]
len_df['word_lenth_bins'] = pd.cut(len_df['word_length'], word_len_bins, right=False, include_lowest=True)
len_df['word_lenth_bins'] = len_df['word_lenth_bins'].astype(np.object)
pivot_by_length = pd.pivot_table(len_df, values='num_words', index='word_lenth_bins', aggfunc=np.sum)
ret_axes: Axes = pivot_by_length.plot(kind='barh')
ret_axes.plot(pivot_by_length['num_words'], list(ret_axes.get_yticks()))
ret_axes.set_xlabel("Bin wise cumulative word numbers by length")
ret_axes.set_title("Whole corpus")
ret_axes.get_legend().remove()
ret_axes.set_axisbelow(True)
ret_axes.grid(True)
plt.show()


plt.figure()
axs = plt.gca()
axs.plot(len_df['word_length'], len_df['frequency'], 'bo--')
axs.set_xlabel("Word length")
axs.set_ylabel("Frequency")
axs.set_title("Whole corpus")
axs.set_xlim(0,10.5)
plt.show()


plt.figure()
axs = plt.gca()
axs.plot(len_df['word_length'], len_df['frequency'], 'bo--')
axs.set_xlabel("Word length")
axs.set_ylabel("Frequency")
axs.set_title("Whole corpus")
axs.set_xlim(10.5, 20.5)
axs.set_ylim(0, 120000)
plt.show()


plt.figure()
axs = plt.gca()
axs.plot(len_df['word_length'], len_df['frequency'], 'bo--')
axs.set_xlabel("Word length")
axs.set_ylabel("Frequency")
axs.set_title("Whole corpus")
axs.set_xlim(20.5, 36.5)
axs.set_xticks([20, 25, 30, 35, 40])
axs.set_ylim(0, 35000)
plt.show()  # peaks at word length 29, 32 and 33 (much larger than the below peaks)

plt.figure()
axs = plt.gca()
axs.plot(len_df['word_length'], len_df['frequency'], 'bo--')
axs.set_xlabel("Word length")
axs.set_ylabel("Frequency")
axs.set_title("Whole corpus")
axs.set_xlim(36.5)
axs.set_xticks([37, 50, 100, 200, 400])
axs.set_ylim(0, 700)
plt.show()  # at word length 42: 329, 43:469, 64: 679


pivot_by_length_frequency = pd.pivot_table(len_df, values='frequency', index='word_lenth_bins', aggfunc=np.sum)
ret_axes: Axes = pivot_by_length_frequency.plot(kind='barh')
ret_axes.plot(pivot_by_length_frequency['frequency'], list(ret_axes.get_yticks()))
ret_axes.set_xlabel("Bin wise cumulative frequencies by word length")
ret_axes.set_title("Whole corpus")
ret_axes.get_legend().remove()
ret_axes.set_axisbelow(True)
ret_axes.grid(True)
plt.show()

ret_axes: Axes = pivot_by_length_frequency[3:].plot(kind='barh')
ret_axes.plot(pivot_by_length_frequency['frequency'].iloc[3:], list(ret_axes.get_yticks()))
ret_axes.set_xlabel("Bin wise cumulative frequencies by word length")
ret_axes.set_title("Whole corpus")
ret_axes.get_legend().remove()
ret_axes.set_axisbelow(True)
ret_axes.grid(True)
plt.show()


bigrams = BigramCollocationFinder.from_words(corpus)
trigrams = TrigramCollocationFinder.from_words(corpus)



phone_nums = {}


def is_phone_number(phone_str: str) -> bool:
    phone_regex_1 = r'(?:\+1(?P<delim>(\.|-)?))([0-9]{3})(?P=delim)([0-9]{3})(?P=delim)([0-9]{4})'
    phone_regex_2 = r'([0-9]{3})(?P<delim>(\.|-)?)([0-9]{3})(?P=delim)([0-9]{4})'
    if phone_str.startswith('+1'):
        match = re.fullmatch(phone_regex_1, phone_str)
        if not match:
            return False
        num = '-'.join(match.groups()[2:])
        phone_nums[num] = phone_nums.get(num, 0) + 1
    else:
        match = re.fullmatch(phone_regex_2, phone_str)
        if not match:
            return False
        num = f"{match.groups()[0]}-" + '-'.join(match.groups()[3:])
        phone_nums[num] = phone_nums.get(num, 0) + 1
    return True


for token in corpus:
    if is_phone_number(token):
        pass

phone_nums = {key: phone_nums[key]  for key in sorted(phone_nums, key=phone_nums.get, reverse=True)}


url_pattern = r'/{0,2}((\S+?\.)*[@-_a-zA-Z0-9]+(\.com|\.me|\.it|\.org|\.net|\.co|\.be|\.gl|\.in))(/\S+)*'  # let's use this
# stupid url pattern
url_regex = re.compile(url_pattern)
urls = {}
url_grps = {}


def is_url(token: str):
    return url_regex.fullmatch(token)


for token in vocab:
    match_obj = is_url(token)
    if match_obj is not None:
        grp_0 = match_obj.groups()[0]
        urls[grp_0] = urls.get(grp_0, 0) + vocab[token]
        url_grps[grp_0] = url_grps.get(grp_0, [])
        url_grps[grp_0].append(token)

urls = {key: urls[key] for key in sorted(urls, key=urls.get, reverse=True)}
#url_pattern_1 = r'.+\.[a-zA-Z]+'
# Number of messages by time (months)
# Number of messages by customer and sender over time (months)
# There's a peak at word length 17, what is it?
# What are the urls contained in messages? What's the time over which they were included?
# Remove phone numbers and dates, look at their frequency, if something looks evident, then explore it.
# Explore how many orders have no conversations within it and last purchase
# try to find the root of urls and their frequency
# start cleaning up text
# associate the numbers and the urls to the transaction data
# lemmatize, remove punctuations, frequent words
# Number of child to difference in order purchase
# Number of items kept to difference in order purchase
# Difference between order request data and order created date
# Explore the difference between ship and settle date


# Observations:- https://macmia.co/2YURYeB  (What are your summertime needs? promotion)
#  'www.joulesusa.com': 6, 'shop.nordstrom.com': 6, mayoral.com: 5, 'www.zara.com': 5, 'www.amazon.com',
# 'www.hatley.com': 5, https://fawnshoppe.com/, www.jcrew.com, www.bodenusa.com, kidbox.com,




