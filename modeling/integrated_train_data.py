from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from wrangling.utils import convert_to_python_datetime, convert_datetime_cols_to_python_datetime, \
    convert_timedelata_to_days, filter_based_on_ranges

pandarallel.initialize()
train_data = pd.read_csv('/Users/saransh/Desktop/practicum/data/order_clustered.csv')
train_data = train_data[~train_data.user_id.isna()]  # orders not corresponding to any user
train_data = train_data[~train_data.ship_date.isna()]  # orders not shipped
train_data = train_data[~train_data.stylist_id.isna()]  # no stylists assigned

transactions_type_mapping = dict(
    orders_id=np.int64,
    user_id=np.int64,
    stylist_id=np.int64,
    items_count=np.int64,
    kept_count=np.int64,
    order_rank=np.int64,
    num_child=np.int64
)

for col, type in transactions_type_mapping.items():
    train_data[col] = train_data[col].astype(type)  # transform dataframe cols to appropriate types

datetime_conversion_func = partial(convert_to_python_datetime, format_str='%Y-%m-%d')
convert_datetime_cols_to_python_datetime(train_data, conversion_func=datetime_conversion_func)
train_data['request_ship_diff'] = train_data.ship_date - train_data.request_date
train_data['keep_rate'] = train_data['kept_count']/train_data['items_count']
train_data['request_ship_diff'] = train_data['request_ship_diff'].apply(convert_timedelata_to_days)


msg_data = pd.read_parquet("/Users/saransh/Desktop/practicum/data/Messaging.parquet")
doc_vec_columns = [f"docvec_{col}" for col in range(200)]

def str_to_tokens(token_str):
    return token_str.split()


def ndarray_to_list(token_arr):
    return token_arr.tolist()


msg_data['tokenized_msg'] = msg_data.tokenized_msg.parallel_apply(ndarray_to_list)
msg_data['processed_msg'] = msg_data.processed_msg.parallel_apply(str_to_tokens)


msg_types = dict(
    user_id=np.int64,
    stylist_id=np.int64
)

for col, type_ in msg_types.items():
    msg_data[col] = msg_data[col].astype(type_)

datetime_conversion_func = partial(convert_to_python_datetime, format_str="%m/%d/%Y %H:%M")
convert_datetime_cols_to_python_datetime(
    msg_data, cols=['created_at'], conversion_func=datetime_conversion_func
)

'''
last_orders = train_data[train_data.order_lag.isna()]
train_data_except_last_orders = filter_based_on_ranges(
    train_data[~train_data.orders_id.isin(last_orders.orders_id)], 'order_lag', upper_bound=400
)
train_data_except_last_orders.set_index(["user_id", "stylist_id"], inplace=True)
'''

train_data_except_last_orders = filter_based_on_ranges(
    train_data, 'order_lag', upper_bound=400
)
train_data_except_last_orders.set_index(["user_id", "stylist_id"], inplace=True)
msg_data = msg_data.set_index(["user_id", "stylist_id"])

msg_transaction_data = msg_data.join(train_data_except_last_orders, how='inner')  # Index based join quite efficient

'''
# Yet another way of joining but quite inefficient
msg_data_w_index = msg_data.reset_index()
msg_data_w_index['uid'] = msg_data_w_index['user_id']
msg_data_w_index['sid'] = msg_data_w_index['stylist_id']
train_data_w_index = train_data_except_last_orders.reset_index()
msg_grp = msg_data_w_index.groupby(['user_id', 'stylist_id'])


def match_order_lag_by_grp(grp):
    msgs = pd.DataFrame()
    # print(list(grp.columns))
    key = grp.iloc[0][['uid', 'sid']].tolist()
    print(key)
    transaction_data = train_data_w_index.loc[
        (train_data_w_index.user_id == key[0]) & (train_data_w_index.stylist_id == key[1])]
    if transaction_data.empty:
        return msgs
    for row in transaction_data.iterrows():
        t_delta = (row[1].request_date - grp.created_at).apply(lambda t: t.days)
        grp_ = grp.loc[(0 < t_delta) & (t_delta <= 30)]
        for col in train_data_w_index:
            grp_[col] = row[1][col]
        msgs = msgs.append(grp_)
    return msgs


dd = msg_grp.apply(match_order_lag_by_grp)
'''

msg_transaction_data['request_date'] = \
    msg_transaction_data['request_date'] + msg_transaction_data['order_lag'].apply(lambda t: timedelta(days=t))
# consider conversations for an order subsequent as order lag is for that subsequent order

msg_transaction_data['conversation_lag'] = (
        msg_transaction_data['request_date'] - msg_transaction_data['created_at']
).apply(lambda t: t.days)

one_month_lag = msg_transaction_data.loc[
    (msg_transaction_data['conversation_lag'] >= 0) & (msg_transaction_data['conversation_lag'] <= 30)
    ]
two_month_lag = msg_transaction_data.loc[
    (msg_transaction_data['conversation_lag'] >= 0) & (msg_transaction_data['conversation_lag'] <= 60)
    ]
three_month_lag = msg_transaction_data.loc[
    (msg_transaction_data['conversation_lag'] >= 0) & (msg_transaction_data['conversation_lag'] <= 90)
    ]


one_month_lag.reset_index(inplace=True)
two_month_lag.reset_index(inplace=True)
three_month_lag.reset_index(inplace=True)

one_month_lag.drop(doc_vec_columns, axis=1, inplace=True)
two_month_lag.drop(doc_vec_columns, axis=1, inplace=True)
three_month_lag.drop(doc_vec_columns, axis=1, inplace=True)


def concatenate_msgs_per_order(grp):
    club_tokens = pd.DataFrame()
    orders_id, sender = grp.name
    print(orders_id, sender)
    tokenized_tokens_ = []
    processed_tokens_ = []
    for conv in grp['tokenized_msg']:
        tokenized_tokens_.extend(conv)
    #
    for conv in grp['processed_msg']:
        processed_tokens_.extend(conv)
    #
    mean_pos_sent = grp['pos_sent'].mean()
    mean_neg_sent = grp['neg_sent'].mean()
    mean_neu_sent = grp['neu_sent'].mean()
    mean_sentiment = grp['sent_score'].mean()
    msg_count = grp.shape[0]
    total_pos = grp[grp['sentiments'] == 'pos'].shape[0]
    total_neg = grp[grp['sentiments'] == 'neg'].shape[0]
    total_neu = grp[grp['sentiments'] == 'neu'].shape[0]
    #
    cols = ['shipped_price', 'kept_price', 'order_lag', 'items_count', 'box_price', 'kept_count', 'order_rank',
            'num_child', 'request_ship_diff', 'keep_rate', 'request_date', 'user_id', 'cl_labels']
    club_tokens = club_tokens.append(grp[cols].iloc[0])
    club_tokens['tokenized_msg'] = ' '.join(tokenized_tokens_)
    club_tokens['processed_msg'] = ' '.join(processed_tokens_)
    club_tokens['pos_sent'] = mean_pos_sent
    club_tokens['neg_sent'] = mean_neg_sent
    club_tokens['neu_sent'] = mean_neu_sent
    club_tokens['sent_score'] = mean_sentiment
    club_tokens['msg_count'] = msg_count
    club_tokens['total_pos'] = total_pos
    club_tokens['total_neg'] = total_neg
    club_tokens['total_neu'] = total_neu
    return club_tokens

[['msg_count', 'total_pos', 'total_neg', 'total_neu', 'sent_score', 'pos_sent', 'neg_sent', 'neu_sent']]
clubbed_one_month = one_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
clubbed_one_month.reset_index(inplace=True)

clubbed_two_month = two_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
clubbed_two_month.reset_index(inplace=True)

clubbed_three_month = three_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
clubbed_three_month.reset_index(inplace=True)


def produce_one_sample_per_order(grp):
    print(grp.name)
    temp = pd.DataFrame()
    cols = ['box_price', 'items_count',
            'keep_rate', 'kept_count', 'kept_price', 'num_child', 'order_lag',
            'order_rank', 'request_ship_diff', 'shipped_price', 'request_date', 'user_id', 'cl_labels']
    temp = temp.append(grp[cols].iloc[0])
    customer_msg_tok_ = ''
    customer_msg_proc = ''
    stylist_msg_tok_ = ''
    stylist_msg_proc_ = ''
    customer_row = grp.loc[grp['sender'] == 'customer']
    stylist_row = grp.loc[grp['sender'] == 'stylist']
    try:
        #
        customer_msg_tok_ = customer_row['tokenized_msg'].values[0]
        customer_msg_proc = customer_row['processed_msg'].values[0]
    except IndexError:
        pass
    try:
        stylist_msg_tok_ = stylist_row['tokenized_msg'].values[0]
        stylist_msg_proc_ = stylist_row['processed_msg'].values[0]
    except IndexError:
        pass
    #
    temp['customer_tokenized_msg'] = customer_msg_tok_
    temp['customer_processed_msg'] = customer_msg_proc
    temp['stylist_tokenized_msg'] = stylist_msg_tok_
    temp['stylist_processed_msg'] = stylist_msg_proc_
    #
    for col in ['pos_sent', 'neg_sent', 'neu_sent', 'sent_score', 'msg_count', 'total_pos', 'total_neg', 'total_neu']:
        temp[f"customer_{col}"] = customer_row[col].values[0] if not customer_row.empty else None
        temp[f"stylist_{col}"] = stylist_row[col].values[0] if not stylist_row.empty else None
    #
    return temp


clubbed_one_month.drop(['level_2'], axis=1, inplace=True)
clubbed_two_month.drop(['level_2'], axis=1, inplace=True)
clubbed_three_month.drop(['level_2'], axis=1, inplace=True)

order_level_one_month = clubbed_one_month.groupby('orders_id').apply(produce_one_sample_per_order)
order_level_two_month = clubbed_two_month.groupby('orders_id').apply(produce_one_sample_per_order)
order_level_three_month = clubbed_three_month.groupby('orders_id').apply(produce_one_sample_per_order)

order_level_one_month.reset_index(inplace=True)
order_level_two_month.reset_index(inplace=True)
order_level_three_month.reset_index(inplace=True)


order_level_one_month = order_level_one_month.loc[
    (order_level_one_month.customer_tokenized_msg != '') & (order_level_one_month.stylist_tokenized_msg != '')]


order_level_two_month = order_level_two_month[
    (order_level_two_month.customer_tokenized_msg != '') & (order_level_two_month.stylist_tokenized_msg != '')]


order_level_three_month = order_level_three_month[
    (order_level_three_month.customer_tokenized_msg != '') & (order_level_three_month.stylist_tokenized_msg != '')]

order_level_one_month.drop('level_1', axis=1, inplace=True)
order_level_two_month.drop('level_1', axis=1, inplace=True)
order_level_three_month.drop('level_1', axis=1, inplace=True)

order_level_one_month.index = range(order_level_one_month.shape[0])
order_level_two_month.index = range(order_level_two_month.shape[0])
order_level_three_month.index = range(order_level_three_month.shape[0])

order_level_one_month.to_parquet('/Users/saransh/Desktop/practicum/data/lagone.parquet', index=False)
order_level_two_month.to_parquet('/Users/saransh/Desktop/practicum/data/lagtwo.parquet', index=False)
order_level_three_month.to_parquet('/Users/saransh/Desktop/practicum/data/lagthree.parquet', index=False)
