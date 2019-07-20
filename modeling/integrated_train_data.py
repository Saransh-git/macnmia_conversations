from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from wrangling.utils import convert_to_python_datetime, convert_datetime_cols_to_python_datetime, \
    convert_timedelata_to_days, filter_based_on_ranges

pandarallel.initialize()
train_data = pd.read_csv('/Users/saransh/Desktop/practicum/data/user_order_train.csv')
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


def str_to_tokens(token_str):
    return token_str.split()


msg_data['tokenized_msg'] = msg_data.tokenized_msg.parallel_apply(str_to_tokens)
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

last_orders = train_data[train_data.order_lag.isna()]
train_data_except_last_orders = filter_based_on_ranges(
    train_data[~train_data.orders_id.isin(last_orders.orders_id)], 'order_lag', upper_bound=400
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


def concatenate_msgs_per_order(grp):
    club_tokens = pd.DataFrame()
    orders_id, sender = grp.name
    print(orders_id, sender)
    tokens_ = []
    for conv in grp['tokenized_msg']:
        tokens_.extend(conv)
    cols = ['shipped_price', 'kept_price', 'order_lag', 'items_count', 'box_price', 'kept_count', 'order_rank',
            'num_child', 'request_ship_diff', 'keep_rate']
    club_tokens = club_tokens.append(grp[cols].iloc[0])
    club_tokens['tokenized_msg'] = ' '.join(tokens_)
    club_tokens['orders_id'] = orders_id
    club_tokens['sender'] = sender
    return club_tokens


clubbed_one_month = one_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
clubbed_two_month = two_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
clubbed_three_month = three_month_lag.groupby(['orders_id', 'sender']).apply(concatenate_msgs_per_order)
