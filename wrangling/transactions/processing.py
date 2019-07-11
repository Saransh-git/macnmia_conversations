import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandarallel import pandarallel
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from scipy.interpolate import make_interp_spline

from wrangling.utils import convert_datetime_cols_to_python_datetime, convert_timedelata_to_days, filter_based_on_ranges

pandarallel.initialize()
transactions_field_mapping = {
    'Orders Order ID': 'orders_id',
    'Assigned Stylists Stylist ID': 'stylist_id',
    'User User ID': 'user_id',
    'Children Child ID': 'child_id',
    'Requests Request Date': 'request_date',
    'Orders Order Created Date': 'created_date',
    'Orders Ship Date': 'ship_date',
    'Orders Settle Date': 'settle_date',
    'Order Items Total Shipped Price': 'shipped_price',
    'Order Items Total Selling Price': 'kept_price',
    'Order Items Return Reason': 'return_reason',
    'Order Items Return Comments': 'return_comments'
}


def load_transactions_data(data_path: str = '/Users/saransh/Desktop/practicum/data/Full data'):
    transactions = None
    path = Path(data_path)
    if not path.is_dir():
        raise ValueError

    for file in path.iterdir():
        if 'Transactions' in file.as_posix():
            dd = pd.read_csv(str(file))
            if transactions is None:
                transactions = dd
            else:
                transactions = pd.concat([dd, transactions], ignore_index=True)

    transactions.rename(transactions_field_mapping, inplace=True, axis=1)
    transactions = transactions[transactions_field_mapping.values()]
    return transactions


transactions = convert_datetime_cols_to_python_datetime(
    load_transactions_data()
)


def num_items_kept(grouped_df: DataFrame):
    kept_price = grouped_df['kept_price'][grouped_df['kept_price'] != 0]
    return kept_price.count()


def price_items_kept(grouped_df: DataFrame):
    kept_price = grouped_df['kept_price'][grouped_df['kept_price'] != 0]
    return kept_price.sum()


grouped_transactions_by_user = transactions.groupby('user_id')
child_cts_by_user = grouped_transactions_by_user.child_id.nunique()
user_and_order_level_data = pd.DataFrame()
usr_ct = 1
for user, grp in grouped_transactions_by_user:
    print(usr_ct)
    usr_ct += 1
    grp = grp.sort_values('request_date')
    grp_by_order: DataFrameGroupBy = grp.groupby(['request_date', 'orders_id'])
    items_count = grp_by_order['shipped_price'].count()
    items_kept = grp_by_order.apply(num_items_kept)
    items_kept_price = grp_by_order.apply(price_items_kept)
    box_price = grp_by_order['shipped_price'].sum()
    grp_by_order = grp_by_order.first().reset_index()
    num_child = [child_cts_by_user[user]] * grp_by_order.shape[0]
    grp_by_order['order_lag'] = grp_by_order['request_date'].diff()
    grp_by_order['items_count'] = items_count.tolist()
    grp_by_order['box_price'] = box_price.tolist()
    grp_by_order['kept_count'] = items_kept.tolist()
    grp_by_order['kept_price'] = items_kept_price.tolist()
    grp_by_order['order_rank'] = grp_by_order['request_date'].rank(method='first', ascending=1)
    grp_by_order['num_child'] = num_child
    user_and_order_level_data = user_and_order_level_data.append(grp_by_order, ignore_index=True)

user_and_order_level_data['order_lag'] = user_and_order_level_data['order_lag'].parallel_apply(
    convert_timedelata_to_days
)

user_and_order_level_data.to_csv('/Users/saransh/Desktop/practicum/data/user_order.csv', index=False)

user_and_order_level_data = filter_based_on_ranges(
    user_and_order_level_data, col_name='items_count', lower_bound=5, upper_bound=15
)

user_and_order_level_data = filter_based_on_ranges(user_and_order_level_data, 'order_rank', upper_bound=17)

order_rank_lag = user_and_order_level_data.groupby('order_rank')['order_lag'].median().reset_index()

plt.figure()
axs: Axes = plt.gca()
order_rank_list = [1] + order_rank_lag.order_rank[1:20].tolist()
axs.plot(order_rank_list, [0] + order_rank_lag.order_lag[1:20].tolist(), 'yo--')
xnew = np.linspace(min(order_rank_list), max(order_rank_list), 300)
spl = make_interp_spline(order_rank_list, [0] + order_rank_lag.order_lag[1:20].tolist(), k=3)
power_smooth = spl(xnew)
axs.plot(xnew, power_smooth, 'b-')
axs.set_xlabel("Purchase times")
axs.set_ylabel("Days between subsequent orders")
axs.set_xticks(order_rank_list)
plt.show()

order_rank_user_grp_ct = user_and_order_level_data.groupby('order_rank')['user_id'].count()

conversions = []
for i in range(1, 20):
    conversions.append(
        order_rank_user_grp_ct.iloc[i] / order_rank_user_grp_ct.iloc[i - 1]
    )

conversions = [np.nan] + conversions
order_rank_user_grp_ct = order_rank_user_grp_ct.reset_index()[:20]
order_rank_user_grp_ct['conversion_rate'] = conversions

plt.figure()
axs: Axes = plt.gca()
axs.plot(order_rank_user_grp_ct.order_rank, order_rank_user_grp_ct.conversion_rate, 'go-')
axs.set_xlabel("Purchase times")
axs.set_ylabel("Conversion rate")
axs.set_xticks(list(range(2, 21)))
plt.show()

bins = range(0, 801, 50)
user_and_order_level_data['box_price_range'] = pd.cut(
    user_and_order_level_data.box_price, bins, right=False, include_lowest=True
)

axs = user_and_order_level_data.boxplot('order_rank', by='num_child', vert=False, notch=True, bootstrap=1000)
axs.set_xlabel('')
axs.set_ylabel('Total child count')
plt.show()

axs = user_and_order_level_data.loc[user_and_order_level_data.order_lag <= 400].boxplot(  # consider only lag
    # within 400 days
    'order_lag', by='order_rank', vert=False, notch=True, bootstrap=1000
)
axs.set_xlabel('')
axs.set_ylabel('Order rank')
plt.show()

axs = user_and_order_level_data.boxplot(
    'order_lag', by='order_rank', vert=False, notch=True, bootstrap=1000
)
axs.set_xlabel('')
axs.set_ylabel('Order rank')
plt.show()

bins = range(0, user_and_order_level_data.items_count.max() + 1, 1)
axs: Axes = user_and_order_level_data.kept_count.plot(kind='hist', bins=bins)
axs.set_xticks(list(bins))
axs.set_xlabel("Item keep count")
axs.set_ylabel("Frequency")
plt.show()

