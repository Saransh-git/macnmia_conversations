from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandarallel import pandarallel

from wrangling.transactions import convert_to_python_datetime, convert_datetime_cols_to_python_datetime, \
    filter_based_on_ranges

pandarallel.initialize()

user_and_order_level_data = pd.read_csv('/Users/saransh/Desktop/practicum/data/user_order.csv')


datetime_conversion_func = partial(convert_to_python_datetime, format_str='%Y-%m-%d')
convert_datetime_cols_to_python_datetime(user_and_order_level_data, conversion_func=datetime_conversion_func)

user_and_order_level_data = filter_based_on_ranges(user_and_order_level_data, lower_bound=5, upper_bound=17)

grouped_transactions_by_user = user_and_order_level_data.groupby('user_id')

train_data = pd.DataFrame()
usr_ct = 1
for user, grp in grouped_transactions_by_user:
    print(usr_ct)
    usr_ct += 1
    _grp = grp.sort_values('request_date')
    _grp = _grp.loc[_grp.box_price != 0]  # remove all the canceled orders
    _grp['order_rank'] = _grp['request_date'].rank(method='first', ascending=1)  # rank the orders again
    foreword_order_lag = _grp.order_lag.iloc[1:, ]
    if foreword_order_lag.empty:
        foreword_order_lag = [np.nan]
    _grp = _grp.iloc[:-1,]
    _grp['order_lag'] = foreword_order_lag
    train_data = train_data.append(_grp, ignore_index=True)

train_data.to_csv('/Users/saransh/Desktop/practicum/data/user_order_train.csv',index=False)

bins = range(0, 801, 50)
train_data['box_price_range'] = pd.cut(
    train_data.box_price, bins, right=False, include_lowest=True
)
axs: Axes = filter_based_on_ranges(train_data, 'order_lag', upper_bound=400).boxplot(
    'order_lag', by='box_price_range', notch=True, vert=False, bootstrap=1000
)
axs.set_xlabel('')
axs.set_ylabel('Box price range')
plt.show()

bins = list(range(0, 100, 7))
train_data['req_ship_lag_bins'] = pd.cut(train_data.req_ship_lag, bins, right=False, include_lowest=True)
axs: Axes = filter_based_on_ranges(
    train_data, 'order_lag', upper_bound=400
).boxplot('order_lag', by='req_ship_lag_bins', vert=False, notch=True)
axs.set_xlabel('Order lag')
axs.set_ylabel('Lag between box request and ship date')
axs.set_title('Impact on days to purchase due to lag b/w request and ship date')
plt.show()
