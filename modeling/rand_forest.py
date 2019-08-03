import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
from matplotlib import pyplot as plt


data = pd.read_parquet("/Users/saransh/Desktop/practicum/data/lagonetrain_lc.parquet")
pandarallel.initialize()

transaction_columns = ['shipped_price', 'kept_price', 'items_count', 'box_price', 'kept_count', 'order_rank',
     'num_child', 'request_ship_diff', 'keep_rate', 'cl_labels']

y_col = ['order_lag']
bookkeeping_cols = ['orders_id', 'user_id']
message_columns = []
for col in ['pos_sent', 'neg_sent', 'neu_sent', 'sent_score', 'msg_count', 'total_pos', 'total_neg', 'total_neu',
            'topic']:
    message_columns.append(f"customer_{col}")
    message_columns.append(f"stylist_{col}")

message_columns = message_columns + [f"custvec_{col}" for col in range(100)]
message_columns = message_columns + [f"stvec_{col}" for col in range(100)]

data = data[transaction_columns + message_columns + y_col + bookkeeping_cols]

data['cl_labels'] = data['cl_labels'].astype('category')
data['customer_topic'] = data['customer_topic'].astype('category')
data['stylist_topic'] = data['stylist_topic'].astype('category')

X = data[transaction_columns + message_columns]
data['repurchase'] = data['order_lag'].apply(lambda r: 1 if r <= 30 else 0)
data['repurchase'] = data['repurchase'].astype('category')
data['repurchase'].value_counts()

y = data['repurchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

forest_clssifier = RandomForestClassifier(n_estimators=100, random_state=60616, max_depth=7)
forest_clssifier.fit(X_train, y_train)

forest_clssifier.score(X_test, y_test)
forest_clssifier.score(X_train, y_train)
y_test_probs = forest_clssifier.predict_proba(X_test)
y_test_scores = forest_clssifier.predict(X_test)
features = pd.DataFrame()
features['col'] = list(X.columns)
features['imp'] = forest_clssifier.feature_importances_

features.sort_values('imp', inplace=True, ascending=False)
confusion_matrix(y_test, y_test_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_test_probs.transpose()[1])

no_skill = pd.DataFrame()
no_skill['x'] = [0, 1]
no_skill['y'] = [0, 1]

plt.figure()
axs = plt.gca()
axs.plot(fpr, tpr, 'b--', label="Random forest model")
axs.plot(no_skill['x'], no_skill['y'], 'r--', label="No skill model")
axs.set_xlabel("False positive rate")
axs.set_ylabel("True positive rate")
axs.set_title("ROC curve")
axs.legend()
plt.show()

bg_color = 'white'
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(20.0, 10.0),
            'xtick.labelsize':25,
            'font.size':20,
            'ytick.labelsize':20})

dd = features.iloc[:20]
p: Axes = sns.barplot('col', 'imp', data=dd)
p.set_xticklabels(p.get_xticklabels(), rotation=90)
p.set_ylabel("Feature Importance")
p.set_xlabel("Features")
plt.show()