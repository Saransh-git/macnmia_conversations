from concurrent.futures import ThreadPoolExecutor
from functools import partial

import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandarallel import pandarallel
from pandas import DataFrame
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph

from wrangling.utils import convert_to_python_datetime, convert_datetime_cols_to_python_datetime, \
    convert_timedelata_to_days, filter_based_on_ranges

pandarallel.initialize()
train_data = pd.read_csv('/Users/saransh/Desktop/practicum/data/user_order_train.csv')

transactions_type_mapping = dict(
    orders_id=np.int64,
    user_id=np.int64,
    items_count=np.int64,
    kept_count=np.int64,
    order_rank=np.int64,
    num_child=np.int64
)
train_data = train_data[~train_data.user_id.isna()]  # orders not corresponding to any user
train_data = train_data[~train_data.ship_date.isna()]  # orders not shipped
train_data = train_data[~train_data.stylist_id.isna()]  # no stylists assigned
for col, type in transactions_type_mapping.items():
    train_data[col] = train_data[col].astype(type)  # transform dataframe cols to appropriate types

# convert datetime cols to appropriate types
datetime_conversion_func = partial(convert_to_python_datetime, format_str='%Y-%m-%d')
convert_datetime_cols_to_python_datetime(train_data, conversion_func=datetime_conversion_func)

train_data = train_data.set_index(['user_id', 'orders_id'])
train_data['request_ship_diff'] = train_data.ship_date - train_data.request_date
train_data['keep_rate'] = train_data['kept_count']/train_data['items_count']
train_data['request_ship_diff'] = train_data['request_ship_diff'].apply(convert_timedelata_to_days)
Y = train_data['order_lag']
X: DataFrame = train_data[
    ['shipped_price', 'kept_price', 'items_count', 'box_price', 'kept_count', 'order_rank',
     'num_child', 'request_ship_diff']
]


X_ = X - X.mean(axis=0)  # center
X_ /= X.std(axis=0)  # scale

pca = PCA(svd_solver='full')

'''
pca_scores = []
for n in range(0,9):
    pca = PCA(svd_solver='full', n_components=n)
    pca_scores.append(np.mean(cross_val_score(pca, X_, cv=5)))
'''
X_r = pca.fit(X_).transform(X_)
p_comps = pd.DataFrame(X_r)

plt.figure()
axs: Axes = plt.gca()
axs.scatter(p_comps[0], p_comps[1], marker='.', alpha=0.04)
axs.set_title("Biplot")
axs.set_xlabel("PC1")
axs.set_ylabel("PC2")
plt.show()


plt.figure()
axs: Axes = plt.gca()
axs.plot(range(X_.shape[1]), pca.explained_variance_ratio_, 'bo-')
axs.axhline(y=1/(X_.shape[1]), linestyle='dashed', color='red', label='Average variance per component')
axs.legend()
axs.set_xlabel("Principal components")
axs.set_ylabel("Relative variance explained")
plt.show()

p_comps = p_comps.iloc[:, :5]
silhouettes = []
for n_cluster in range(2,8):
    print(f"Starting with cluster size {n_cluster}")
    sil_scores = []
    futures = []
    executor = ThreadPoolExecutor(max_workers=6)
    cl = partial(AgglomerativeClustering, n_clusters=n_cluster, linkage='ward')
    for conn in [10, 15, 20, 30, 40, 50]:
        connectivity = kneighbors_graph(p_comps, n_neighbors=conn, include_self=False)
        cl_ = cl(connectivity=connectivity)
        futures.append(executor.submit(cl_.fit, p_comps))
    executor.shutdown()
    for ft in futures:
        sil_scores.append(silhouette_score(p_comps, ft.result().labels_))
    silhouettes.append(np.mean(sil_scores))


executor = ThreadPoolExecutor(max_workers=4)
futures = []
silhouettes = []
connectivity = kneighbors_graph(p_comps, n_neighbors=50, include_self=False)
cl = partial(AgglomerativeClustering, connectivity=connectivity, linkage='ward')
for n_cluster in range(2,6):
    print(f"Starting with cluster size {n_cluster}")
    cl_ = cl(n_clusters=n_cluster)
    futures.append(executor.submit(cl_.fit, p_comps))

executor.shutdown()

executor = ThreadPoolExecutor(max_workers=4)
result_fts = []
for ft, n_cluster in zip(futures, range(2,6)):
    print(f"Computing Silhouette for {n_cluster}")
    result_fts.append(executor.submit(silhouette_score, p_comps, ft.result().labels_))

executor.shutdown()
for ft in result_fts:
    silhouettes.append(ft.result())

sil_15 = [0.18627172270800177, 0.14481988402718535, 0.14712724372975608, 0.14444048153087216]
sil_20 = [0.15975867873403884, 0.13524309234040838, 0.12276975129993783, 0.1257405254250921]
sil_25_30_and_50 = [0.13951766770474536, 0.13751989917711452, 0.14666958904917365, 0.14974942877045486]
avg_silhouettes_over_neighbors = [0.15450515839588114, 0.13880298045604902, 0.14054978298997323,
                                  0.14184250997247824, 0.12330216541761356, 0.11209507135789999]
plt.figure()
axs: Axes = plt.gca()
axs.plot(range(2,6), sil_25_30_and_50, 'bo-')
axs.set_xlabel("Cluster sizes")
axs.set_ylabel("Silhouette index")
axs.set_title("Silhouette metric")
plt.show()

connectivity = kneighbors_graph(p_comps, n_neighbors=50, include_self=False)
cl_ = cl(n_clusters=5).fit(p_comps)
plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("PC1")
axs.set_ylabel("PC2")

p_comps_grp = p_comps.groupby(cl_.labels_)
for key, grp in p_comps_grp:
    axs.scatter(grp[0], grp[1], marker='.', label=key, alpha=0.04)

axs.legend()
plt.show()

plt.figure()
axs: Axes = plt.gca()
axs.set_xlabel("PC1")
axs.set_ylabel("PC2")
axs.set_title("Density biplot")
p_comps_grp = p_comps.groupby(cl_.labels_)
for key, grp in p_comps_grp:
    axs.scatter(grp[0], grp[1], marker ='.', label=key)

axs.legend()
plt.show()

# cl_ = KMeans(n_clusters=5, random_state=60616).fit(p_comps)
# create a decision tree to profile these clusters
train_data['cl_labels'] = cl_.labels_
train_data['cl_labels'].value_counts()
train_data.groupby('cl_labels').order_lag.mean()

filter_based_on_ranges(train_data, 'order_lag', upper_bound=400).boxplot(
    "order_lag", by='cl_labels', vert=False, showmeans=True)
train_data.boxplot(
    "order_lag", by='cl_labels', vert=False, showmeans=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, cl_.labels_, train_size=0.8, test_size=0.2, random_state=60616, stratify=cl_.labels_
)

trees = []
tree_acc_scores_validation = []
tree_acc_std_errors = []
tree_classifier = partial(
    tree.DecisionTreeClassifier, criterion="entropy", random_state=60616, class_weight="balanced"
)


for tree_depth in range(2,25):
    trees.append(
        tree_classifier(max_depth=tree_depth)
    )

for t in trees:
    scores = cross_val_score(t, X_train, y_train, cv=5)
    tree_acc_scores_validation.append(scores.mean())
    tree_acc_std_errors.append(scores.std())

plt.figure()
axs: Axes = plt.gca()
axs.plot(range(2, 25), tree_acc_scores_validation, 'bo--')
axs.set_xlabel("Tree depth")
axs.set_ylabel("Validation set accuracy")
axs.set_xticks(list(range(2, 25)))
axs.set_title("Tree pruning/ Hyperparameter selection")
plt.show()


t = tree_classifier(max_depth=4).fit(X_train, y_train)
t.score(X_test, y_test)
t.feature_importances_


axs: Axes = train_data.boxplot('order_rank', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Order rank")

axs: Axes = train_data[(train_data.request_ship_diff < 50) & (train_data.request_ship_diff > 0)].boxplot('request_ship_diff', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Lag b/w request date and ship date")

axs: Axes = train_data.boxplot('kept_count', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Kept count")

axs: Axes = train_data.boxplot('keep_rate', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Keep rate")

axs: Axes = train_data.boxplot('box_price', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Box price")

axs: Axes = train_data.boxplot('num_child', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_ylabel("Number of children")
axs.set_title("Boxplot grouped by cluster labels")

axs: Axes = train_data.boxplot('kept_price', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Kept price")
plt.show()


axs: Axes = train_data.boxplot('order_lag', by='cl_labels')
axs.set_xlabel("Cluster labels")
axs.set_title("Boxplot grouped by cluster labels")
axs.set_ylabel("Order lag")
plt.show()
dot_data = tree.export_graphviz(
    t, feature_names=X.columns, class_names=['0', '1', '2', '3', '4'], filled=True, rounded=True,
)

graph = graphviz.Source(dot_data)
graph.render("Cluster profiling", "visualizations", cleanup=True)
