import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tp4.ej2.kohonen import Kohonen
from tp4.ej2.k_means import KMeans
from tp4.ej2.hierarchical_clustering import HierarchicalClustering

# returns pandas df
# attributes: list of attributes to remove
# scaler: sklearn scaler to fit and transform data
def load_data(filename, attributes=None):
    df = pd.read_csv(filename, delimiter=';')
    if attributes:
        df = df[attributes]
    return df


# inplace replaces nan with most frequent value for each column
def replace_nan(df: pd.DataFrame):
    nan_counts = []
    columns = df.columns
    for col in columns:
        nan_counts.append((col, df[col].isna().sum()))
        frequent = df[col].mode()
        df[col] = df[col].fillna(float(frequent))
    return nan_counts

def scale_data(df):
    # TODO: justify why this one
    scaler = StandardScaler()
    headers = df.columns
    np_df = df.values
    scaled = scaler.fit_transform(np_df)
    return scaler, pd.DataFrame(scaled, columns=headers)

# ************ MAIN PROGRAM ************ #

attributes = ['sex', 'age', 'cad.dur', 'choleste']
label = ['sigdz']

dataframe = load_data('../data/acath.csv', attributes + label)
info = replace_nan(dataframe)

print('Nan values per column in original dataframe')
print(info)

data = dataframe[attributes]
labels = dataframe[label]
std_scaler, data = scale_data(data)

train, test, train_y, test_y = train_test_split(data, labels, test_size=0.97, random_state=42) #TODO: change 0.9
print(f'{len(train)} train records and {len(test)} test records')

# Hierarchical clustering, TRAIN.
hc = HierarchicalClustering(np.array(train), k=2)
hc.run()
hc.add_cluster_classification(np.array(train_y))
# there is the possibility of both having the same label. For now, we are leaving it that way
# TODO: discuss.
print(f'Classes for HC are {hc.cluster_classifications} for clusters {hc.clusters}')

# Hierarchical clustering, TEST.
predictions = hc.predict(np.array(test))
# TODO: metrics to evaluate

# feats = 1
# k = 10
# eta = 0.1
# epochs = 100
# data, y = make_blobs(n_samples=1000, n_features=feats, centers=3)
# som = Kohonen(k, k, eta, feats, data, rand_weights=False)
# som.train(epochs)
# som.u_matrix('u_matrix.png')
# som.activations('activations.png')

