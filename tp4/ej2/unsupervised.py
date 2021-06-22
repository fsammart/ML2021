import numpy as np
from sklearn.model_selection import train_test_split
from tp4.ej2.kohonen import Kohonen
from tp4.ej2.k_means import KMeans
from tp4.ej2.hierarchical_clustering import HierarchicalClustering
from tp4.ej2.preprocessing import load_data, replace_nan, scale_data
from tp4.ej2.utils import get_confusion_matrix, get_precision, plot_heatmap


def hierarchical_clustering_program(train, test, train_y, test_y, filename):
    # train section
    hc = HierarchicalClustering(np.array(train), k=10)
    hc.run()
    hc.add_cluster_classification(np.array(train_y))
    # there is the possibility of both having the same label. For now, we are leaving it that way
    # TODO: discuss.
    print(f'Classes for HC are {hc.cluster_classifications} for clusters {hc.clusters}')

    # test section
    predictions = hc.predict(np.array(test))
    # TODO: metrics to evaluate


def k_means_program(train, test, train_y, test_y, filename):

    clusters_group = [50, 40, 30, 20, 10, 5, 2]

    for clusters in clusters_group:
        # train section
        k_means = KMeans(clusters, train)
        k_means.run(4000)
        k_means.add_cluster_classification(train_y)
        print(f'Classes for KMeans are {k_means.cluster_classifications} for clusters {k_means.clusters}')
        k_means.plot_variances(f'{filename}_variances_per_epoch.jpg')

        # test section
        predictions = k_means.predict(test)

        confusion_matrix = get_confusion_matrix(predictions, test_y)
        precision = get_precision(confusion_matrix)
        plot_heatmap(confusion_matrix, f'./results/{filename}_{clusters}_clusters_confusion.jpg')
        print(f'Precision for {filename} with {clusters} clusters is {round(precision, 4)}')


def kohonen_program(train, test, train_y, test_y, filename):
    # feats = 1
    # k = 10
    # eta = 0.1
    # epochs = 100
    # data, y = make_blobs(n_samples=1000, n_features=feats, centers=3)
    # som = Kohonen(k, k, eta, feats, data, rand_weights=False)
    # som.train(epochs)
    # som.u_matrix('u_matrix.png')
    # som.activations('activations.png')
    pass


def run_program(kohonen=False, k_means=False, hc=False, balance=None, remove_duplicates=False):
    attributes = ['age', 'cad.dur', 'choleste']
    label = ['sigdz']

    dataframe, dataframe_info = load_data('../data/acath.csv', attributes + label, balance=balance)
    if remove_duplicates:
        with_removed = dataframe.drop_duplicates(subset=attributes)
        print(f'Removed {len(dataframe)-len(with_removed)} duplicates from dataframe.')
        dataframe = with_removed
    print(dataframe_info)

    replacement_info = replace_nan(dataframe, differentiate_label=True)
    print('Nan values per column in original dataframe')
    print(replacement_info)

    data = dataframe[attributes]
    labels = dataframe[label]
    std_scaler, data = scale_data(data)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    print(f'{len(data)} registers > {len(x_train)} train records and {len(x_test)} test records')

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy().flatten()

    balance_suffix = '' if balance is None else f'_{balance}'
    removed_suffix = '' if remove_duplicates is False else f'_no_duplicates'

    if hc:
        hierarchical_clustering_program(x_train, x_test, y_train, y_test, f'hc{balance_suffix}{removed_suffix}')
    if k_means:
        k_means_program(x_train, x_test, y_train, y_test, f'k_means{balance_suffix}{removed_suffix}')
    if kohonen:
        kohonen_program(x_train, x_test, y_train, y_test, f'kohonen{balance_suffix}{removed_suffix}')

    print('\n')


if __name__ == '__main__':
    run_program(k_means=False, kohonen=False, hc=False, remove_duplicates=True)
    # print("\n*** NO DUPLICATES DATA ****\n")
    # run_program(k_means=True, kohonen=False, hc=False, remove_duplicates=True)
    # print("*** RAW DATA ****\n")
    # run_program(k_means=True, kohonen=False, hc=False, balance=None)
    # print("\n*** BALANCED DATA ****\n")
    # run_program(k_means=True, kohonen=False, hc=False, balance='sigdz')
