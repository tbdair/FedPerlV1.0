import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

log_path = '/home/tariq/code/isic2019/N8logs/nyp_done/'



def plot_log():
    metric = []
    y = []
    for i in range(500):
        y.append(i)

    for logfile in os.listdir(log_path):
        if 'peers' in logfile:
            arr = np.load(log_path + logfile, allow_pickle=True)
        else:
            arr = np.load(log_path + logfile)

        logfile = logfile.replace('.npy', '')
        name = logfile.split("_")[2]
        name += 'PA' if 'avgTrue' in logfile else ''
        file_name = name  # +'_'+logfile.split("_")[1]

        if 'peers' in logfile:
            d = dict(enumerate(arr.flatten(), 1))
            clsses = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8',
                      'client9']
            percent = np.zeros((10, 10))

            for i in range(10):
                # print(i, np.unique(d[1]['client'+str(i)], return_counts=True))
                clnt, cont = np.unique(d[1]['client' + str(i)], return_counts=True)
                total = np.sum(cont)
                cont = cont / total
                cont = cont * 100
                cont = np.around(cont, decimals=1)
                for c in range(len(clnt)):
                    percent[i, clnt[c]] = cont[c]

            plot_confusion_matrix(percent, clsses, name, file_name)


def plot_confusion_matrix(cm, classes, name, filename):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    normalize = False
    title = 'Confusion Matrix {}'.format(name)
    cmap = plt.cm.hot
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=35)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="grey" if cm[i, j] > thresh else "grey")

    plt.tight_layout()
    # plt.ylabel('Clients')
    # plt.xlabel('Clients')
    plt.savefig('Peers{}.png'.format(filename), figsize=(1000, 1000), dpi=100)
    plt.clf()


if __name__ == '__main__':
    plot_log()
