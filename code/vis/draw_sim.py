import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log_path = 'path to log file contains the similarity matrix'


def plot_log():
    clsses = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8',
              'client9']
    sorted_files = sorted(Path(log_path).iterdir(), key=os.path.getmtime)
    c = 0
    for logfile in sorted_files:
        logfile = logfile.name
        arr = np.around(np.load(log_path + logfile) * 100, decimals=1)
        plot_confusion_matrix(arr, clsses, 'similarity', c)
        c += 1


def plot_confusion_matrix(cm, classes, name, filename):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    normalize = False
    title = 'Confusion Matrix {}'.format(name)
    cmap = plt.cm.hot
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
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
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('SM_{}{}.png'.format(name, filename), figsize=(1000, 1000), dpi=100)
    plt.clf()


if __name__ == '__main__':
    plot_log()
