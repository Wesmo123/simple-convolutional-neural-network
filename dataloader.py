import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# !!preparing data!!

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features, features is the actual data of a dataset, represented by x
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels, labels are the classes of the dataset, represented by y

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features, features is the actual data of a dataset, represented by x
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels, labels are the classes of the dataset, represented by y

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# load data