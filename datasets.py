
import torch
import numpy as np


def load_dataset(path, reps, cuda, sample):
    path_data = path
    replications = reps
    sample_size = sample
    # which features are binary
    binfeats = []
    # which features are continuous
    contfeats = [i for i in range(6) if i not in binfeats]

    data = np.loadtxt(path_data + '/Syn_train' + str(sample_size) + '_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t, y = data[:, 6], data[:, 7][:, np.newaxis]
    mu_0, mu_1, x = data[:, 7][:, np.newaxis], data[:, 8][:, np.newaxis], data[:, 0:6]
    true_ite = mu_1 - mu_0
    # perm = binfeats + contfeats
    # x = x[:, perm]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
    train = (x, t, y), true_ite

    data_test = np.loadtxt(path_data + '/Syn_test' + str(sample_size) + '_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t_test, y_test = data_test[:, 6][:, np.newaxis], data_test[:, 7][:, np.newaxis]
    mu_0_test, mu_1_test, x_test = data_test[:, 7][:, np.newaxis], data_test[:, 8][:, np.newaxis], data_test[:, 0:6]

    # x_test = x_test[:, perm]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).squeeze()
    t_test = torch.from_numpy(t_test).squeeze()
    if cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        t_test = t_test.cuda()
    true_ite_test = mu_1_test - mu_0_test
    test = (x_test, t_test, y_test), true_ite_test
    return train, test, contfeats, binfeats

