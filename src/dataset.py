import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import os
import urllib
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "a1a"  : "a1a",
                      "a2a"  : "a2a",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}
                    
def load_cifar10(train=False): # cifar10
    transform_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
    return CIFAR10('dataset/cifar10_py/',train=train,transform=transform_cifar10,download=True)

def load_cifar100(train=False): # cifar100
    transform_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
    return CIFAR100('dataset', train=train, transform=transform_cifar100, download=True)

def load_svm_dataset(dataset_name, train, datadir='dataset/svm_dataset/'): # mushrooms, ijcnn, rcv1, w8a
    assert dataset_name in ["mushrooms", "w8a",
                        "rcv1", "ijcnn", 'a1a','a2a',
                        "mushrooms_convex", "w8a_convex",
                        "rcv1_convex", "ijcnn_convex", 'a1a_convex'
                        , 'a2a_convex']
    sigma_dict = {"mushrooms": 0.5,
                    "w8a":20.0,
                    "rcv1":0.25 ,
                    "ijcnn":0.05}

    X, y = load_libsvm(dataset_name.replace('_convex', ''), 
                        data_dir=datadir)

    labels = np.unique(y)

    y[y==labels[0]] = 0
    y[y==labels[1]] = 1
    # splits used in experiments
    splits = train_test_split(X, y, test_size=0.2, shuffle=True, 
                random_state=9513451)
    X_train, X_test, Y_train, Y_test = splits

    if "_convex" in dataset_name:
        if train:
            # training set
            X_train = torch.FloatTensor(X_train.toarray())
            Y_train = torch.FloatTensor(Y_train)
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        else:
            # test set
            X_test = torch.FloatTensor(X_test.toarray())
            Y_test = torch.FloatTensor(Y_test)
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)

        # return DatasetWrapper(dataset, split=split)

    if train:
        # fname_rbf = "%s/rbf_%s_%s_train.pkl" % (datadir, dataset_name, sigma_dict[dataset_name])
        fname_rbf = "%s/rbf_%s_%s_train.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
        if os.path.exists(fname_rbf):
            k_train_X = np.load(fname_rbf)
        else:
            k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
            np.save(fname_rbf, k_train_X)
            print('%s saved' % fname_rbf)

        X_train = k_train_X
        X_train = torch.FloatTensor(X_train)
        Y_train = torch.LongTensor(Y_train)

        dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    else:
        fname_rbf = "%s/rbf_%s_%s_test.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
        if os.path.exists(fname_rbf):
            k_test_X = np.load(fname_rbf)
        else:
            k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
            np.save(fname_rbf, k_test_X)
            print('%s saved' % fname_rbf)

        X_test = k_test_X
        X_test = torch.FloatTensor(X_test)
        Y_test = torch.LongTensor(Y_test)

        dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    return dataset


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y

def rbf_kernel(A, B, sigma):
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K

def load_matrix_fac(train=True, datadir='dataset'):
    fname = datadir + '/matrix_fac.pkl'
    if not os.path.exists(fname):
        data = generate_synthetic_matrix_factorization_data()
        # print(data)
        with open(fname, "wb") as f:
            pickle.dump(data, f)
            f.close()
        os.rename(fname, fname)
        # ut.save_pkl(fname, data)
    with open(fname, "rb") as f:
        A, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=9513451)

    training_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    test_set = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

    if train:
        dataset = training_set
    else:
        dataset = test_set
    return DatasetWrapper(dataset, split='train' if train else 'test')


def generate_synthetic_matrix_factorization_data(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
    """
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
    See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    """
    Atrue = np.linspace(1, A_condition_number, ydim
       ).reshape(-1, 1) * np.random.rand(ydim, xdim)
    # the inputs
    X = np.random.randn(xdim, nsamples)
    # the y's to fit
    Ytrue = Atrue.dot(X)
    data = (X.T, Ytrue.T)
    # print(data)
    return data

class DatasetWrapper:
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]


        return data, target