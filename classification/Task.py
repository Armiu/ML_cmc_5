import numpy as np


class Preprocesser:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocesser):

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.features = [np.unique(X[col]) for col in X.columns]

    def transform(self, X):
        data = np.zeros((X.shape[0], sum(len(f) for f in self.features)), dtype=self.dtype)
        start = 0
        for i, col in enumerate(X.columns):
            feature = self.features[i]
            for j, val in enumerate(feature):
                data[:, start + j] = (X[col] == val).astype(int)
            start += len(feature)
        return data

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}

class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.success = {
            (j, el[0]): [np.mean(Y[np.array(X)[:, ind] == j]), len(Y[np.array(X)[:, ind] == j]) / len(Y)]
            for ind, el in enumerate([[column, sorted({*X[column].tolist()})] for column in X.columns.tolist()])
            for j in el[1]
        }

    def transform(self, X, a=1e-5, b=1e-5):
        return np.column_stack([
            X[i].apply(lambda x: [*(z := self.success[(x, i)]), (z[0] + a) / (z[1] + b)]).tolist()
            for i in X.columns.tolist()
        ])

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.folds_ = []
        self.fold_index = None
        self.successes = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds_ = list(group_k_fold(len(X), self.n_folds, seed))
        self.fold_index = np.empty(len(X), dtype=int)
        for count, (train_idx, _) in enumerate(self.folds_):
            self.fold_index[train_idx] = count

        self.successes = [None] * self.n_folds
        for k in range(self.n_folds):
            train_idx = self.folds_[k][1]
            x = X.iloc[train_idx].values
            y = Y.iloc[train_idx].values
            self.successes[k] = []
            for i in range(x.shape[1]):
                keys = np.unique(x[:, i])
                arr = x[:, i]
                d = {}
                for j in keys:
                    col2 = np.count_nonzero(arr == j) / x.shape[0]
                    arr_y = y[np.argwhere(arr == j)]
                    d[j] = (np.mean(arr_y), col2)
                self.successes[k].append(d)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        x = X.values
        ret = np.empty((x.shape[0], 0))
        for i in range(x.shape[1]):
            col1 = np.empty((x.shape[0], 1))
            for j in range(x.shape[0]):
                k = self.fold_index[j]
                col1[j] = self.successes[k][i][x[j, i]][0]
            ret = np.append(ret, col1, axis=1)
            col2 = np.empty((x.shape[0], 1))
            for j in range(x.shape[0]):
                k = self.fold_index[j]
                col2[j] = self.successes[k][i][x[j, i]][1]
            ret = np.append(ret, col2, axis=1)
            col3 = (col1 + a) / (col2 + b)
            ret = np.append(ret, col3, axis=1)
        return ret

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    weight = np.empty(np.unique(x).shape[0])
    for i in range(np.unique(x).shape[0]):
        A = y[np.argwhere(x == np.unique(x)[i])]
        weight[i] = np.sum(A) / A.shape[0]
    return weight
