import numpy as np
from sklearn.model_selection import KFold


class Layers:

    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.manager = []

    def add(self, C):
        self.manager.append(C)

    def getMetaData(self, p):
        layers_num = len(self.manager)
        models_index_start = 0
        models_index_end = 0
        train_X, train_y, test_X, test_y = self.train_X, self.train_y, self.test_X, self.test_y
        for i in range(layers_num):
            bc = Combination(train_X, train_y, test_X, test_y)
            bc.getModel(self.manager[i])
            models_index_start = models_index_end
            models_index_end += len(bc.Models)
            meta_train, meta_test = bc.getMetaData(p[models_index_start:models_index_end])
            train_X, test_X = meta_train, meta_test

        return train_X, test_X

class Combination:

    def __init__(self, train_X, train_y, test_X, test_y):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.KF = None

        self.Models = []
        self._Models = {}

        self.meta_train_X = None
        self.meta_test_X = None

    def getKF(self, n_folds):
        self.KF = KFold(n_splits=n_folds)
        return self.KF

    def getStacking(self, clf, n_folds):
        train_num, test_num = self.train_X.shape[0], self.test_X.shape[0]
        second_level_train_set = np.zeros((train_num,))
        second_level_test_set = np.zeros((test_num,))
        test_nfolds_sets = np.zeros((test_num, n_folds))
        kf = self.getKF(n_folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.train_X)):
            x_tra, y_tra = self.train_X[train_index], self.train_y[train_index]
            x_tst, y_tst = self.train_X[test_index], self.train_y[test_index]

            clf.fit(x_tra, y_tra)

            second_level_train_set[test_index] = clf.predict(x_tst)
            test_nfolds_sets[:, i] = clf.predict(self.test_X)

        second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
        return second_level_train_set, second_level_test_set

    def getModel(self, Models):
        self.Models = Models
        for index, m in enumerate(Models):
            self._Models[index] = m

    def _getIndex(self, p):
        res = []
        for index, i in enumerate(p):
            if np.round(i) == 1:
                res.append(index)
        return res

    def _getModelList(self, p):
        index = self._getIndex(p)
        model_list = []
        for i in index:
            model_list.append(self._Models[i])
        return model_list

    def getMetaData(self, p, n_folds=5):
        model_list = self._getModelList(p)
        if len(model_list) == 0:
            return self.train_X, self.test_X
        train_x, test_x, train_y, test_y = self.train_X, self.test_X, self.train_y, self.test_y

        train_sets = []
        test_sets = []
        for clf in model_list:
            train_set, test_set = self.getStacking(clf, n_folds)
            train_sets.append(train_set)
            test_sets.append(test_set)

        meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
        meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

        self.meta_train_X = meta_train
        self.meta_test_X = meta_test

        return meta_train, meta_test