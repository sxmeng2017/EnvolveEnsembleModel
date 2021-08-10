from layers import *
from sko.GA import GA

class Envolve:
    """
    Input:
    train_X:Input Data applied in train
    train_y:Target of Input Data applied in train
    test_X:Input Data applied in test
    test_y:Target of Input Data applied in test
    """
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.layers = Layers(train_X, train_y, test_X, test_y)
        self.gene_length = 0

        self.output_model = None
        self.metric = None

    def add(self, model_list):
        self.layers.add(model_list)
        self.gene_length += len(model_list)

    def getMetaData(self, p):
        meta_train, meta_test = self.layers.getMetaData(p)
        return meta_train, meta_test

    def SetOutputModel(self, M):
        self.output_model = M

    def SetMetric(self, metric):
        self.metric = metric

    def _envolve(self, p):
        meta_train, meta_test = self.getMetaData(p)
        self.output_model.fit(meta_train, self.train_y)
        model_predict = self.output_model.predict(meta_test)

        loss = self.metric(model_predict, self.test_y)
        return loss

    def envolve(self, size_pop=50, max_iter=8):
        """
        :param size_pop: size of population
        :param max_iter: maximum iterations
        :return: best combination of ensemble and its result
        """
        if self.metric is None:
            raise ValueError('You should set a metric, such as mae')
        if self.output_model is None:
            raise ValueError("You need a ultimate model for ensemble")
        lb = np.zeros((self.gene_length,))
        ub = np.ones((self.gene_length,))
        ga = GA(func=self._envolve, n_dim=self.gene_length, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub,
                precision=ub)
        best_x, best_y = ga.run()
        print('best_combination:', best_x, '\n', 'best_result:', best_y)
        return best_x, best_y