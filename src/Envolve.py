from layers import *
from sko.GA import GA
import time
import deap
from deap import tools
from deap import base, creator
from scipy.stats import *
toolbox = base.Toolbox()
GENE_LENGTH = 5

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

    def GARecord(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # weights 1.0, 求最大值,-1.0 求最小值
        # (1.0,-1.0,)求第一个参数的最大值,求第二个参数的最小值
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialization
        import random
        from deap import tools

        IND_SIZE = 10  # 种群数

        toolbox = base.Toolbox()
        toolbox.register('Binary', bernoulli.rvs, 0.5)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
        # 用tools.initRepeat生成长度为GENE_LENGTH的Individual
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        print(toolbox.population(n=2))

        # Operators
        # difine evaluate function
        # Note that a comma is a must
        def evaluate(individual):
            loss = self._envolve(individual)
            return loss,

        # use tools in deap to creat our application
        toolbox.register("mate", tools.cxTwoPoint)  # mate:交叉
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # mutate : 变异
        toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
        toolbox.register("evaluate", evaluate)  # commit our evaluate
        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=50)
        record = {}
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        '''
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        #
        # NGEN  is the number of generations for which the
        #       evolution runs
        '''

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))
        print("-- Iterative %i times --" % NGEN)

        for g in range(NGEN):
            if g % 10 == 0:
                print("-- Generation %i --" % g)
            t1 = time.time()
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Change map to list,The documentation on the official website is wrong

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            best_ = tools.selBest(pop, 1)[0]
            t2 = time.time()
            record[g] = (best_, best_.fitness.values, t2 - t1)
            # The population is entirely replaced by the offspring
            pop[:] = offspring

        print("-- End of (successful) evolution --")

        #best_ind = tools.selBest(pop, 1)[0]

        return record