import numpy as np
from sko.GA import GA

def get_fitness(algorithm):
    algorithm.FitV = np.argsort(np.argsort(algorithm.Y))
    return algorithm.FitV

demo_func = lambda x: x[0] + 10* np.sin(5*x[0])+ 7*np.cos(4*x[0])
ga = GA(func=demo_func, n_dim=3, size_pop=64, max_iter=100000, prob_mut=0.001,
        lb=[-10, -10, -10], ub=[10, 10, 10], precision=[1e-7, 1e-7, 1])

ga.register(operator_name='ranking', operator=get_fitness)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)