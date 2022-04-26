from sko.PSO import PSO


def func(x):
    x1, x2 = x
    return x1 * (x1 - 1) + x2 * (x2 - 2)

pso = PSO(func=func, n_dim=2, pop=100000, max_iter=1000, lb=[-100, -100], ub=[100, 100], w=0.5, c1=2.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)