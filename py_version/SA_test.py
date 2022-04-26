from sko.SA import SA


def func(x):
    x1, x2 = x
    return x1 * (x1 - 1) + x2 * (x2 - 2)


sa = SA(func=func, x0=[100, 100], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y', best_y)
