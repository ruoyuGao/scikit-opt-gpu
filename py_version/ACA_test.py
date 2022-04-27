import numpy as np
from scipy import spatial
from ACA import ACA_TSP
import sys

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

num_ants = int(sys.argv[1])
max_interations = int(sys.argv[2])
textfile = sys.argv[3]

dataset = []

fp = open(textfile, 'r')

line = fp.readline()
while line:
    no, x, y = line.split()
    x = float(x)
    y = float(y)
    dataset.append(x)
    dataset.append(y)
    line = fp.readline()

points_coordinate = np.array(dataset)
points_coordinate = points_coordinate.reshape(-1, 2)


print('%d ants' % num_ants)
print('%d cities' % points_coordinate.shape[0])
print('%d iterations' % max_interations)


#points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

aca = ACA_TSP(func=cal_total_distance, n_dim=points_coordinate.shape[0],
              size_pop=num_ants, max_iter=max_interations,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

sum = 0

for i in range(best_x.shape[0]):
    n1 = best_x[i]
    n2 = best_x[(i + 1) % best_x.shape[0]]
    sum += distance_matrix[n1, n2]

print('route: ')
print(best_x)
print('total distance: %f' % sum)


