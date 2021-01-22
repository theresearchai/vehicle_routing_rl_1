import numpy as np
from scipy.spatial import distance_matrix

def get_num_points(config):
    if config.test_model is None:
        return config.num_training_points
    else:
        return config.num_test_points

def get_random_capacities(n):
    capacities = np.random.randint(9, size=n) + 1
    depot_capacity_map = {
        10: 20,
        20: 30,
        50: 40,
        100: 50
    }
    capacities[0] = depot_capacity_map.get(n - 1, 50)
    return capacities

class Problem:
    def __init__(self, locations, capacities):
        self.locations = locations.copy()
        self.capacities = capacities.copy()
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.total_customer_capacities = np.sum(capacities[1:])
        self.change_at = np.zeros([len(self.locations) + 1])
        self.no_improvement_at = {}
        self.num_solutions = 0
        self.num_traversed = np.zeros((len(locations), len(locations)))
        self.distance_hashes = set()

    def record_solution(self, solution, distance):
        inv_dist = 1.0 / distance
        self.num_solutions += inv_dist
        for path in solution:
            if len(path) > 2:
                for to_index in range(1, len(path)):
                    self.num_traversed[path[to_index - 1]][path[to_index]] += inv_dist
                    self.num_traversed[path[to_index]][path[to_index - 1]] += inv_dist

    def add_distance_hash(self, distance_hash):
        self.distance_hashes.add(distance_hash)

    def get_location(self, index):
        return self.locations[index]

    def get_capacity(self, index):
        return self.capacities[index]

    def get_num_customers(self):
        return len(self.locations) - 1

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]

    def get_frequency(self, from_index, to_index):
        return self.num_traversed[from_index][to_index] / (1.0 + self.num_solutions)

    def reset_change_at_and_no_improvement_at(self):
        self.change_at = np.zeros([len(self.locations) + 1])
        self.no_improvement_at = {}

    def mark_change_at(self, step, path_indices):
        for path_index in path_indices:
            self.change_at[path_index] = step

    def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        self.no_improvement_at[key] = step

    def should_try(self, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        no_improvement_at = self.no_improvement_at.get(key, -1)
        return self.change_at[index_first] >= no_improvement_at or \
               self.change_at[index_second] >= no_improvement_at or \
               self.change_at[index_third] >= no_improvement_at

def generate_problem(config):
    np.random.seed(config.problem_seed)
    config.problem_seed += 1

    num_sample_points = get_num_points(config) + 1
    locations = np.random.uniform(size=(num_sample_points, 2))
    capacities = get_random_capacities(num_sample_points)
    problem = Problem(locations, capacities)
    np.random.seed(config.problem_seed * 10)
    return problem
